"""Prepare a copied Go2 scene and run its existing loopback HTTP service."""

from __future__ import annotations

import argparse
import importlib.util
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


def prepare_scene(robot_root: Path, map_dir: Path, output: Path) -> Path:
    """Relocate generated assets without changing the copied robot repository."""
    meta = yaml.safe_load((map_dir / "meta.yaml").read_text())
    go2_dir = robot_root / "unitree_mujoco/unitree_robots/go2"
    source = go2_dir / f"scene_slam_{meta['map_id']}.xml"
    tree = ET.parse(source)
    for node in tree.getroot().iter():
        filename = node.get("file")
        if not filename:
            continue
        if node.tag == "include":
            target = (source.parent / filename).resolve()
        elif "/slam_map_demo/maps/" in filename:
            # Keep the resource suffix, selecting the requested local map version.
            suffix = filename.split("/slam_map_demo/maps/", 1)[1].split("/", 2)
            if len(suffix) != 3 or suffix[0] != meta["map_id"]:
                raise ValueError(f"unexpected scene asset: {filename}")
            target = (map_dir / suffix[2]).resolve()
            if not target.is_relative_to(map_dir):
                raise ValueError(f"asset escapes map directory: {filename}")
        else:
            continue
        if not target.is_file():
            raise FileNotFoundError(target)
        node.set("file", str(target))

    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    expected_assets = go2_dir / "assets"
    if assets.is_symlink() or assets.exists():
        if assets.resolve() != expected_assets.resolve():
            raise ValueError(f"refusing to replace existing assets: {assets}")
    else:
        assets.symlink_to(expected_assets, target_is_directory=True)
    destination = output / source.name
    tree.write(destination, encoding="unicode")
    print(f"scene: {destination}", flush=True)
    print(f"map: {meta['map_id']}/{meta['version']}", flush=True)
    return destination


def validate_physics(scene: Path) -> None:
    """Check scene loading and finite physics, without claiming navigation success."""
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(scene))
    if model.nu != 12:
        raise ValueError(f"expected 12 Go2 actuators, got {model.nu}")
    data = mujoco.MjData(model)
    for _ in range(100):
        mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise ValueError("physics state is not finite")
    print(
        f"MuJoCo {mujoco.__version__}: nq={model.nq}, nv={model.nv}, "
        f"nu={model.nu}, ngeom={model.ngeom}; 100 steps, t={data.time:.3f}s",
        flush=True,
    )


def serve(robot_root: Path, map_dir: Path, scene: Path, port: int, simulator_dir: Path) -> None:
    """Reuse sim_service.py and its DDS controller, leaving streaming disabled."""
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    binary = simulator_dir / "unitree_mujoco"
    check = subprocess.run(["ldd", str(binary)], capture_output=True, text=True)
    diagnostic = check.stdout + check.stderr
    if check.returncode or "not found" in diagnostic:
        raise RuntimeError(f"native simulator dependencies are unavailable:\n{diagnostic}")

    source = robot_root / "slam_map_demo/scripts/sim_service.py"
    spec = importlib.util.spec_from_file_location("robot_local_sim_service", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(service)
    upstream_class = service.SimService

    class LocalSimService(upstream_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Both the native simulator and the preview accept an absolute scene.
            self.scene_name = str(scene)
            self.scene_path = scene

    service.SimService = LocalSimService
    service.SIM_DIR = simulator_dir
    service.WORK = scene.parent / "work"
    sys.argv = [str(source), str(map_dir), "--host", "127.0.0.1",
                "--port", str(port), "--stream-port", "0"]
    service.main()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--robot-root", type=Path, default=repo)
    parser.add_argument("--map", default="family_house/v1")
    parser.add_argument("--output", type=Path, default=repo / "output/robot-mujoco-local")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--simulator-dir", type=Path)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    robot_root = args.robot_root.resolve()
    maps_root = robot_root / "slam_map_demo/maps"
    map_dir = (maps_root / args.map).resolve()
    if not map_dir.is_relative_to(maps_root.resolve()):
        parser.error("map must be inside slam_map_demo/maps")
    try:
        scene = prepare_scene(robot_root, map_dir, args.output.resolve())
        if args.validate:
            validate_physics(scene)
        if args.serve:
            simulator_dir = args.simulator_dir or (args.output / "native/build")
            serve(robot_root, map_dir, scene, args.port, simulator_dir.resolve())
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Cannot run robot MuJoCo: {exc}\n")


if __name__ == "__main__":
    main()
