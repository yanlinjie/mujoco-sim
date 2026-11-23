"""Prepare a Go2 scene from a map package and optionally validate its physics."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument("--robot-root", type=Path, default=repo)
    parser.add_argument("--map", default="room1110/2026-11-11.2")
    parser.add_argument("--output", type=Path, default=repo / "output/robot-mujoco-local")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    robot_root = args.robot_root.resolve()
    maps_root = robot_root / "slam_map_demo/maps"
    map_dir = (maps_root / args.map).resolve()
    if not map_dir.is_relative_to(maps_root.resolve()):
        parser.error("map must be inside slam_map_demo/maps")
    try:
        scene = prepare_scene(robot_root, map_dir, args.output.resolve())
        if args.validate:
            validate_physics(scene)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Cannot run robot MuJoCo: {exc}\n")


if __name__ == "__main__":
    main()
