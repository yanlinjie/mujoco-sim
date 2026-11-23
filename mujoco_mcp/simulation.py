"""Manage the existing simulator controller within the MCP process."""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.robot_mujoco_local import prepare_scene
from slam_map_demo.scripts.sim_service import SimService, processes

from .protocol import ServiceError, validate_pose


def _dds_domain(command, flags):
    arguments = command.split()
    for index, argument in enumerate(arguments):
        flag, equals, value = argument.partition("=")
        if flag in flags:
            try:
                return int(value if equals else arguments[index + 1])
            except (IndexError, ValueError):
                break
        if "-i" in flags and argument.startswith("-i") and argument[2:].isdigit():
            return int(argument[2:])
    return 1


class Simulation(SimService):
    """Keep ownership of the selected DDS domain and the native child process."""

    def __init__(self, robot_root, map_dir, output, simulator_dir, dds_domain=1):
        super().__init__(
            map_dir, sys.executable, work_dir=output / "work",
            simulator_dir=simulator_dir, dds_domain=dds_domain, subscribe=False,
        )
        self.robot_root = robot_root
        self.output = output
        self._lease = None
        self._subs = []

    def start(self):
        lease_path = Path(tempfile.gettempdir()) / f"unitree-mujoco-{os.getuid()}-{self.dds_domain}.lock"
        self._lease = lease_path.open("a")
        try:
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ServiceError(f"DDS domain {self.dds_domain} already has a MuJoCo service") from exc
        if self.other_controllers() or any(
            comm == "unitree_mujoco" and _dds_domain(cmd, {"-i", "--domain_id"}) == self.dds_domain
            for _, comm, cmd in processes()
        ):
            raise ServiceError(f"DDS domain {self.dds_domain} already has a simulator or controller")

        binary = self.simulator_dir / "unitree_mujoco"
        if not binary.is_file():
            raise ServiceError(f"{binary} is missing. Run scripts/setup_robot_mujoco.sh first.")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.scene_path = prepare_scene(self.robot_root, self.map_dir, self.output)
        self.scene_name = str(self.scene_path)
        self._init_dds()
        self.start_simulator({})

    def simulators(self):
        process = self.sim_proc
        if process is not None and process.poll() is None:
            return [{"pid": process.pid, "cmdline": str(self.scene_path)}]
        return []

    def other_controllers(self):
        return [
            controller for controller in super().other_controllers()
            if _dds_domain(controller["cmdline"], {"--dds-domain"}) == self.dds_domain
        ]

    def get_pose(self):
        pose = self.pose()
        if pose is None or not self.simulators():
            raise ServiceError("simulator is not sending a fresh pose")
        validate_pose(pose)
        return {"ok": True, **pose}

    def get_walk_status(self):
        return self.get_walk()

    def stop_walk(self, walk_id):
        return super().stop_walk({"walk_id": walk_id})

    def close(self):
        try:
            super().close()
        finally:
            try:
                if self.sim_proc is not None and self.sim_proc.poll() is None:
                    self.sim_proc.terminate()
                    try:
                        self.sim_proc.wait(8)
                    except subprocess.TimeoutExpired:
                        self.sim_proc.kill()
                        self.sim_proc.wait()
                for subscriber in self._subs:
                    subscriber.Close()
            finally:
                if self._lease is not None:
                    self._lease.close()
