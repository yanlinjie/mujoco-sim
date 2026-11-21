"""Shared deterministic controller fixtures; never publish DDS."""

import time
import numpy as np

def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true")


def state():
    return {"speed": 0.0, "yaw_rate": 0.0, "tilt_deg": 0.0, "motor_q": [0.0] * 12,
            "base_pose": {"x": 0.0, "y": 0.0, "z": 0.3, "yaw": 0.0},
            "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0, "tilt_deg": 0.0, "fallen": False}}


class FakeBackend:
    def __init__(self):
        self.commands = []
        self.damped = self.closed = False

    def step(self, vx, wz, motion, dt):
        self.commands.append((vx, wz))
        settled = vx == 0 and wz == 0
        return {"ready": True, "phase": "holding" if settled else "moving", "settled": settled,
                "applied": {"vx": vx, "wz": wz}, "swing_scale": 0 if settled else 1}

    def damp(self, motion):
        self.damped = True

    def close(self):
        self.closed = True


class FakeTracker:
    def __init__(self):
        self.solve_ms, self.error = 1.0, ""
        self.previous_command = (0.0, 0.0)
        self.queued = None
        self.closed = False
        self.paths, self.poses = [], []

    def set_world_path(self, path):
        self.paths.append(np.array(path))
        self.queued, self.error = None, ""

    def submit(self, pose):
        self.poses.append(np.array(pose))
        self.queued = (.1, .05)

    def poll(self):
        result, self.queued = self.queued, None
        return result

    def reset(self):
        self.queued = None

    def close(self):
        self.closed = True

