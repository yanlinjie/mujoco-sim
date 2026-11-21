"""Rolling local/map waypoint references tracked by MPC inside the Go2 service."""

import math
import time

import numpy as np

from control_runtime import ControlSession, finite_number, validate_timeout


DEFAULT_TIMEOUT = 2.0
MAX_POINTS = 64
SOLVE_PERIOD = 0.1
SOLVE_TIMEOUT = 0.5


def make_tracker():
    # Optional until a trajectory session starts. A* clients do not
    # import CasADi, and runtime imports only the project-local MPC copy.
    from trajectory_mpc import MpcTracker
    return MpcTracker()


def parse_reference(req):
    """Validate the wire payload and anchor every local point at capture time."""
    seq = req.get("seq")
    if type(seq) is not int or seq < 0:
        raise ValueError("seq must be a nonnegative integer")
    stop = req.get("stop", False)
    if type(stop) is not bool:
        raise ValueError("stop must be boolean")
    points = req.get("waypoints")
    if stop or points == []:
        return {"seq": seq, "stop": stop, "frame": None, "stamp_ns": None,
                "capture_pose": None, "waypoints_map": []}
    if not isinstance(points, list) or not 1 <= len(points) <= MAX_POINTS:
        raise ValueError("waypoints must contain 1 to 64 [x, y, yaw] rows, or [] to hold")
    parsed = []
    for index, row in enumerate(points):
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"waypoints[{index}] must be [x, y, yaw]")
        parsed.append([finite_number(v, f"waypoints[{index}]") for v in row])
    frame, stamp = req.get("frame"), req.get("stamp_ns")
    if frame not in ("base_link", "map"):
        raise ValueError("frame must be base_link (at capture time) or map")
    if type(stamp) is not int or stamp <= 0:
        raise ValueError("stamp_ns must be the positive host capture Unix timestamp")
    capture_pose = None
    path = np.array(parsed, dtype=float)
    if frame == "base_link":
        pose = req.get("capture_pose")
        if not isinstance(pose, dict):
            raise ValueError("base_link references require capture_pose from observation.base_pose")
        capture_pose = {k: finite_number(pose.get(k), f"capture_pose.{k}") for k in ("x", "y", "yaw")}
        x, y, yaw = (capture_pose[k] for k in ("x", "y", "yaw"))
        c, s = math.cos(yaw), math.sin(yaw)
        path[:, 0] = x + c * np.array(parsed)[:, 0] - s * np.array(parsed)[:, 1]
        path[:, 1] = y + s * np.array(parsed)[:, 0] + c * np.array(parsed)[:, 1]
        path[:, 2] += yaw
    elif "capture_pose" in req:
        raise ValueError("map-frame waypoints do not take capture_pose")
    path[:, 2] = np.arctan2(np.sin(path[:, 2]), np.cos(path[:, 2]))
    if not np.isfinite(path).all():
        raise ValueError("transformed trajectory must be finite")
    return {"seq": seq, "stop": False, "frame": frame, "stamp_ns": stamp,
            "capture_pose": capture_pose, "waypoints_map": path.tolist()}


class TrajectorySession(ControlSession):
    """The actuator loop owns one asynchronous 10 Hz MPC and its references."""

    mode = "trajectory"

    def __init__(self, control_id, state_provider, timeout_s=DEFAULT_TIMEOUT,
                 backend_factory=None, tracker_factory=None):
        super().__init__(control_id, state_provider, validate_timeout(timeout_s), backend_factory)
        self.tracker_factory = tracker_factory or make_tracker
        self.tracker = None
        self.reference = None
        self.revision = 0
        self.last_stamp_ns = None
        self.tracking = {"state": "waiting_reference", "reference_reached": False}
        # The following fields belong to the actuator thread, never HTTP threads.
        self._seen_revision = -1
        self._tracking_active = False
        self._held_at_end = False
        self._next_solve = 0.0
        self._last_solved = None
        self._solver_command = (0.0, 0.0)

    def trajectory(self, req):
        reference = parse_reference(req)
        now, wall_ns = time.monotonic(), time.time_ns()
        stamp = reference["stamp_ns"]
        if stamp is not None:
            age_ns = wall_ns - stamp
            if age_ns < -50_000_000 or age_ns >= int(self.timeout_s * 1e9):
                raise RuntimeError("capture timestamp is stale or in the future; get a fresh observation")
            captured_at = now - max(0, age_ns) / 1e9
        else:
            captured_at = now
        with self.lock:
            if self.state not in ("starting", "running") or self.stop_at is not None:
                raise RuntimeError("control session is stopping or finished")
            if reference["waypoints_map"] and not self.output["ready"]:
                raise RuntimeError("wait for ready=true before submitting a trajectory")
            if self.last_seq is not None and reference["seq"] <= self.last_seq:
                raise RuntimeError("seq must increase; old packets do not refresh the reference")
            if stamp is not None and self.last_stamp_ns is not None and stamp <= self.last_stamp_ns:
                raise RuntimeError("capture timestamps must increase; replay cannot refresh the watchdog")
            self.last_seq = reference["seq"]
            self.last_reference_at = captured_at  # Expiry is anchored to the observation, not receipt.
            if stamp is not None:
                self.last_stamp_ns = stamp
            self.reference = reference
            self.revision += 1
        return reference

    def status(self):
        with self.lock:
            result = self._status_locked()
            result["reference"] = self.reference
            result["tracking"] = dict(self.tracking)
            if self.reference is not None and self.tracking.get("reference_seq") != self.last_seq:
                result["tracking"] = {"state": "reference_pending", "reference_seq": self.last_seq,
                                      "reference_reached": False}
        return result

    def _prepare_source(self):
        self.tracker = self.tracker_factory()

    def _close_source(self):
        if self.tracker is not None:
            self.tracker.close()

    def _target_for_state(self, state, now, stopping, expired):
        with self.lock:
            reference, revision = self.reference, self.revision
            output = self.output
        path = reference["waypoints_map"] if reference is not None else []
        tracking = {"state": "tracking", "reference_reached": False,
                    "reference_seq": reference["seq"] if reference is not None else None}
        if stopping or expired or not path:
            if self._tracking_active:
                self.tracker.reset()
            self._tracking_active = False
            self._solver_command, self._last_solved = (0.0, 0.0), None
            tracking["state"] = "stopping" if stopping else "stale" if expired else "holding"
            command = (0.0, 0.0)
        else:
            if revision != self._seen_revision:
                self.tracker.set_world_path(path)
                self._seen_revision = revision
                self._tracking_active = True
                self._held_at_end = False
                self._solver_command, self._last_solved = (0.0, 0.0), None
                self._next_solve = now
            pose = np.array([state["base_pose"][k] for k in ("x", "y", "yaw")])
            end = np.array(path[-1])
            distance = float(np.hypot(*(end[:2] - pose[:2])))
            yaw_error = abs(math.atan2(math.sin(end[2] - pose[2]), math.cos(end[2] - pose[2])))
            tracking.update(position_error_m=distance, yaw_error_rad=yaw_error)
            if distance <= 0.04 and yaw_error <= 0.06:
                self._held_at_end = True
            if self._held_at_end:
                if self._tracking_active:
                    self.tracker.reset()
                self._tracking_active = False
                command = (0.0, 0.0)
                tracking["state"] = "holding_reference"
                tracking["reference_reached"] = bool(
                    distance <= 0.15 and yaw_error <= 0.2 and output.get("settled", False)
                    and state["speed"] <= 0.08 and abs(state["yaw_rate"]) <= 0.15
                )
            else:
                solved = self.tracker.poll()
                if solved is not None:
                    self._solver_command, self._last_solved = solved, now
                tracking.update(solve_ms=self.tracker.solve_ms, error=self.tracker.error,
                                solve_age_s=None if self._last_solved is None else now - self._last_solved)
                if not self.tracker.error and now >= self._next_solve:
                    self.tracker.previous_command = (output["applied"]["vx"], output["applied"]["wz"])
                    self.tracker.submit(pose)
                    self._next_solve = now + SOLVE_PERIOD
                if self.tracker.error or self._last_solved is None or now - self._last_solved > SOLVE_TIMEOUT:
                    command = (0.0, 0.0)
                    tracking["state"] = "solver_error" if self.tracker.error else "waiting_solver"
                else:
                    command = self._solver_command
            tracking["reference_reached"] = bool(
                distance <= 0.15 and yaw_error <= 0.2 and output.get("settled", False)
                and state["speed"] <= 0.08 and abs(state["yaw_rate"]) <= 0.15
            )
        # These are diagnostics of the *internal* actuator target, never a
        # velocity request expected from RPent or the LightNav caller.
        vx = float(np.clip(command[0], 0.0, 0.3))
        wz = float(np.clip(command[1], -0.3, 0.3))
        if not np.isfinite([vx, wz]).all():
            raise RuntimeError("MPC returned a non-finite command")
        with self.lock:
            self.target = {"vx": vx, "wz": wz}
            self.tracking = tracking
        return vx, wz
