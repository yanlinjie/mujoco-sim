"""Internal Go2 actuation and lifecycle used by simulation-side trajectory MPC."""

import math
from pathlib import Path
import sys
import threading
import time

import numpy as np


W_MAX = 0.3
DEFAULT_TIMEOUT = 2.0
PERIOD = 0.002
STOP_TIMEOUT = 7.0
LIMITS = {"mpc_linear_speed_m_s": [0.0, 0.3], "mpc_angular_speed_rad_s": [-W_MAX, W_MAX],
          "trajectory_timeout_s": [0.1, 2.0]}


def finite_number(value, name):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def validate_timeout(value):
    value = finite_number(value, "trajectory_timeout_s")
    if not 0.1 <= value <= 2.0:
        raise ValueError("trajectory_timeout_s must be between 0.1 and 2.0")
    return value


class GaitBackend:
    """Reuse waypoint_tour's joint targets, gravity compensation and DDS writer."""

    def __init__(self):
        here = Path(__file__).resolve().parent
        sys.path.insert(0, str(here))  # The service may have been started with -I.
        import waypoint_tour
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
        from unitree_sdk2py.utils.crc import CRC

        self.tour, self.gait = waypoint_tour, waypoint_tour.gait
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()  # Reuse the service's domain 1 / lo participant.
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.cmd.head[0], self.cmd.head[1] = 0xFE, 0xEF
        self.cmd.level_flag = 0xFF
        for motor in self.cmd.motor_cmd:
            motor.mode = 0x01
        self.crc = CRC()
        self.elapsed = self.gait_time = self.swing_scale = 0.0
        self.vx = self.wz = 0.0
        self.start_q = None

    def write(self, q, dq, kp, kd, tau):
        self.gait.SetCmd(self.cmd, q, kp, kd, dq, tau)
        self.cmd.crc = self.crc.Crc(self.cmd)
        if not self.pub.Write(self.cmd):
            raise RuntimeError("DDS rejected the motor command")

    def step(self, vx, wz, state, dt):
        self.elapsed += dt
        stand_q, stand_tau = self.gait.StandJointPos(), self.gait.StandTorque()
        ready = self.elapsed > self.gait.STAND_UP_TIME + self.gait.HOLD_TIME
        if not ready:
            if self.start_q is None:
                self.start_q = np.array(state["motor_q"])
            phase = np.tanh(self.elapsed / 0.6)
            q = phase * stand_q + (1 - phase) * self.start_q
            self.write(q, np.zeros(12), phase * self.gait.KP_STAND + (1 - phase) * 20,
                       self.gait.KD_STAND, phase * stand_tau)
            phase_name = "standing_up"
        else:
            moving = max(abs(vx), abs(wz), abs(self.vx), abs(self.wz)) > 1e-3
            scale_target = 1.0 if moving else 0.0
            self.swing_scale += float(np.clip(scale_target - self.swing_scale, -dt, dt))
            if self.swing_scale < 0.9:
                vx = wz = 0.0
            self.vx += float(np.clip(vx - self.vx, -0.5 * dt, 0.5 * dt))
            wz = float(np.clip(wz, -W_MAX, W_MAX))
            self.wz += float(np.clip(wz - self.wz, -1.5 * dt, 1.5 * dt))
            self.gait_time += dt
            if self.swing_scale < 1e-6:
                self.write(stand_q, np.zeros(12), self.gait.KP_STAND, self.gait.KD_STAND, stand_tau)
                phase_name = "holding"
            else:
                q, dq, tau = self.tour.gait_command(self.gait_time, self.vx, self.wz, self.swing_scale)
                self.write(q, dq, self.gait.KP_WALK, self.gait.KD_WALK, tau)
                phase_name = "moving" if moving else "settling"
        return {"ready": ready, "phase": phase_name,
                "applied": {"vx": self.vx, "wz": self.wz}, "swing_scale": self.swing_scale,
                "settled": ready and self.swing_scale < 1e-6
                           and max(abs(self.vx), abs(self.wz)) < 1e-3}

    def damp(self, state):
        # On state loss/fall/error, leave zero feedforward and zero desired joint
        # velocity, rather than the last moving command in the native bridge.
        self.write(np.zeros(12), np.zeros(12), 0.0, self.gait.KD_STAND, np.zeros(12))

    def close(self):
        self.pub.Close()


class ControlSession:
    """Internal lease, reference watchdog and confirmed-stop lifecycle."""

    mode = "trajectory"

    def __init__(self, control_id, state_provider, timeout_s=DEFAULT_TIMEOUT, backend_factory=None):
        self.control_id = control_id
        self.state_provider = state_provider
        self.timeout_s = validate_timeout(timeout_s)
        self.backend_factory = backend_factory or GaitBackend
        self.lock = threading.Lock()
        self.thread = None
        self.state = "starting"
        self.last_seq = None
        self.last_reference_at = None
        self.target = {"vx": 0.0, "wz": 0.0}
        self.stop_at = None
        self.stop_confirmed = False
        self.error = None
        self.output = {"ready": False, "phase": "initializing", "applied": {"vx": 0.0, "wz": 0.0}}
        self.motion = None

    @property
    def active(self):
        with self.lock:
            return self.state in ("starting", "running", "stopping") or bool(
                self.thread is not None and self.thread.is_alive()
            )

    @property
    def requires_recovery(self):
        with self.lock:
            return self.state == "failed"

    def start(self):
        self.thread = threading.Thread(target=self._run, name="go2-trajectory-control", daemon=True)
        self.thread.start()

    def request_stop(self):
        with self.lock:
            if self.state in ("starting", "running", "stopping"):
                if self.stop_at is None:
                    self.stop_at = time.monotonic()
                self.state = "stopping"

    def wait(self, timeout=STOP_TIMEOUT + 2):
        if self.thread is not None:
            self.thread.join(timeout)
        return not self.active

    def status(self):
        with self.lock:
            return self._status_locked()

    def _status_locked(self):
        """Build one status while the session lock is held, including subclasses."""
        age = None if self.last_reference_at is None else time.monotonic() - self.last_reference_at
        expired = age is not None and age > self.timeout_s
        return {"control_id": self.control_id, "mode": self.mode, "state": self.state, **self.output,
                    "ready": self.output["ready"] and self.state == "running" and self.stop_at is None,
                    "trajectory_timeout_s": self.timeout_s, "last_seq": self.last_seq,
                    "reference_age_s": age, "watchdog_active": expired,
                    "mpc_target": {"vx": 0.0, "wz": 0.0} if expired or self.stop_at is not None or self.state == "failed"
                                  else dict(self.target),
                    "motion": self.motion, "stop_confirmed": self.stop_confirmed,
                    "requires_recovery": self.state == "failed", "error": self.error,
                    "limits": LIMITS}

    def _prepare_source(self):
        pass

    def _close_source(self):
        pass

    def _target_for_state(self, state, now, stopping, expired):
        raise NotImplementedError("a simulation-side trajectory tracker must supply the target")

    def _run(self):
        backend = None
        state = None
        stable_since = None
        try:
            backend = self.backend_factory()
            self._prepare_source()
            previous = time.monotonic()
            while True:
                began = time.monotonic()
                state = self.state_provider()
                if state is None:
                    raise RuntimeError("DDS state is missing, stale or invalid; simulator recovery required")
                if state["tilt_deg"] > 30:
                    raise RuntimeError("Go2 fell; simulator recovery required")
                with self.lock:
                    stopping = self.stop_at is not None
                    expired = self.last_reference_at is not None and began - self.last_reference_at > self.timeout_s
                    stop_at = self.stop_at
                vx, wz = self._target_for_state(state, began, stopping, expired)
                dt = min(max(began - previous, PERIOD), 0.02)
                previous = began
                output = backend.step(vx, wz, state, dt)
                motion = {k: state[k] for k in ("speed", "yaw_rate", "tilt_deg")}
                with self.lock:
                    self.output, self.motion = output, motion
                    if self.state == "starting":
                        self.state = "running"
                if stopping:
                    stable = output["settled"] and state["speed"] <= 0.08 and abs(state["yaw_rate"]) <= 0.15
                    stable_since = (stable_since or began) if stable else None
                    if stable_since is not None and began - stable_since >= 0.4:
                        with self.lock:
                            self.stop_confirmed = True
                            self.state = "finished"
                        break
                    if began - stop_at > STOP_TIMEOUT:
                        raise RuntimeError("could not confirm a settled stop; simulator recovery required")
                time.sleep(max(0.0, PERIOD - (time.monotonic() - began)))
        except Exception as exc:
            if backend is not None:
                try:
                    backend.damp(state)
                except Exception as damping_error:
                    exc = RuntimeError(f"{exc}; damping command also failed: {damping_error}")
            with self.lock:
                self.error, self.state = str(exc), "failed"
        finally:
            try:
                self._close_source()
            except Exception as exc:
                with self.lock:
                    self.error, self.state, self.stop_confirmed = str(exc), "failed", False
            if backend is not None:
                try:
                    backend.close()
                except Exception as exc:
                    with self.lock:
                        self.error, self.state, self.stop_confirmed = str(exc), "failed", False
