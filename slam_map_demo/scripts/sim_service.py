"""Control the Go2 simulation and its DDS processes in the MCP runtime."""
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from uuid import uuid4
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # Explicit project-local imports also work under -I.
import control_runtime
import trajectory_control
ROBOT = HERE.parents[1]
SIM_DIR = ROBOT / "unitree_mujoco" / "simulate" / "build"
SCENE_DIR = ROBOT / "unitree_mujoco" / "unitree_robots" / "go2"
SIM_START_TIMEOUT = 90  # 秒：等待仿真器发出第一条状态消息
WORK = Path("/tmp/sim_service")
CONTROLLER_SCRIPTS = ("follow_path.py", "waypoint_tour.py", "nav_demo.py", "walk_go2.py")
STATE_STALE = 1.0  # 这么久没收到仿真器状态，视为仿真器没在运行


class ServiceError(Exception):
    def __init__(self, msg, status=400):
        super().__init__(msg)
        self.status = status


def processes():
    """[(pid, 进程名, 命令行)]，直接读 /proc，避免依赖 psutil。"""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            comm = (d / "comm").read_text().strip()
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        if not cmd or "zombie" in _proc_state(int(d.name)):  # 已退出、还没被回收的进程
            continue
        out.append((int(d.name), comm, cmd))
    return out


def stop_process(pid, timeout=8.0):
    """先 SIGTERM，等不到退出再 SIGKILL；是本进程的子进程时顺便回收，避免留下僵尸进程。"""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    end = time.time() + timeout
    while time.time() < end and Path(f"/proc/{pid}").exists() and "zombie" not in _proc_state(pid):
        time.sleep(0.2)
    if Path(f"/proc/{pid}").exists() and "zombie" not in _proc_state(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass  # 不是本进程的子进程，由它自己的父进程回收


def _proc_state(pid):
    try:
        return Path(f"/proc/{pid}/status").read_text().split("State:")[1].split("\n")[0].lower()
    except (OSError, IndexError):
        return "zombie"


class SimService:
    def __init__(self, map_dir, python, *, work_dir=None, simulator_dir=SIM_DIR, dds_domain=1, subscribe=True):
        self.map_dir = Path(map_dir).resolve()
        meta = yaml.safe_load((self.map_dir / "meta.yaml").read_text())
        self.map_id, self.map_version = meta["map_id"], str(meta["version"])
        self.scene_name = f"scene_slam_{self.map_id}.xml"
        self.scene_path = SCENE_DIR / self.scene_name
        self.python = python
        self.simulator_dir = simulator_dir
        self.work_dir = work_dir if work_dir is not None else WORK
        self.dds_domain = dds_domain
        self.lock = threading.RLock()
        self.state_lock = threading.Lock()
        self.closing = False
        self.control = None
        imu = ET.parse(SCENE_DIR / "go2.xml").find(".//site[@name='imu']")
        self.imu_offset = np.fromstring(imu.get("pos", "0 0 0"), sep=" ")
        if self.imu_offset.shape != (3,):
            raise ValueError("Go2 IMU offset must have three components")
        self.sim_proc = None
        self.walk = None  # {"id", "proc", "log", "result", "started", "request"}
        self.walk_count = 0
        self.low = self.sport = None
        self.last_state = 0.0
        self.last_low_state = 0.0
        if subscribe:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            self._init_dds()

    # ---------- DDS 状态 ----------
    def _init_dds(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_

        ChannelFactoryInitialize(self.dds_domain, "lo")
        self._subs = [ChannelSubscriber("rt/sportmodestate", SportModeState_),
                      ChannelSubscriber("rt/lowstate", LowState_)]
        self._subs[0].Init(self._on_sport, 10)
        self._subs[1].Init(self._on_low, 10)

    def _on_sport(self, msg):
        with self.state_lock:
            self.sport, self.last_state = msg, time.monotonic()

    def _on_low(self, msg):
        with self.state_lock:
            self.low, self.last_low_state = msg, time.monotonic()

    def _state_pair(self):
        with self.state_lock:
            if (self.low is None or self.sport is None
                    or time.monotonic() - min(self.last_state, self.last_low_state) >= STATE_STALE):
                return None
            return self.low, self.sport

    def sim_alive(self):
        return self._state_pair() is not None

    def motion_state(self):
        """Read fresh low/sport state for the controller; no service lock needed."""
        pair = self._state_pair()
        if pair is None:
            return None
        low, sport = pair
        q = [float(m.q) for m in low.motor_state[:12]]
        quat = np.asarray(low.imu_state.quaternion, dtype=float)
        position, velocity = np.asarray(sport.position), np.asarray(sport.velocity)
        gyro = np.asarray(low.imu_state.gyroscope)
        if len(q) != 12 or not np.isfinite([*q, *quat, *position, *velocity, *gyro]).all():
            return None
        if np.linalg.norm(quat) < 1e-9:
            return None
        w, qx, qy, qz = quat / np.linalg.norm(quat)
        yaw = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (qx * qx + qy * qy), -1, 1))))
        rotation = np.array([[1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - w*qz), 2 * (qx*qz + w*qy)],
                             [2 * (qx*qy + w*qz), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - w*qx)],
                             [2 * (qx*qz - w*qy), 2 * (qy*qz + w*qx), 1 - 2 * (qx*qx + qy*qy)]])
        base = position - rotation @ self.imu_offset
        return {"motor_q": q, "speed": float(np.hypot(*velocity[:2])),
                "yaw_rate": float(gyro[2]), "tilt_deg": tilt,
                "base_pose": {"x": float(base[0]), "y": float(base[1]), "z": float(base[2]), "yaw": float(yaw)},
                "pose": {"x": round(float(position[0]), 3), "y": round(float(position[1]), 3),
                         "yaw": round(float(yaw), 4), "tilt_deg": round(tilt, 1), "fallen": tilt > 30}}

    def pose(self):
        state = self.motion_state()
        return state["pose"] if state is not None else None

    # ---------- 进程查询 ----------
    def simulators(self):
        return [{"pid": pid, "cmdline": cmd} for pid, comm, cmd in processes() if comm == "unitree_mujoco"]

    def other_controllers(self):
        mine = self.walk["proc"].pid if self.walk and self.walk["proc"].poll() is None else None
        return [{"pid": pid, "cmdline": cmd} for pid, comm, cmd in processes()
                if comm.startswith("python") and pid not in (mine, os.getpid())
                and any(Path(tok).name in CONTROLLER_SCRIPTS for tok in cmd.split())]

    def scene_map_version(self):
        try:
            m = re.search(r"maps/([^\"/]+)/([^\"/]+)/(?:mujoco|assets)/", self.scene_path.read_text())
            return f"{m.group(1)}/{m.group(2)}" if m else None
        except OSError:
            return None

    def walk_running(self):
        return self.walk is not None and self.walk["proc"].poll() is None

    def _check_open(self):
        if self.closing:
            raise ServiceError("服务正在关闭", 409)

    def _check_control_free(self):
        if self.control is not None:
            if self.control.active:
                raise ServiceError("持续控制占用中，请先停止持续控制并确认停止", 409)
            if self.control.requires_recovery:
                raise ServiceError("持续控制异常，需重启仿真器后再运动", 409)

    @staticmethod
    def _control_fields(req, allowed, required=()):
        if set(req) - set(allowed) or set(required) - set(req):
            raise ServiceError(f"请求字段应为 {', '.join(allowed)}；必填 {', '.join(required)}")

    def start_control(self, req):
        self._control_fields(req, ("trajectory_timeout_s",))
        try:
            timeout_s = control_runtime.validate_timeout(req.get("trajectory_timeout_s", trajectory_control.DEFAULT_TIMEOUT))
        except ValueError as exc:
            raise ServiceError(str(exc)) from exc
        with self.lock:
            self._check_open()
            self._check_control_free()
            if self.walk_running():
                raise ServiceError("路线执行占用中，请先停止行走并确认停止", 409)
            pose = self.pose()
            if pose is None or pose["fallen"]:
                raise ServiceError("需要正在运行且未跌倒的仿真器", 409)
            if self.other_controllers():
                raise ServiceError("有其他控制程序在运行，拒绝争抢 DDS 控制权", 409)
            self.control = trajectory_control.TrajectorySession("control_" + uuid4().hex[:16], self.motion_state, timeout_s)
            try:
                self.control.start()
            except Exception:
                self.control = None
                raise
            return self.get_control()

    def get_control(self):
        # Status queries must remain available while stop holds the mutation
        # lock and waits for the robot to settle. Keep one session reference.
        current = self.control
        if current is None:
            raise ServiceError("还没有持续控制会话", 404)
        return {"ok": True, **current.status(), "pose": self.pose()}

    def _matching_control(self, req):
        if self.control is None:
            raise ServiceError("没有持续控制会话", 404)
        if req.get("control_id") != self.control.control_id:
            raise ServiceError("control_id 与当前会话不符", 409)
        return self.control

    def trajectory_control(self, req):
        self._control_fields(req, ("control_id", "seq", "waypoints", "frame", "stamp_ns", "capture_pose", "stop"),
                             ("control_id", "seq"))
        with self.lock:
            self._check_open()
            control = self._matching_control(req)
            try:
                accepted = control.trajectory(req)
            except ValueError as exc:
                raise ServiceError(str(exc)) from exc
            except RuntimeError as exc:
                raise ServiceError(str(exc), 409) from exc
            return {**self.get_control(), "accepted": accepted}

    def _stop_control(self):
        if self.control is not None:
            self.control.request_stop()
            self.control.wait()

    def stop_control(self, req):
        self._control_fields(req, ("control_id",), ("control_id",))
        with self.lock:
            self._matching_control(req)
            self._stop_control()
            result = self.get_control()
            result["ok"] = result["stop_confirmed"] and not self.control.active
            if not result["ok"]:
                result["error"] = result.get("error") or "未确认停止，控制权未安全释放"
            destination = self.work_dir / f"{self.control.control_id}_result.json"
            destination.write_text(json.dumps(result, ensure_ascii=False))
            return result

    def close(self):
        with self.lock:
            self.closing = True
            if self.walk_running():
                self._stop_walk()
            self._stop_control()

    # ---------- 接口实现 ----------
    def health(self):
        sims = self.simulators()
        scene_ver = self.scene_map_version()
        warnings = []
        if scene_ver and scene_ver != f"{self.map_id}/{self.map_version}":
            warnings.append(f"场景文件 {self.scene_name} 引用的地图是 {scene_ver}，与服务的地图包不一致")
        if any(self.scene_name not in s["cmdline"] for s in sims):
            warnings.append("正在运行的仿真器加载的不是本服务地图对应的场景")
        if len(sims) > 1:
            warnings.append("有多个仿真器同时运行，会在同一 DDS 域冲突")
        others = self.other_controllers()
        if others:
            warnings.append("有不是本服务启动的控制程序在运行，会与本服务抢控制权")
        current_control = self.control
        control = current_control.status() if current_control is not None else None
        mode = (current_control.mode if current_control is not None and current_control.active else
                "blocked" if current_control is not None and current_control.requires_recovery else
                "route" if self.walk_running() else "idle")
        return {"ok": True, "map_id": self.map_id, "map_version": self.map_version,
                "scene": self.scene_name, "scene_map": scene_ver,
                "simulator_running": self.sim_alive(), "simulator_processes": sims,
                "pose": self.pose(), "walk": self.walk_summary(brief=True),
                "control_mode": mode, "control": control,
                "other_controllers": others, "warnings": warnings}

    def start_simulator(self, req):
        with self.lock:
            self._check_open()
            if self.walk_running():
                raise ServiceError("正在行走，请先停止行走", 409)
            if self.control is not None and self.control.active:
                raise ServiceError("持续控制占用中，请先停止持续控制", 409)
            sims = self.simulators()
            if self.control is not None and self.control.requires_recovery and sims and not req.get("restart"):
                raise ServiceError("持续控制异常，请用 {\"restart\": true} 重启仿真器", 409)
            if sims and not req.get("restart"):
                if self.sim_alive():
                    return {"ok": True, "message": "仿真器已在运行", "pose": self.pose()}
                raise ServiceError("有仿真器进程但收不到状态，请用 {\"restart\": true} 重启", 409)
            for s in sims:
                stop_process(s["pid"])
            self.control = None
            if not self.scene_path.exists():
                raise ServiceError(f"场景文件 {self.scene_path} 不存在", 500)
            with self.state_lock:
                self.sport = self.low = None
            log = open(self.work_dir / "simulator.log", "w")
            self.sim_proc = subprocess.Popen(["./unitree_mujoco", "-r", "go2", "-s", self.scene_name,
                                              "-i", str(self.dds_domain), "-n", "lo"],
                                             cwd=self.simulator_dir, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)
            # 大场景（如 room1110：70 万面网格 + 8192×8784 纹理）加载约 30 s，留足余量
            end = time.time() + SIM_START_TIMEOUT
            while time.time() < end:
                if self.sim_proc.poll() is not None:
                    tail = (self.work_dir / "simulator.log").read_text(errors="replace")[-800:]
                    raise ServiceError(f"仿真器启动后立即退出（退出码 {self.sim_proc.returncode}），"
                                       f"日志末尾：{tail}", 500)
                if self.sim_alive():
                    time.sleep(2.0)  # 等机器狗落地
                    return {"ok": True, "message": "仿真器已启动，机器狗在地图原点", "pid": self.sim_proc.pid,
                            "pose": self.pose()}
                time.sleep(0.3)
            raise ServiceError(f"仿真器 {SIM_START_TIMEOUT} 秒内没有发出状态消息（窗口可能打不开，见 {self.work_dir / 'simulator.log'}）",
                               500)

    def stop_simulator(self):
        with self.lock:
            self._stop_control()
            if self.walk_running():
                self._stop_walk()
            sims = self.simulators()
            for s in sims:
                stop_process(s["pid"])
            with self.state_lock:
                self.sport = self.low = None  # 不等状态超时，立即反映为“没有运行”
            if self.control is not None:
                self.control.wait(1)
                if not self.control.active:
                    self.control = None
            return {"ok": True, "message": f"已关闭 {len(sims)} 个仿真器进程"}

    def _path_file(self, req, name):
        """把请求中的路线写成 follow_path.py --path-json 能读的文件。"""
        if "plan" in req:
            data = req["plan"]
        elif "legs" in req:
            data = {k: v for k, v in req.items() if k not in ("force", "spawn")}
        elif "points" in req:
            data = {"points": req["points"]}
            if req.get("final_yaw") is not None:
                data["final_yaw"] = req["final_yaw"]
        else:
            raise ServiceError('请求体需要 "plan"（A* 返回）或 "points"（[[x, y], ...]）')
        if not isinstance(data, dict):
            raise ServiceError('"plan" 应为 A* MCP 的 get_plan 返回的路线')
        f = self.work_dir / name
        f.write_text(json.dumps(data, ensure_ascii=False))
        return f

    def start_walk(self, req):
        with self.lock:
            self._check_open()
            self._check_control_free()
            if self.walk_running():
                raise ServiceError("上一次行走还没结束，请先查询行走状态", 409)
            if not self.sim_alive():
                raise ServiceError("仿真器没有运行", 409)
            p = self.pose()
            if p is None:
                raise ServiceError("仿真器状态无效或过期", 409)
            if p["fallen"]:
                raise ServiceError("机器狗倒了，需要重启仿真器", 409)
            others = self.other_controllers()
            if others:
                raise ServiceError("有其他控制程序在运行：" + "；".join(f"{o['pid']} {o['cmdline']}" for o in others)
                                   + "。请先结束其他控制程序", 409)
            self.walk_count += 1
            wid = f"walk_{time.strftime('%H%M%S')}_{self.walk_count}"
            path_file = self._path_file(req, f"{wid}_path.json")
            result, log = self.work_dir / f"{wid}_result.json", self.work_dir / f"{wid}.log"
            cmd = [self.python, "-I", "-u", str(HERE / "follow_path.py"), "dds", str(self.map_dir),
                   "--path-json", str(path_file), "--result-json", str(result),
                   "--dds-domain", str(self.dds_domain), "--no-hold"]
            if req.get("force"):
                cmd.append("--force")
            proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, cwd=HERE.parent,
                                    start_new_session=True)
            self.walk = {"id": wid, "proc": proc, "log": log, "result": result, "started": time.time(),
                         "start_pose": p}
            # 输入有误（路线穿墙、地图版本不符等）时子进程会很快退出，直接把原因返回
            for _ in range(15):
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
            if proc.poll() is not None and proc.returncode != 0:
                raise ServiceError("行走程序拒绝执行：" + " ".join(self._log_lines(log)[-5:]))
            return {"ok": True, "walk_id": wid, "start_pose": p, "message": "已开始行走，可查询行走状态"}

    def _log_lines(self, log):
        try:
            lines = Path(log).read_text(errors="replace").splitlines()
        except OSError:
            return []
        return [l for l in lines if l.strip() and not l.startswith("t=") and "multicast" not in l]

    def walk_summary(self, brief=False):
        if self.walk is None:
            return None
        w = self.walk
        running = w["proc"].poll() is None
        out = {"walk_id": w["id"], "state": "running" if running else "finished",
               "elapsed_s": round(time.time() - w["started"], 1)}
        if not running:
            out["exit_code"] = w["proc"].returncode
        try:
            out["result"] = json.loads(w["result"].read_text())
        except (OSError, ValueError):
            out["result"] = None
        if brief:
            return out
        raw = Path(w["log"]).read_text(errors="replace").splitlines() if Path(w["log"]).exists() else []
        status = [l for l in raw if l.startswith("t=")]
        out["log"] = self._log_lines(w["log"])[-30:]
        out["last_status"] = status[-1] if status else None
        out["fell"] = any("[摔倒]" in l for l in raw)
        out["pose"] = self.pose()
        return out

    def get_walk(self):
        if self.walk is None:
            raise ServiceError("还没有行走过", 404)
        return {"ok": True, **self.walk_summary()}

    def _stop_walk(self):
        proc = self.walk["proc"]
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)  # follow_path.py 收到后先站稳 3 秒再退出
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def stop_walk(self, req=None):
        with self.lock:
            walk_id = (req or {}).get("walk_id")
            if walk_id is not None and (self.walk is None or self.walk["id"] != walk_id):
                raise ServiceError("walk_id 已更换，拒绝停止其他行走", 409)
            if not self.walk_running():
                return {"ok": True, "message": "当前没有在行走", "walk": self.walk_summary()}
            self._stop_walk()
            return {"ok": True, "message": "已停止，机器狗站立", "walk": self.walk_summary()}

    def preview(self, req, *, timeout_s=600):
        spawn = req.get("spawn")
        if spawn is None:
            p = self.pose()
            spawn = [p["x"], p["y"], p["yaw"]] if p else [0.0, 0.0, 0.0]
        try:
            spawn = [float(v) for v in spawn]
            assert len(spawn) == 3
        except (TypeError, ValueError, AssertionError):
            raise ServiceError('"spawn" 应为 [x, y, yaw]')
        pid = f"preview_{time.strftime('%H%M%S')}_{os.getpid() % 1000}_{threading.get_ident() % 1000}"
        path_file = self._path_file(req, f"{pid}_path.json")
        result, image = self.work_dir / f"{pid}_result.json", self.work_dir / f"{pid}.png"
        cmd = [self.python, "-I", "-u", str(HERE / "follow_path.py"), "headless", str(self.map_dir),
               "--scene", str(self.scene_path), "--path-json", str(path_file), "--result-json", str(result),
               "--image", str(image), "--spawn", *map(str, spawn)]
        if req.get("force"):
            cmd.append("--force")
        env = dict(os.environ, MUJOCO_GL="egl")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=HERE.parent, env=env)
        except subprocess.TimeoutExpired as exc:
            raise ServiceError(f"预演超过 {timeout_s:g} 秒，已放弃", 500) from exc
        lines = [l for l in (r.stdout + r.stderr).splitlines() if l.strip() and not l.startswith("t=")]
        try:
            res = json.loads(result.read_text())
        except (OSError, ValueError):
            res = None
        out = {"ok": r.returncode == 0, "spawn": spawn, "result": res, "log": lines[-30:],
               "image": f"/files/{image.name}" if image.exists() else None}
        if r.returncode != 0:
            out["error"] = "预演没有全部到达或路线被拒绝，见 log"
        return out
