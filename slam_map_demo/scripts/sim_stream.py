"""MuJoCo 仿真画面网页推流（MJPEG）：局域网内用浏览器打开 http://<服务器IP>:8771/ 即可观看。

原理：unitree_mujoco（C++ 仿真器）的窗口画面属于另一个进程，无法直接读取。本脚本加载同一个场景文件，
只读订阅仿真器发出的状态消息（rt/lowstate 的 12 个关节角和 IMU 姿态、rt/sportmodestate 的机身位置），
把机器狗摆到同样的姿态，再用 mujoco.Renderer 离屏渲染，经 OpenCV 编码为 JPEG，由 Flask 以
multipart/x-mixed-replace 持续输出。前端只需 <img src="http://<服务器IP>:8771/stream">。
不发送任何控制指令，可与 sim_service.py / follow_path.py 同时运行；仿真器没运行时画面显示 "waiting"。

默认是固定的斜侧视角，自动取景使整张地图放进画面；--azimuth/--elevation/--zoom 可微调。
画面左上角显示年月日和时间；3D 画面下方拼接一个滚动文字板块（画在视频帧里，只嵌 <img> 也能看到）：
每秒滚入一行实时状态（位置、朝向、速度、机身高度、倾斜、行走进度），仿真器连上/断开、开始/结束行走、
仿真控制服务（sim_service.py，默认 127.0.0.1:8770）的行走日志、摔倒等事件以醒目颜色插入。

用法（conda 环境 unitree-mujoco）：
  $P -I -u scripts/sim_stream.py                           # 默认场景 scene_slam_room1110.xml，端口 8771
  $P -I -u scripts/sim_stream.py scene_slam_xxx.xml --width 1280 --height 720 --fps 15

接口：
  GET /             网页（地图全景，右下角显示 Go2 第一视角小窗）
  GET /stream       MJPEG 视频流
  GET /snapshot.jpg 当前一帧
  GET /health       JSON 状态（是否收到仿真器状态、实际帧率、相机参数）
  GET /status       JSON 实时状态与最近的滚动文字（?n=行数，默认 50，最多 300）
  GET /camera/front 第一视角网页；/camera/front/stream 为无文字叠加的 MJPEG
  GET /camera/front.jpg 第一视角 JPEG；状态或帧过期时返回 503
  GET /observation  第一视角 JPEG base64、采集时间及同帧 IMU／机身位姿；过期返回 503
"""
import argparse
import importlib.util
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # 无窗口离屏渲染；必须在 import mujoco 之前设置

import cv2
import mujoco
import numpy as np
import yaml
from flask import Flask, Response, request
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
# sim_service.py starts this script with -I, which removes the script directory
# from sys.path. Load the sibling explicitly, never from an unrelated checkout.
_spec = importlib.util.spec_from_file_location("go2_first_person", HERE / "first_person.py")
_first_person = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_first_person)
FirstPersonFrames = _first_person.FirstPersonFrames
add_observation_routes = _first_person.add_observation_routes
configure_camera = _first_person.configure_camera
SCENE_DIR = HERE.parent.parent / "unitree_mujoco" / "unitree_robots" / "go2"
STATE_TIMEOUT = 1.0  # 秒：超过这么久没收到状态消息就认为仿真器没在运行
FONT_CANDIDATES = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                   "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                   "/usr/share/fonts/truetype/arphic/uming.ttc"]
# 滚动文字的颜色（RGB）：常规状态 / 仿真器未运行 / 事件 / 告警
LINE_COLORS = {"status": (215, 220, 225), "idle": (140, 145, 150), "event": (255, 210, 90), "warn": (255, 110, 100)}
FALL_TILT_DEG = 30

PAGE = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MuJoCo 仿真画面</title>
<style>
body{margin:0;background:#111;color:#ccc;font-family:sans-serif;text-align:center}
a{color:#8cf}
.overview{max-width:100vw;max-height:calc(100vh - 2em)}
.first-person-preview{position:fixed;right:12px;bottom:12px;z-index:10;
width:clamp(160px,24vw,280px);max-width:calc(100vw - 24px);overflow:hidden;
background:#171a20;border:1px solid #414853;border-radius:8px;
box-shadow:0 4px 20px #0008}
.first-person-preview header{display:flex;align-items:center;justify-content:space-between;
padding:7px 10px;font-size:12px}
.first-person-preview img{display:block;width:100%%;height:auto;aspect-ratio:16/9;
object-fit:contain;background:#000}
</style></head>
<body><img class="overview" src="/stream" alt="MuJoCo 地图全景"><div>场景 %s
｜ 实时状态 <a href="/status">/status</a>
｜ 推流状态 <a href="/health">/health</a></div>
<aside class="first-person-preview" aria-label="Go2 第一视角">
<header><span>Go2 第一视角</span><a href="/camera/front">放大 ↗</a></header>
<img src="/camera/front/stream" width="480" height="270" alt="Go2 实时第一视角">
</aside></body></html>"""


class StateMirror:
    """只读订阅仿真器状态，按消息把机器狗摆到 qpos。"""

    def __init__(self, model):
        self.lock = threading.Lock()
        self.low = None
        self.sport = None
        self.t_low = 0.0
        self.t_sport = 0.0
        # 机身自由关节
        base_body = model.site_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")]
        jnt = model.body_jntadr[base_body]
        assert model.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_FREE, "机身不是自由关节"
        self.base_qadr = model.jnt_qposadr[jnt]
        # imu 站点在机身坐标系中的位置（frame_pos / imu_quat 都是这个站点的量）
        self.imu_offset = model.site_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")].copy()
        # 仿真器的第 i 个电机 = 第 i 个执行器 → 它驱动的关节
        self.motor_qadr = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]

    def start(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_
        ChannelFactoryInitialize(1, "lo")  # 与仿真器相同：DDS 域 1、回环网卡
        self._subs = [ChannelSubscriber("rt/lowstate", LowState_), ChannelSubscriber("rt/sportmodestate", SportModeState_)]
        self._subs[0].Init(self._on_low, 10)
        self._subs[1].Init(self._on_sport, 10)

    def _on_low(self, m):
        with self.lock:
            self.low = ([s.q for s in m.motor_state[:len(self.motor_qadr)]], list(m.imu_state.quaternion))
            self.t_low = time.monotonic()
            self.low_received_ns = time.time_ns()

    def _on_sport(self, m):
        with self.lock:
            self.sport = list(m.position)
            self.velocity = list(m.velocity)
            self.t_sport = time.monotonic()
            self.sport_received_ns = time.time_ns()

    def summary(self):
        """机身实时状态；仿真器没在发状态时返回 None。位置是 IMU 处（与 sim_service /pose 一致）。"""
        with self.lock:
            if self.low is None or self.sport is None or not self.live():
                return None
            (x, y, z), (w, qx, qy, qz) = self.sport, self.low[1]
            v = getattr(self, "velocity", [0.0, 0.0, 0.0])
        yaw = np.degrees(np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        tilt = np.degrees(np.arccos(np.clip(1 - 2 * (qx * qx + qy * qy), -1, 1)))
        return {"x": round(float(x), 3), "y": round(float(y), 3), "z": round(float(z), 3),
                "yaw_deg": round(float(yaw), 1), "tilt_deg": round(float(tilt), 1),
                "speed": round(float(np.hypot(v[0], v[1])), 3), "fallen": bool(tilt > FALL_TILT_DEG)}

    def live(self):
        now = time.monotonic()
        return now - self.t_low < STATE_TIMEOUT and now - self.t_sport < STATE_TIMEOUT

    def apply(self, data):
        """把最新状态写进 qpos；返回是否有新鲜状态。"""
        snapshot = self.apply_snapshot(data)
        return snapshot is not None and snapshot["live"]

    def apply_snapshot(self, data):
        """使用同一次 DDS 状态快照设置 qpos，并保留该帧的位姿和主机时间。"""
        with self.lock:
            low, sport, live = self.low, self.sport, self.live()
            captured = time.monotonic()
            stamp_ns = time.time_ns()
            received = {"lowstate": getattr(self, "low_received_ns", None),
                        "sportmodestate": getattr(self, "sport_received_ns", None)}
            oldest_state = min(self.t_low, self.t_sport)
            state_age = {"lowstate": max(0.0, captured - self.t_low),
                         "sportmodestate": max(0.0, captured - self.t_sport)}
        if low is None or sport is None:
            return None
        q, quat = low
        quat = np.asarray(quat, dtype=float)  # w x y z
        if not np.isfinite([*q, *quat, *sport]).all() or np.linalg.norm(quat) < 1e-9:
            return None
        quat = quat / np.linalg.norm(quat)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, quat)
        a = self.base_qadr
        base = np.asarray(sport, dtype=float) - rot.reshape(3, 3) @ self.imu_offset
        data.qpos[a:a + 3] = base
        data.qpos[a + 3:a + 7] = quat
        for adr, v in zip(self.motor_qadr, q):
            data.qpos[adr] = v
        w, qx, qy, qz = quat
        yaw = float(np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (qx * qx + qy * qy), -1, 1))))
        pose = {"x": float(sport[0]), "y": float(sport[1]), "z": float(sport[2]),
                "yaw": yaw, "quaternion_wxyz": quat.tolist(), "tilt_deg": tilt,
                "fallen": bool(tilt > FALL_TILT_DEG)}
        return {"live": live, "stamp_ns": stamp_ns, "capture_monotonic": captured,
                "oldest_state_monotonic": oldest_state, "state_received_ns": received,
                "state_age_at_capture_s": state_age, "pose": pose,
                "base_pose": {**pose, "x": float(base[0]), "y": float(base[1]), "z": float(base[2])}}


class StatusFeed:
    """滚动文字的内容：定时写入实时状态行，状态变化和行走日志作为事件行插入。"""

    def __init__(self, mirror, service_url, interval, max_lines=300):
        self.mirror, self.service_url, self.interval = mirror, service_url.rstrip("/"), interval
        self.lines = deque(maxlen=max_lines)  # (文字, 类别)
        self.lock = threading.Lock()
        self.t_new = 0.0  # 最近一次新增行的时刻，用于滚动动画
        self.current = {"simulator": False, "pose": None, "walk": None}
        self.prev_live = self.prev_fallen = None
        self.walk_id = self.walk_state = None
        self.seen = set()
        self.first_poll = True

    def add(self, text, kind="event"):
        with self.lock:
            self.lines.append((time.strftime("%Y-%m-%d %H:%M:%S") + "  " + text, kind))
            self.t_new = time.time()

    def snapshot(self, n):
        with self.lock:
            return list(self.lines)[-n:], self.t_new

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        time.sleep(1.5)  # 等 DDS 订阅收到第一批消息，避免开头误报“仿真器未运行”
        tick = 0
        while True:
            t0 = time.time()
            try:
                self._step(tick)
            except Exception as e:  # noqa: BLE001  状态文字出错不能影响推流
                self.add(f"状态读取出错：{type(e).__name__}: {e}", "warn")
            tick += 1
            time.sleep(max(0.0, self.interval - (time.time() - t0)))

    def _poll_walk(self):
        """GET 仿真控制服务 /walk。没有行走记录（404）返回 {}；服务连不上或返回异常时返回 None。"""
        try:
            with urllib.request.urlopen(self.service_url + "/walk", timeout=0.8) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            return {} if e.code == 404 else None
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def _walk_events(self, w):
        if w is None:  # 连不上服务：不算一次有效查询，连上后仍按“启动时已有的行走”处理
            return
        if not w:  # 服务还没有行走记录
            self.first_poll = False
            return
        if w.get("walk_id") != self.walk_id:
            self.walk_id, self.seen = w.get("walk_id"), set()
            if self.first_poll:  # 推流启动前的行走：不重放旧日志
                self.seen = set(w.get("log") or [])
                self.walk_state = w.get("state")
            else:
                self.walk_state = None
                self.add(f"开始行走（{self.walk_id}）")
        self.first_poll = False
        for line in w.get("log") or []:
            if line not in self.seen:
                self.seen.add(line)
                self.add("行走日志：" + line)
        if self.walk_state == "running" and w.get("state") == "finished":
            legs = (w.get("result") or {}).get("legs") or []
            if legs:
                parts = [f"{l.get('name') or l.get('id')} {'到达' if l.get('reached') else '未到达'}"
                         + (f"（误差 {l['position_error']:.2f} m）" if l.get("position_error") is not None else "")
                         for l in legs]
                ok = all(l.get("reached") for l in legs)
                self.add(f"行走结束，用时 {w.get('elapsed_s', 0):.0f} s：" + "；".join(parts), "event" if ok else "warn")
            else:
                self.add(f"行走结束（退出码 {w.get('exit_code')}），没有结果", "warn")
        self.walk_state = w.get("state")

    def _step(self, tick):
        st = self.mirror.summary()
        live = st is not None
        w = self._poll_walk()
        if live != self.prev_live:
            if live:
                self.add("已收到仿真器状态，开始同步画面")
            elif self.prev_live is None:
                self.add("等待仿真器状态（仿真器未运行）", "idle")
            else:
                self.add("仿真器状态中断：仿真器已关闭或卡住", "warn")
            self.prev_live = live
        self._walk_events(w)
        fallen = bool(st and st["fallen"])
        if fallen != self.prev_fallen and self.prev_fallen is not None and live:
            self.add(f"机器狗摔倒了（倾斜 {st['tilt_deg']:.0f}°），需要重启仿真器" if fallen else "机器狗已恢复站立",
                     "warn" if fallen else "event")
        self.prev_fallen = fallen

        walking = bool(w) and w.get("state") == "running"
        walk_txt = ""
        if walking:
            m = re.search(r"\[\s*([^\]]+?)\s*\]", w.get("last_status") or "")
            walk_txt = f"  行走中 {w.get('elapsed_s', 0):.0f} s" + (f"（{m.group(1)}）" if m else "")
        with self.lock:
            self.current = {"simulator": live, "pose": st,
                            "walk": {k: w.get(k) for k in ("walk_id", "state", "elapsed_s", "last_status")} if w else None}
        if live:
            self.add(f"运行中  位置 ({st['x']:+.2f}, {st['y']:+.2f}) m  朝向 {st['yaw_deg']:+.0f}°  "
                     f"速度 {st['speed']:.2f} m/s  机身高度 {st['z']:.2f} m  倾斜 {st['tilt_deg']:.1f}°" + walk_txt,
                     "warn" if fallen else "status")
        elif tick % 5 == 0:
            self.add("仿真器未运行，等待状态消息", "idle")


class StatusPanel:
    """把 StatusFeed 的最近几行画成位于 3D 画面下方的文字板块（RGB 数组），新行出现时整体向上滑动。"""

    def __init__(self, feed, width, n_lines, font_size=20):
        self.feed, self.width, self.n = feed, width, n_lines
        path = next((f for f in FONT_CANDIDATES if Path(f).exists()), None)
        self.font = ImageFont.truetype(path, font_size) if path else ImageFont.load_default()
        self.line_h = int(font_size * 1.45)
        self.pad = 8
        self.height = self.n * self.line_h + 2 * self.pad

    def render(self):
        lines, t_new = self.feed.snapshot(self.n + 1)
        p = min(1.0, (time.time() - t_new) / 0.35)
        offset = int(round((1 - p) ** 2 * self.line_h))  # 先快后慢地滑到位
        img = Image.new("RGB", (self.width, self.height), (16, 18, 24))
        draw = ImageDraw.Draw(img)
        for i, (text, kind) in enumerate(reversed(lines)):  # 最新一行在最下面
            y = self.height - self.pad - (i + 1) * self.line_h + offset
            if y < -self.line_h:
                break
            draw.text((14, y + 2), text, font=self.font, fill=LINE_COLORS.get(kind, LINE_COLORS["status"]))
        draw.rectangle([(0, 0), (self.width, self.pad)], fill=(16, 18, 24))  # 遮住滚出顶部的旧行
        draw.line([(0, 0), (self.width, 0)], fill=(70, 80, 95), width=2)
        return np.asarray(img)


def map_points(model, data):
    """地图在世界系中的点集：固定环境中默认可见的几何体，包含嵌套的固定 body。
    网格取全部顶点，其余取包围盒角点；忽略机器人等活动 body。
    碰撞高度场和地面平面在第 3 组、默认隐藏，范围比实际地图大，不计入。"""
    pts = []
    for g in range(model.ngeom):
        if (model.body_weldid[model.geom_bodyid[g]] != 0 or model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
                or model.geom_group[g] >= 3):
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            m = model.geom_dataid[g]
            local = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        else:
            c, h = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
            local = c + h * np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
        pts.append(local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
    if not pts:
        c, e = model.stat.center, model.stat.extent
        return np.array([c - e, c + e])
    return np.concatenate(pts)


def fit_camera(model, pts, width, height, azimuth, elevation, zoom, margin=0.04):
    """固定方位角和俯仰角的自由相机，调整视线中心和距离，使地图所有点恰好落在画面内并居中。"""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation = azimuth, elevation
    tan_y = np.tan(np.radians(model.vis.global_.fovy) / 2)
    tan_x = tan_y * width / height
    az, el = np.radians(azimuth), np.radians(elevation)
    # 相机朝向（MuJoCo 自由相机约定）及其右、上方向
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    lookat = (pts.min(0) + pts.max(0)) / 2
    dist = np.linalg.norm(pts.max(0) - pts.min(0))
    for _ in range(30):  # 交替调整距离和视线中心，几步即收敛
        p = pts - lookat
        f, r, u = p @ fwd, p @ right, p @ up
        # 相机在 lookat - d*fwd：要求 |r| <= (d + f) * tan_x，|u| <= (d + f) * tan_y
        dist = max(np.max(np.abs(r) / tan_x - f), np.max(np.abs(u) / tan_y - f)) * (1 + margin)
        depth = dist + f
        sx, sy = r / (depth * tan_x), u / (depth * tan_y)  # 归一化屏幕坐标，±1 为画面边缘
        mx, my = (sx.max() + sx.min()) / 2, (sy.max() + sy.min()) / 2
        if abs(mx) < 1e-3 and abs(my) < 1e-3:
            break
        lookat = lookat + right * mx * dist * tan_x + up * my * dist * tan_y
    cam.lookat[:] = lookat
    cam.distance = dist / zoom
    return cam


def main():
    ap = argparse.ArgumentParser(description="MuJoCo 仿真画面 MJPEG 网页推流")
    ap.add_argument("scene", nargs="?", default="scene_slam_room1110.xml", help="场景文件名（在 unitree_robots/go2 下）或路径")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--quality", type=int, default=80, help="JPEG 质量 1-100")
    ap.add_argument("--azimuth", type=float, default=-90, help="相机方位角（度）；-90 表示相机在地图 +y 一侧、看向 -y")
    ap.add_argument("--elevation", type=float, default=None, help="相机俯仰角（度）；默认读取地图配置，无配置时为 -40")
    ap.add_argument("--zoom", type=float, default=1.0, help=">1 拉近，<1 拉远")
    ap.add_argument("--panel-lines", type=int, default=7, help="下方滚动文字板块显示的行数，0 表示不显示")
    ap.add_argument("--status-interval", type=float, default=1.0, help="实时状态行的写入间隔（秒）")
    ap.add_argument("--sim-service", default="http://127.0.0.1:8770", help="仿真控制服务地址，用于读取行走进度和日志")
    ap.add_argument("--map-dir", type=Path, help="地图包目录，用于在观测中返回 map_id/map_version")
    ap.add_argument("--front-width", type=int, default=480)
    ap.add_argument("--front-height", type=int, default=270)
    ap.add_argument("--front-hfov", type=float, default=112.0, help="第一视角水平视场角（度）")
    a = ap.parse_args()
    if not np.isfinite(a.fps) or a.fps <= 0:
        ap.error("--fps must be finite and positive")
    map_identity = {}
    front_visible_groups = []
    default_elevation = -40
    if a.map_dir is not None:
        meta = yaml.safe_load((a.map_dir / "meta.yaml").read_text())
        map_identity = {"map_id": meta["map_id"], "map_version": str(meta["version"])}
        front_visible_groups = meta.get("front_visible_geom_groups", [])
        default_elevation = meta.get("overview_elevation_deg", default_elevation)
        if not isinstance(front_visible_groups, list) or any(type(g) is not int or not 0 <= g <= 5 for g in front_visible_groups):
            ap.error("front_visible_geom_groups must contain geometry group numbers 0–5")
    elevation = default_elevation if a.elevation is None else a.elevation
    if type(elevation) not in (int, float) or not np.isfinite(elevation):
        ap.error("overview elevation must be finite")

    scene = Path(a.scene)
    if not scene.is_absolute() and not scene.exists():
        scene = SCENE_DIR / a.scene
    t0 = time.time()
    model = mujoco.MjModel.from_xml_path(str(scene))
    front_camera = configure_camera(model, a.front_width, a.front_height, a.front_hfov)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, a.width, a.front_width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, a.height, a.front_height)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    print(f"场景 {scene.name} 加载 {time.time() - t0:.1f} s", flush=True)

    pts = map_points(model, data)
    cam = fit_camera(model, pts, a.width, a.height, a.azimuth, elevation, a.zoom)
    print(f"地图范围 {np.round(pts.min(0), 2).tolist()} ~ {np.round(pts.max(0), 2).tolist()}，"
          f"相机看向 {np.round(cam.lookat, 2).tolist()}，距离 {cam.distance:.1f} m", flush=True)

    mirror = StateMirror(model)
    mirror.start()
    feed = StatusFeed(mirror, a.sim_service, a.status_interval)
    feed.start()

    frame = {"jpeg": None, "seq": 0, "fps": 0.0, "live": False}
    cond = threading.Condition()
    observations = FirstPersonFrames(front_camera, scene.resolve(), map_identity, STATE_TIMEOUT)

    def render_loop():
        # OpenGL 上下文绑定在创建它的线程，渲染器必须在本线程内创建和使用
        renderer = mujoco.Renderer(model, a.height, a.width)
        front_renderer = mujoco.Renderer(model, a.front_height, a.front_width)
        panel = StatusPanel(feed, a.width, a.panel_lines) if a.panel_lines > 0 else None
        opt = mujoco.MjvOption()  # 默认可见组与仿真器窗口一致（第 3 组碰撞体隐藏）
        front_opt = mujoco.MjvOption()
        front_opt.sitegroup[:] = 0  # 模型输入不画点位标记、IMU 站点及文字
        for group in front_visible_groups:
            front_opt.geomgroup[group] = 1  # 合成住宅的天花板只在第一视角显示。
        period, t_last, ema = 1.0 / a.fps, time.time(), 0.0
        while True:
            t_start = time.time()
            captured = mirror.apply_snapshot(data)
            live = captured is not None and captured["live"]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_camlight(model, data)  # 跟随机身更新固定相机的世界位姿
            if live:
                front_renderer.update_scene(data, camera=front_camera["name"], scene_option=front_opt)
                front_rgb = front_renderer.render()
                front_ok, front_buf = cv2.imencode(".jpg", cv2.cvtColor(front_rgb, cv2.COLOR_RGB2BGR),
                                                  [cv2.IMWRITE_JPEG_QUALITY, a.quality])
                observations.publish(front_buf.tobytes() if front_ok else None, captured)
            else:
                observations.publish(None, None)
            renderer.update_scene(data, camera=cam, scene_option=opt)
            rgb = renderer.render()
            if panel is not None:
                rgb = np.vstack([rgb, panel.render()])
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            label = time.strftime("%Y-%m-%d %H:%M:%S") + ("" if live else "  waiting for simulator state")
            cv2.putText(bgr, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(bgr, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            now = time.time()
            ema = 0.9 * ema + 0.1 / max(now - t_last, 1e-6) if ema else 1.0 / max(now - t_last, 1e-6)
            t_last = now
            if ok:
                with cond:
                    frame.update(jpeg=buf.tobytes(), seq=frame["seq"] + 1, fps=ema, live=live)
                    cond.notify_all()
            time.sleep(max(0.0, period - (time.time() - t_start)))

    threading.Thread(target=render_loop, daemon=True).start()

    app = Flask(__name__)
    add_observation_routes(app, observations)

    @app.get("/")
    def index():
        return PAGE % scene.name

    @app.get("/stream")
    def stream():
        def gen():
            seq = -1
            while True:
                with cond:
                    cond.wait_for(lambda: frame["seq"] != seq, timeout=5)
                    if frame["jpeg"] is None:
                        continue
                    seq, jpeg = frame["seq"], frame["jpeg"]
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"})

    @app.get("/snapshot.jpg")
    def snapshot():
        with cond:
            cond.wait_for(lambda: frame["jpeg"] is not None, timeout=10)
            jpeg = frame["jpeg"]
        if jpeg is None:
            return Response("还没有渲染出画面", status=503)
        return Response(jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-cache"})

    @app.get("/status")
    def status():
        lines, _ = feed.snapshot(max(1, min(int(request.args.get("n", 50)), 300)))
        with feed.lock:
            cur = dict(feed.current)
        body = {"ok": True, **cur, "lines": [t for t, _ in lines]}
        return Response(json.dumps(body, ensure_ascii=False), mimetype="application/json")

    @app.get("/health")
    def health():
        body = {"ok": frame["jpeg"] is not None, "simulator_state": frame["live"], "fps": round(frame["fps"], 1),
                "scene": scene.name, "size": [a.width, a.height],
                "first_person": {**front_camera, "ready": observations.latest() is not None,
                                 "observation_url": "/observation", "viewer_url": "/camera/front"},
                "camera": {"lookat": np.round(cam.lookat, 2).tolist(), "distance": round(cam.distance, 2),
                           "azimuth": cam.azimuth, "elevation": cam.elevation}}
        return Response(json.dumps(body, ensure_ascii=False), mimetype="application/json")

    print(f"推流地址 http://{a.host}:{a.port}/ （视频流 /stream）", flush=True)
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == "__main__":
    main()
