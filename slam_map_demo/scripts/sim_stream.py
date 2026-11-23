"""Render the map and Go2 state on the simulation MCP server's web routes."""

import logging
import os
import re
import threading
import time
from collections import deque
from contextlib import ExitStack
from html import escape
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont
from starlette.responses import HTMLResponse, JSONResponse, Response

from .first_person import FirstPersonFrames, add_observation_routes, configure_camera, mjpeg_response

STATE_TIMEOUT = 1.0
FALL_TILT_DEG = 30

FONT_CANDIDATES = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                   "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                   "/usr/share/fonts/truetype/arphic/uming.ttc"]
# 滚动文字的颜色（RGB）：常规状态 / 仿真器未运行 / 事件 / 告警
LINE_COLORS = {"status": (215, 220, 225), "idle": (140, 145, 150), "event": (255, 210, 90), "warn": (255, 110, 100)}
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
%s</body></html>"""

FRONT_WINDOW = """<aside class="first-person-preview" aria-label="Go2 第一视角">
<header><span>Go2 第一视角</span><a href="/camera/front">放大 ↗</a></header>
<img src="/camera/front/stream" width="480" height="270" alt="Go2 实时第一视角">
</aside>"""


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

    def start(self, domain):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_
        ChannelFactoryInitialize(domain, "lo")
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


    def close(self):
        for subscriber in self._subs:
            subscriber.Close()


class StatusFeed:
    """滚动文字的内容：定时写入实时状态行，状态变化和行走日志作为事件行插入。"""

    def __init__(self, mirror, get_walk, interval=1.0, max_lines=300):
        self.mirror, self.get_walk, self.interval = mirror, get_walk, interval
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

    def start(self, stop):
        self.thread = threading.Thread(target=self._loop, args=(stop,), daemon=True)
        self.thread.start()

    def _loop(self, stop):
        stop.wait(1.5)  # 等 DDS 订阅收到第一批消息，避免开头误报“仿真器未运行”
        tick = 0
        while not stop.is_set():
            t0 = time.time()
            try:
                self._step(tick)
            except Exception as e:  # noqa: BLE001  状态文字出错不能影响推流
                self.add(f"状态读取出错：{type(e).__name__}: {e}", "warn")
            tick += 1
            stop.wait(max(0.0, self.interval - (time.time() - t0)))

    def _walk_events(self, w):
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
        w = self.get_walk()
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


class Viewer:
    def __init__(self, simulation, *, first_person=True):
        self.simulation = simulation
        self.first_person = first_person
        self.feed = None
        self.camera = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="mujoco-viewer", daemon=True)
        self.frame = None
        self.observations = None
        self.error = None

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(5)

    def _run(self):
        try:
            self._render()
        except Exception as exc:
            self.error = str(exc)
            logging.getLogger(__name__).exception("viewer rendering failed")

    def _jpeg(self, rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return encoded.tobytes() if ok else None

    def _render(self):
        scene = self.simulation.scene_path
        model = mujoco.MjModel.from_xml_path(str(scene))
        if self.stop.is_set():
            return
        width, height = 1280, 720
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        data = mujoco.MjData(model)
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        meta = yaml.safe_load((self.simulation.map_dir / 'meta.yaml').read_text())
        camera = fit_camera(
            model, map_points(model, data), width, height,
            -90, meta.get("overview_elevation_deg", -40), 1,
        )
        mirror = StateMirror(model)
        self.feed = StatusFeed(mirror, self.simulation.walk_summary)
        panel = StatusPanel(self.feed, width, 7)
        self.camera = {"lookat": np.round(camera.lookat, 2).tolist(),
                       "distance": round(camera.distance, 2),
                       "azimuth": camera.azimuth, "elevation": camera.elevation}
        identity = {"map_id": self.simulation.map_id, "map_version": self.simulation.map_version}
        with ExitStack() as renderers:
            mirror.start(self.simulation.dds_domain)
            renderers.callback(mirror.close)
            self.feed.start(self.stop)
            overview = renderers.enter_context(mujoco.Renderer(model, height, width))
            front = None
            if self.first_person:
                front_camera = configure_camera(model, 480, 270, 112.0)
                self.observations = FirstPersonFrames(front_camera, scene, identity)
                front = renderers.enter_context(mujoco.Renderer(model, 270, 480))
                front_options = mujoco.MjvOption()
                front_options.sitegroup[:] = 0
                for group in meta.get("front_visible_geom_groups", []):
                    if type(group) is not int or not 0 <= group <= 5:
                        raise ValueError("front_visible_geom_groups must use group numbers 0–5")
                    front_options.geomgroup[group] = 1
            seq, last_frame, fps = 0, time.monotonic(), 0.0
            options = mujoco.MjvOption()
            while not self.stop.is_set():
                started = time.monotonic()
                sample = mirror.apply_snapshot(data)
                live = sample is not None and sample["live"]
                mujoco.mj_kinematics(model, data)
                mujoco.mj_camlight(model, data)
                if front is not None:
                    jpeg = None
                    if live:
                        front.update_scene(data, camera=front_camera["name"], scene_option=front_options)
                        jpeg = self._jpeg(front.render())
                    self.observations.publish(jpeg, sample)
                overview.update_scene(data, camera=camera, scene_option=options)
                rgb = np.vstack([overview.render(), panel.render()])
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                label = time.strftime("%Y-%m-%d %H:%M:%S") + ("" if live else "  waiting for simulator state")
                cv2.putText(bgr, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(bgr, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                jpeg = encoded.tobytes() if ok else None
                if jpeg is not None:
                    now = time.monotonic()
                    instantaneous = 1 / max(now - last_frame, 1e-6)
                    fps = 0.9 * fps + 0.1 * instantaneous if seq else instantaneous
                    seq, last_frame = seq + 1, now
                    self.frame = {"seq": seq, "jpeg": jpeg, "fps": fps, "live": live}
                self.stop.wait(max(0, 0.1 - (time.monotonic() - started)))

    def add_routes(self, server):
        if self.first_person:
            add_observation_routes(server, lambda: self.observations)

        @server.custom_route("/", methods=["GET"])
        async def index(_request):
            scene = escape(self.simulation.scene_path.name)
            return HTMLResponse(PAGE % (scene, FRONT_WINDOW if self.first_person else ""))

        @server.custom_route("/stream", methods=["GET"])
        async def stream(_request):
            return mjpeg_response(lambda: self.frame)

        @server.custom_route("/snapshot.jpg", methods=["GET"])
        async def snapshot(_request):
            frame = self.frame
            if frame is None:
                return Response("还没有渲染出画面", status_code=503)
            return Response(frame["jpeg"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

        @server.custom_route("/status", methods=["GET"])
        async def status(request):
            try:
                count = max(1, min(int(request.query_params.get("n", 50)), 300))
            except ValueError:
                return JSONResponse({"ok": False, "error": "n must be an integer"}, status_code=400)
            feed = self.feed
            if feed is None:
                return JSONResponse({"ok": True, "simulator": False, "pose": None, "walk": None, "lines": []})
            lines, _ = feed.snapshot(count)
            with feed.lock:
                current = feed.current.copy()
            return JSONResponse({"ok": True, **current, "lines": [line for line, _ in lines]})

        @server.custom_route("/health", methods=["GET"])
        async def health(_request):
            frame = self.frame
            front = self.observations
            return JSONResponse({
                "ok": frame is not None and self.error is None,
                "simulator_state": bool(frame and frame["live"]),
                "fps": round(frame["fps"], 1) if frame else 0,
                "scene": self.simulation.scene_path.name,
                "size": [1280, 720], "camera": self.camera,
                "first_person": {
                    **(front.camera if front else {}),
                    "enabled": self.first_person, "ready": bool(front and front.latest()),
                    "observation_url": "/observation" if self.first_person else None,
                    "viewer_url": "/camera/front" if self.first_person else None,
                },
                "error": self.error,
            })
