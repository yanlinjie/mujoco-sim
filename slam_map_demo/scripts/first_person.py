"""First-person frames from the existing read-only Go2 DDS renderer."""

import base64
import math
import threading
import time

import mujoco
from anyio import sleep
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse


CAMERA_NAME = "go2_front_rgb"
NO_CACHE = {"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"}


def configure_camera(model, width, height, hfov):
    """Configure a body-mounted camera; MuJoCo stores vertical FOV in degrees."""
    if width <= 0 or height <= 0 or not math.isfinite(hfov) or not 0 < hfov < 180:
        raise ValueError("front camera requires positive dimensions and 0 < hfov < 180")
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if camera_id < 0:
        raise ValueError(f"Go2 model has no {CAMERA_NAME} camera")
    model.cam_fovy[camera_id] = math.degrees(
        2 * math.atan(math.tan(math.radians(hfov) / 2) * height / width)
    )
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.cam_bodyid[camera_id])
    return {
        "name": CAMERA_NAME,
        "width": width,
        "height": height,
        "hfov_deg": hfov,
        "vfov_deg": float(model.cam_fovy[camera_id]),
        "mount_frame": body,
        "position_m": model.cam_pos[camera_id].tolist(),
        "quaternion_wxyz": model.cam_quat[camera_id].tolist(),
        "axes": "MuJoCo camera: +x right, +y up, -z forward",
    }


class FirstPersonFrames:
    """Publish image and capture-time poses together; reject stale DDS or frames.

    The renderer mirrors independently received low/sport messages. Timestamps
    refer to receipt and capture on this host, not the native simulation clock.
    """

    def __init__(self, camera, scene, map_identity=None, max_age_s=1.0):
        self.camera = camera
        self.scene = str(scene)
        self.map_identity = map_identity or {}
        self.max_age_s = max_age_s
        self.lock = threading.Lock()
        self.sequence = 0
        self.frame = None

    def publish(self, jpeg, snapshot):
        with self.lock:
            if jpeg is not None and snapshot is not None and snapshot["live"]:
                self.sequence += 1
                self.frame = {
                    "seq": self.sequence,
                    "jpeg": jpeg,
                    "snapshot": snapshot,
                }
            else:
                self.frame = None

    def latest(self):
        with self.lock:
            frame = self.frame
        if frame is None:
            return None
        snapshot = frame["snapshot"]
        now = time.monotonic()
        if max(now - snapshot["capture_monotonic"],
               now - snapshot["oldest_state_monotonic"]) > self.max_age_s:
            return None
        return frame

    def observation(self, frame):
        snapshot = frame["snapshot"]
        return {
            "ok": True,
            **self.map_identity,
            "scene": self.scene,
            "seq": frame["seq"],
            "stamp_ns": snapshot["stamp_ns"],
            "timestamp_source": "host_capture_unix_ns",
            "frame": "map",
            "pose_reference": "imu",
            "pose": snapshot["pose"],
            "base_pose": snapshot["base_pose"],
            "state_received_ns": snapshot["state_received_ns"],
            "state_age_at_capture_s": snapshot["state_age_at_capture_s"],
            "age_s": max(0.0, time.monotonic() - snapshot["capture_monotonic"]),
            "camera": self.camera,
            "image": {
                "encoding": "jpeg",
                "color_space": "RGB",
                "width": self.camera["width"],
                "height": self.camera["height"],
                "data": base64.b64encode(frame["jpeg"]).decode("ascii"),
            },
        }


def mjpeg_response(read_frame):
    async def frames():
        seq = -1
        while True:
            frame = read_frame()
            if frame is not None and seq != frame["seq"]:
                seq, jpeg = frame["seq"], frame["jpeg"]
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode() + b"\r\nX-Frame-Seq: " + str(seq).encode()
                       + b"\r\n\r\n" + jpeg + b"\r\n")
            await sleep(0.05)

    return StreamingResponse(
        frames(), media_type="multipart/x-mixed-replace; boundary=frame", headers=NO_CACHE,
    )


def add_observation_routes(server, get_frames):
    """Expose clean Go2 frames and their capture-time pose on the same port."""

    def latest():
        frames = get_frames()
        return frames.latest() if frames is not None else None

    def unavailable():
        return JSONResponse(
            {"ok": False, "error": "no fresh first-person observation"},
            status_code=503, headers=NO_CACHE,
        )

    @server.custom_route("/observation", methods=["GET"])
    async def observation(_request):
        frames = get_frames()
        frame = frames.latest() if frames is not None else None
        if frame is None:
            return unavailable()
        return JSONResponse(frames.observation(frame), headers=NO_CACHE)

    @server.custom_route("/camera/front.jpg", methods=["GET"])
    async def snapshot(_request):
        frame = latest()
        if frame is None:
            return unavailable()
        headers = {**NO_CACHE, "X-Frame-Seq": str(frame["seq"]),
                   "X-Capture-Time-Ns": str(frame["snapshot"]["stamp_ns"])}
        return Response(frame["jpeg"], media_type="image/jpeg", headers=headers)

    @server.custom_route("/camera/front/stream", methods=["GET"])
    async def stream(_request):
        return mjpeg_response(latest)

    @server.custom_route("/camera/front", methods=["GET"])
    async def front_page(_request):
        return HTMLResponse(
            '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Go2 第一视角</title>'
            '<style>body{margin:0;background:#111;color:#ccc;text-align:center;font-family:sans-serif}'
            'img{max-width:100vw;max-height:90vh}a{color:#8cf}</style>'
            '<body><img src="/camera/front/stream" alt="Go2 第一视角"><div>'
            '<a href="/">地图全景</a> ｜ <a href="/observation">观测 JSON</a></div></body></html>',
            headers=NO_CACHE,
        )
