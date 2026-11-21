"""First-person frames from the existing read-only Go2 DDS renderer."""

import base64
import json
import math
import threading
import time

import mujoco
from flask import Response


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
        self.condition = threading.Condition()
        self.sequence = 0
        self.frame = None

    def publish(self, jpeg, snapshot):
        with self.condition:
            if jpeg is not None and snapshot is not None and snapshot["live"]:
                self.sequence += 1
                self.frame = {
                    "seq": self.sequence,
                    "jpeg": jpeg,
                    "snapshot": snapshot,
                }
            else:
                self.frame = None
            self.condition.notify_all()

    def latest(self):
        with self.condition:
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


def add_observation_routes(app, frames):
    """Expose a clean RGB image, MJPEG view, and one coherent VLN observation."""
    def unavailable():
        return Response(json.dumps({"ok": False, "error": "no fresh first-person observation"}),
                        status=503, mimetype="application/json", headers=NO_CACHE)

    @app.get("/observation")
    def observation():
        frame = frames.latest()
        if frame is None:
            return unavailable()
        return Response(json.dumps(frames.observation(frame), allow_nan=False),
                        mimetype="application/json", headers=NO_CACHE)

    @app.get("/camera/front.jpg")
    def front_snapshot():
        frame = frames.latest()
        if frame is None:
            return unavailable()
        headers = {**NO_CACHE, "X-Frame-Seq": str(frame["seq"]),
                   "X-Capture-Time-Ns": str(frame["snapshot"]["stamp_ns"])}
        return Response(frame["jpeg"], mimetype="image/jpeg", headers=headers)

    @app.get("/camera/front/stream")
    def front_stream():
        if frames.latest() is None:
            return unavailable()

        def gen():
            seq = -1
            while True:
                with frames.condition:
                    frames.condition.wait_for(
                        lambda: frames.sequence != seq and frames.latest() is not None, timeout=1
                    )
                    frame = frames.latest()
                if frame is None or frame["seq"] == seq:
                    continue
                seq, jpeg = frame["seq"], frame["jpeg"]
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode() + b"\r\nX-Frame-Seq: " + str(seq).encode()
                       + b"\r\n\r\n" + jpeg + b"\r\n")

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame",
                        headers=NO_CACHE)

    @app.get("/camera/front")
    def front_page():
        return Response(
            '<!doctype html><html><head><meta charset="utf-8"><title>Go2 第一视角</title>'
            '<style>body{margin:0;background:#111;color:#ccc;text-align:center;font-family:sans-serif}'
            'img{max-width:100vw;max-height:90vh}a{color:#8cf}</style></head>'
            '<body><img src="/camera/front/stream"><div>Go2 第一视角 ｜ '
            '<a href="/">地图全景</a> ｜ <a href="/observation">观测 JSON</a></div></body></html>',
            mimetype="text/html", headers=NO_CACHE,
        )
