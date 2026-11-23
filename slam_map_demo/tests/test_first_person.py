"""Observation consistency and stale-state checks without a running simulator."""

import asyncio
import base64
import math
import time
from types import SimpleNamespace
import unittest

import cv2
import mujoco
import numpy as np

from mcp.server.fastmcp import FastMCP
from starlette.testclient import TestClient

from slam_map_demo.scripts.first_person import (
    FirstPersonFrames, add_observation_routes, configure_camera, mjpeg_response,
)
from slam_map_demo.scripts.sim_stream import StateMirror


MODEL = """<mujoco><worldbody><body name="base_link">
  <freejoint/><geom type="box" size="0.1 0.1 0.1" mass="1"/>
  <site name="imu" pos="-0.02557 0 0.04232"/>
  <camera name="go2_front_rgb" pos="0.35 0 0.08" xyaxes="0 -1 0 0 0 1"/>
  <body><joint name="motor"/><geom size="0.01" mass="0.1"/></body>
</body></worldbody><actuator><motor joint="motor"/></actuator></mujoco>"""


class FirstPersonTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(MODEL)
        self.data = mujoco.MjData(self.model)
        self.mirror = StateMirror(self.model)
        self.set_state()
        self.camera = configure_camera(self.model, 480, 270, 112.0)
        self.frames = FirstPersonFrames(self.camera, "test.xml", {"map_id": "test", "map_version": "1"})
        server = FastMCP("camera")
        add_observation_routes(server, lambda: self.frames)
        self.client = TestClient(server.streamable_http_app())
        self.addCleanup(self.client.close)
        rgb = np.zeros((270, 480, 3), dtype=np.uint8)
        rgb[:, :, 0] = 240
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        self.assertTrue(ok)
        self.jpeg = encoded.tobytes()

    def set_state(self, position=(1.0, 2.0, 0.35), yaw=math.pi / 2):
        self.mirror._on_low(SimpleNamespace(
            motor_state=[SimpleNamespace(q=0.12)],
            imu_state=SimpleNamespace(quaternion=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]),
        ))
        self.mirror._on_sport(SimpleNamespace(position=position, velocity=[0, 0, 0]))

    def publish(self):
        snapshot = self.mirror.apply_snapshot(self.data)
        self.frames.publish(self.jpeg, snapshot)
        return snapshot

    def test_camera_tracks_rotated_body_and_imu_lever_arm(self):
        snapshot = self.publish()
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera["name"])
        rotation = self.data.cam_xmat[cid].reshape(3, 3)
        np.testing.assert_allclose(-rotation[:, 2], [0, 1, 0], atol=1e-12)
        base = snapshot["base_pose"]
        np.testing.assert_allclose([base[k] for k in ("x", "y", "z")], [1, 2.02557, 0.30768])
        np.testing.assert_allclose(self.data.cam_xpos[cid], [1, 2.37557, 0.38768])
        self.assertAlmostEqual(snapshot["pose"]["yaw"], math.pi / 2)
        self.assertAlmostEqual(self.camera["vfov_deg"], 79.6521943015147)
        self.assertEqual(self.camera["mount_frame"], "base_link")

    def test_http_observation_keeps_capture_pose_after_new_dds_arrives(self):
        snapshot = self.publish()
        self.set_state(position=(10, 20, 0.35), yaw=0)
        response = self.client.get("/observation")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["pose"], snapshot["pose"])
        self.assertEqual(body["base_pose"], snapshot["base_pose"])
        self.assertEqual(body["stamp_ns"], snapshot["stamp_ns"])
        self.assertEqual(body["state_received_ns"], snapshot["state_received_ns"])
        self.assertEqual(body["image"]["width"], 480)
        self.assertEqual(body["image"]["height"], 270)
        self.assertEqual(body["map_id"], "test")
        self.assertEqual(body["pose_reference"], "imu")
        self.assertEqual(base64.b64decode(body["image"]["data"]), self.jpeg)
        decoded = cv2.imdecode(np.frombuffer(self.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(decoded[10, 10, 2], 230)  # RGB red survives JPEG's BGR encoder.
        self.assertIn("no-store", response.headers["Cache-Control"])

    def test_jpeg_and_mjpeg_have_matching_frame_identity(self):
        snapshot = self.publish()
        response = self.client.get("/camera/front.jpg")
        self.assertEqual(response.content, self.jpeg)
        self.assertEqual(response.headers["X-Capture-Time-Ns"], str(snapshot["stamp_ns"]))
        self.assertEqual(response.headers["X-Frame-Seq"], "1")
        async def read_frame():
            return await anext(mjpeg_response(self.frames.latest).body_iterator)

        chunk = asyncio.run(read_frame())
        self.assertIn(b"X-Frame-Seq: 1", chunk)
        self.assertIn(self.jpeg, chunk)

    def assert_unavailable(self):
        for path in ("/observation", "/camera/front.jpg"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 503)
                self.assertFalse(response.json()["ok"])
                self.assertNotIn("image", response.json())

    def test_uninitialized_state_is_not_an_observation(self):
        self.assert_unavailable()
        self.assertEqual(self.client.get("/camera/front").status_code, 200)

    def test_old_dds_is_rejected_even_with_recent_capture(self):
        snapshot = self.mirror.apply_snapshot(self.data)
        snapshot["oldest_state_monotonic"] = time.monotonic() - 2
        self.frames.publish(self.jpeg, snapshot)
        self.assert_unavailable()

    def test_frozen_renderer_is_rejected_even_with_recent_dds(self):
        snapshot = self.mirror.apply_snapshot(self.data)
        snapshot["capture_monotonic"] = time.monotonic() - 2
        self.frames.publish(self.jpeg, snapshot)
        self.assert_unavailable()

    def test_lost_state_clears_previous_frame(self):
        self.publish()
        self.mirror.t_low -= 2
        snapshot = self.mirror.apply_snapshot(self.data)
        self.assertFalse(snapshot["live"])
        self.frames.publish(self.jpeg, snapshot)
        self.assert_unavailable()

    def test_invalid_state_does_not_enter_renderer(self):
        self.set_state(position=(float("nan"), 0, 0))
        self.assertIsNone(self.mirror.apply_snapshot(self.data))
        self.set_state()
        self.mirror.low = ([0.0], [0.0, 0.0, 0.0, 0.0])
        self.assertIsNone(self.mirror.apply_snapshot(self.data))

    def test_invalid_camera_parameters(self):
        for width, height, hfov in [(0, 270, 112), (480, -1, 112), (480, 270, 180),
                                     (480, 270, float("nan"))]:
            with self.subTest(width=width, height=height, hfov=hfov):
                with self.assertRaises(ValueError):
                    configure_camera(self.model, width, height, hfov)


if __name__ == "__main__":
    unittest.main()
