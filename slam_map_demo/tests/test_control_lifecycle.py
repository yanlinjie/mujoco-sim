"""Trajectory ownership, HTTP concurrency, state freshness and cleanup checks."""

from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sim_service
import control_runtime
from control_helpers import FakeBackend, FakeTracker, state, wait_for

class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        (root / "meta.yaml").write_text("map_id: test\nversion: v1\n")
        with patch.object(sim_service.SimService, "_init_dds"):
            self.service = sim_service.SimService(root, sys.executable, stream_port=0)
        self.motion = state()
        self.service.motion_state = lambda: self.motion
        self.service.sim_alive = lambda: self.motion is not None
        self.service.other_controllers = lambda: []
        self.service.simulators = lambda: []
        self.work_patch = patch.object(sim_service, "WORK", root)
        self.work_patch.start()
        self.addCleanup(self.work_patch.stop)
        self.backend_patch = patch.object(sim_service.control_runtime, "GaitBackend", FakeBackend)
        self.backend_patch.start()
        self.addCleanup(self.backend_patch.stop)
        self.tracker_patch = patch.object(sim_service.trajectory_control, "make_tracker", FakeTracker)
        self.tracker_patch.start()
        self.addCleanup(self.tracker_patch.stop)
        self.addCleanup(self.service.close)

    def error(self, status, function, *args):
        with self.assertRaises(sim_service.ServiceError) as caught:
            function(*args)
        self.assertEqual(caught.exception.status, status)

    def test_trajectory_owns_route_and_simulator_start(self):
        self.service.start_control({})
        self.error(409, self.service.start_walk, {"points": [[0, 0], [1, 0]]})
        self.error(409, self.service._start_simulator, {"restart": True})
        self.error(409, self.service.start_control, {})
        self.assertEqual(self.service.health()["control_mode"], "trajectory")

    def test_running_route_blocks_trajectory(self):
        self.service.walk = {"proc": SimpleNamespace(poll=lambda: None)}
        try:
            self.error(409, self.service.start_control, {})
        finally:
            self.service.walk = None

    def test_other_controller_and_invalid_state_block_start(self):
        self.service.other_controllers = lambda: [{"pid": 1}]
        self.error(409, self.service.start_control, {})
        self.service.other_controllers = lambda: []
        self.motion = None
        self.error(409, self.service.start_control, {})

    def test_old_lease_cannot_stop_or_command_new_session(self):
        old = self.service.start_control({})["control_id"]
        self.assertTrue(self.service.stop_control({"control_id": old})["ok"])
        new = self.service.start_control({})["control_id"]
        self.assertNotEqual(old, new)
        self.error(409, self.service.stop_control, {"control_id": old})
        self.error(409, self.service.trajectory_control, {"control_id": old, "seq": 9, "waypoints": []})
        self.assertTrue(self.service.control.active)

    def test_unknown_fields_and_missing_values_are_rejected(self):
        self.error(400, self.service.start_control, {"vx": 0.1})
        self.service.start_control({})
        self.error(400, self.service.trajectory_control, {"seq": 1, "vx": 0.1})
        self.error(400, self.service.stop_control, {})

    def test_removed_mode_configuration_cannot_create_a_session(self):
        for body in [{'mode':'velocity'}, {'command_timeout_s':.5}, {'mode':'trajectory'}]:
            with self.subTest(body=body):
                self.error(400, self.service.start_control, body)
                self.assertIsNone(self.service.control)

    def test_close_stops_control_and_prevents_new_motion(self):
        self.service.start_control({})
        self.service.close()
        self.assertTrue(self.service.control.stop_confirmed)
        self.assertFalse(self.service.control.active)
        self.error(409, self.service.start_control, {})

    def test_status_remains_available_during_stop(self):
        cid = self.service.start_control({})["control_id"]
        with ThreadPoolExecutor(1) as executor:
            stopped = executor.submit(self.service.stop_control, {"control_id": cid})
            wait_for(lambda: self.service.control.status()["state"] == "stopping")
            began = time.monotonic()
            self.assertEqual(self.service.get_control()["state"], "stopping")
            self.assertLess(time.monotonic() - began, 0.2)
            self.assertTrue(stopped.result()["stop_confirmed"])

    def test_failed_control_blocks_motion(self):
        self.service.start_control({})
        self.motion = None
        wait_for(lambda: not self.service.control.active)
        self.error(409, self.service.start_walk, {})
        self.error(409, self.service.start_control, {})
        self.assertEqual(self.service.health()["control_mode"], "blocked")

    def test_concurrent_starts_have_only_one_owner(self):
        def start():
            try:
                self.service.start_control({})
                return 200
            except sim_service.ServiceError as error:
                return error.status
        with ThreadPoolExecutor(2) as executor:
            self.assertEqual(sorted(executor.map(lambda _: start(), range(2))), [200, 409])

    def test_real_http_lifecycle(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), sim_service.make_handler(self.service))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{server.server_port}", trust_env=False) as client:
                self.assertEqual(client.get("/control").status_code, 404)
                created = client.post("/control/start", json={})
                self.assertEqual(created.status_code, 200)
                cid = created.json()["control_id"]
                command = {"control_id": cid, "seq": 1, "frame": "map", "stamp_ns": time.time_ns(), "waypoints": [[0.5,0,0]]}
                self.assertEqual(client.post("/control/command", json=command).status_code, 200)
                self.assertEqual(client.post("/control/command", json=command).status_code, 409)
                command["seq"], command["waypoints"] = 2, [[0.5,0]]
                self.assertEqual(client.post("/control/command", json=command).status_code, 400)
                stopped = client.post("/control/stop", json={"control_id": cid})
                self.assertEqual(stopped.status_code, 200)
                self.assertTrue(stopped.json()["stop_confirmed"])
                self.assertEqual(client.post("/control/stop", json={"control_id": cid}).status_code, 200)
                self.assertEqual(client.get("/health").json()["control_mode"], "idle")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)


class ValidationTests(unittest.TestCase):
    def test_rejects_invalid_timeout(self):
        for timeout in [True, 0, 0.09, 2.1, float("nan"), "0.5"]:
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    control_runtime.validate_timeout(timeout)


class StateFreshnessTests(unittest.TestCase):
    def test_both_dds_topics_must_be_fresh_and_finite(self):
        service = sim_service.SimService.__new__(sim_service.SimService)
        service.state_lock = threading.Lock()
        service.imu_offset = [-0.02557, 0.0, 0.04232]
        service._on_low(SimpleNamespace(motor_state=[SimpleNamespace(q=0.0)] * 12,
                                       imu_state=SimpleNamespace(quaternion=[1, 0, 0, 0], gyroscope=[0, 0, 0])))
        service._on_sport(SimpleNamespace(position=[0, 0, .3], velocity=[0, 0, 0]))
        self.assertIsNotNone(service.motion_state())
        service.last_low_state -= 2
        self.assertIsNone(service.motion_state())
        service.last_low_state = time.monotonic()
        service.sport.velocity[0] = float("nan")
        self.assertIsNone(service.motion_state())

