"""Waypoint protocol, capture-time alignment, rolling MPC and HTTP contract."""

from concurrent.futures import Future
import math
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import control_runtime
import trajectory_control as tc
import trajectory_mpc as mpc
from control_helpers import FakeBackend, FakeTracker, state, wait_for


def packet(seq=1, **updates):
    value = {"seq": seq, "frame": "base_link", "stamp_ns": time.time_ns(),
             "capture_pose": {"x": 10.0, "y": 20.0, "yaw": math.pi / 2},
             "waypoints": [[.2, 0, 0], [.5, .1, .2]]}
    value.update(updates)
    return value



class ProtocolTests(unittest.TestCase):
    def test_local_rows_are_anchored_individually_at_capture_pose(self):
        result = tc.parse_reference(packet())
        np.testing.assert_allclose(result["waypoints_map"],
                                   [[10, 20.2, math.pi/2], [9.9, 20.5, math.pi/2+.2]])

    def test_map_rows_are_not_transformed(self):
        body = packet(frame="map", waypoints=[[1, 2, 4 * math.pi + .1]])
        body.pop("capture_pose")
        np.testing.assert_allclose(tc.parse_reference(body)["waypoints_map"], [[1, 2, .1]])

    def test_clear_and_model_stop_do_not_need_a_fresh_picture(self):
        for body in [{"seq": 1, "waypoints": []}, {"seq": 2, "stop": True}]:
            self.assertEqual(tc.parse_reference(body)["waypoints_map"], [])

    def test_rejects_ambiguous_or_invalid_references(self):
        updates = [{"frame": "camera"}, {"capture_pose": None}, {"waypoints": [[.1, 0]]},
                   {"waypoints": [[True, 0, 0]]}, {"waypoints": [[float("nan"), 0, 0]]},
                   {"waypoints": [[0, 0, 0]] * 65}, {"stamp_ns": True}, {"stop": "false"},
                   {"capture_pose": {"x": 0, "y": 0}}, {"seq": 1.2}]
        for update in updates:
            with self.subTest(update=update):
                with self.assertRaises(ValueError):
                    tc.parse_reference(packet(**update))


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.motion = state()
        self.motion["base_pose"] = {"x": 10.0, "y": 20.0, "yaw": math.pi / 2}
        self.backend, self.tracker = FakeBackend(), FakeTracker()
        self.session = tc.TrajectorySession("test", lambda: self.motion, timeout_s=.3,
                                            backend_factory=lambda: self.backend,
                                            tracker_factory=lambda: self.tracker)
        self.session.start()
        self.addCleanup(self.cleanup)
        wait_for(lambda: self.session.status()["ready"])

    def cleanup(self):
        self.session.request_stop()
        self.session.wait(2)
        self.assertFalse(self.session.active)

    def test_update_replaces_path_without_restarting_actuator(self):
        thread = self.session.thread
        self.session.trajectory(packet())
        wait_for(lambda: len(self.tracker.paths) == 1)
        self.session.trajectory(packet(seq=2, capture_pose={"x": 100, "y": 20, "yaw": 0}))
        wait_for(lambda: len(self.tracker.paths) == 2)
        self.assertIs(self.session.thread, thread)
        np.testing.assert_allclose(self.tracker.paths[-1][-1], [100.5, 20.1, .2])
        self.assertGreater(len(self.tracker.poses), 0)
        np.testing.assert_allclose(self.tracker.poses[0], [10, 20, math.pi/2])

    def test_expiry_counts_from_image_capture_not_http_receipt(self):
        self.session.trajectory(packet(stamp_ns=time.time_ns()-250_000_000))
        wait_for(lambda: self.session.status()["watchdog_active"])
        wait_for(lambda: self.backend.commands[-1] == (0.0, 0.0))
        self.assertTrue(self.session.active)
        self.assertGreaterEqual(self.session.status()["reference_age_s"], .3)

    def test_old_sequence_and_replayed_image_do_not_refresh_expiry(self):
        body = packet()
        self.session.trajectory(body)
        deadline = self.session.last_reference_at
        for update in [body, {**body, "seq": 2}]:
            with self.assertRaises(RuntimeError):
                self.session.trajectory(update)
        self.assertEqual(self.session.last_reference_at, deadline)
        self.assertEqual(self.session.last_seq, 1)

    def test_stale_future_and_invalid_packets_are_not_accepted(self):
        for update in [packet(stamp_ns=time.time_ns()-1_000_000_000),
                       packet(stamp_ns=time.time_ns()+1_000_000_000)]:
            with self.assertRaises(RuntimeError):
                self.session.trajectory(update)
        self.assertIsNone(self.session.last_seq)

    def test_clear_reference_keeps_late_solver_result_from_moving_robot(self):
        self.session.trajectory(packet())
        wait_for(lambda: (.1, .05) in self.backend.commands)
        self.session.trajectory({"seq": 2, "stop": True})
        wait_for(lambda: self.backend.commands[-1] == (0.0, 0.0))
        self.tracker.queued = (.3, .3)
        time.sleep(.03)
        self.assertEqual(self.backend.commands[-1], (0.0, 0.0))
        self.assertEqual(self.session.status()["tracking"]["state"], "holding")

    def test_solver_error_holds_until_new_reference(self):
        self.session.trajectory(packet())
        wait_for(lambda: len(self.tracker.paths) == 1)
        self.tracker.error = "solver rejected reference"
        wait_for(lambda: self.session.status()["tracking"]["state"] == "solver_error")
        self.assertEqual(self.backend.commands[-1], (0.0, 0.0))
        self.session.trajectory(packet(seq=2))
        wait_for(lambda: len(self.tracker.paths) == 2)
        wait_for(lambda: self.backend.commands[-1] == (.1, .05))

    def test_expired_solver_output_is_zeroed_even_if_reference_is_fresh(self):
        self.session.trajectory(packet())
        wait_for(lambda: (.1, .05) in self.backend.commands)
        self.tracker.submit = lambda pose: None
        self.tracker.queued = None
        with patch.object(tc, "SOLVE_TIMEOUT", .01):
            wait_for(lambda: self.session.status()["tracking"]["state"] == "waiting_solver")
            self.assertEqual(self.backend.commands[-1], (0.0, 0.0))

    def test_velocity_commands_are_rejected_in_trajectory_mode(self):
        self.assertFalse(hasattr(self.session, "command"))

    def test_state_loss_damps_and_blocks_further_references(self):
        self.motion = None
        wait_for(lambda: not self.session.active)
        self.assertTrue(self.session.requires_recovery)
        self.assertTrue(self.backend.damped)
        self.assertEqual(self.session.status()['mpc_target'], {'vx':0.0,'wz':0.0})
        with self.assertRaises(RuntimeError):
            self.session.trajectory(packet())

    def test_fall_requires_recovery(self):
        self.motion['tilt_deg'] = 45
        wait_for(lambda: not self.session.active)
        self.assertTrue(self.session.requires_recovery)
        self.assertTrue(self.backend.damped)

    def test_backend_failure_closes_tracker_without_reporting_stop(self):
        with patch.object(self.backend, 'step', side_effect=RuntimeError('writer failed')):
            wait_for(lambda: not self.session.active)
        self.assertTrue(self.session.requires_recovery)
        self.assertTrue(self.tracker.closed)
        self.assertFalse(self.session.stop_confirmed)

    def test_stop_does_not_confirm_while_physical_robot_is_moving(self):
        self.motion['speed'] = .2
        with patch.object(control_runtime, 'STOP_TIMEOUT', .1):
            self.session.request_stop()
            self.session.wait(2)
        self.assertFalse(self.session.stop_confirmed)
        self.assertTrue(self.session.requires_recovery)
        self.assertTrue(self.backend.damped)

    def test_stopped_session_rejects_new_references(self):
        self.session.request_stop()
        self.assertTrue(self.session.wait(2))
        self.assertTrue(self.session.stop_confirmed)
        self.assertTrue(self.backend.closed)
        with self.assertRaises(RuntimeError):
            self.session.trajectory(packet())

    def test_new_reference_cannot_inherit_previous_reached_status(self):
        self.session.trajectory(packet(waypoints=[[0,0,0]]))
        wait_for(lambda: self.session.status()['tracking']['reference_reached'])
        entered, release = threading.Event(), threading.Event()
        original = self.session._target_for_state
        def blocked(*args):
            entered.set()
            release.wait(2)
            return original(*args)
        try:
            with patch.object(self.session, '_target_for_state', blocked):
                self.assertTrue(entered.wait(1))
                self.session.trajectory(packet(seq=2))
                status = self.session.status()
                self.assertEqual(status['last_seq'], status['reference']['seq'])
                self.assertEqual(status['tracking']['reference_seq'], 2)
                self.assertEqual(status['tracking']['state'], 'reference_pending')
                self.assertFalse(status['tracking']['reference_reached'])
        finally:
            release.set()


class MpcTests(unittest.TestCase):
    def test_real_ipopt_solve_obeys_go2_limits(self):
        controller = mpc.MPCController(horizon=5, dt_s=.1, w_max=.3, a_max_v=.5,
                                       a_max_w=1.5, q_weights=(10,10,1), r_weights=(.1,.1))
        command, prediction = controller.solve(np.zeros(3), np.array([[.1*(i+1),0,0] for i in range(5)]), (0,0), .3)
        self.assertGreater(command[0], 0)
        self.assertLessEqual(command[0], .051)
        self.assertAlmostEqual(command[1], 0, places=5)
        self.assertEqual(prediction.shape, (6,3))
        self.assertTrue(np.isfinite(prediction).all())

    def test_replaced_path_discards_old_result_and_old_exception(self):
        for exception in [False, True]:
            with self.subTest(exception=exception):
                tracker = mpc.MpcTracker()
                try:
                    tracker.set_world_path([[.2, 0, 0]])
                    future = Future()
                    tracker._future, tracker._future_generation = future, tracker._generation
                    tracker.set_world_path([[0, .2, .3]])
                    if exception:
                        future.set_exception(RuntimeError("old solve failed"))
                    else:
                        future.set_result(mpc.MpcSolveResult(tracker._generation-1, (.3, .3),
                                                            np.zeros((5,3)), np.zeros((6,3)), 1))
                    self.assertIsNone(tracker.poll())
                    self.assertEqual(tracker.command, (0,0))
                    self.assertEqual(tracker.error, "")
                finally:
                    tracker.close()
