"""Check static-home camera bounds and collision-based navigation projection."""

from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "slam_map_demo/scripts"))

from import_family_house import navigation_grid
import import_family_house
from follow_path import load_legs
from slam_map_demo.scripts.nav_service import Planner
from slam_map_demo.scripts.sim_stream import map_points
from sim_service import SimService


class FamilyHouseTests(unittest.TestCase):
    def test_exported_room_goal_is_accepted_by_the_existing_walk_executor(self):
        """Exercise export -> A* plan -> the real executor's route decoder."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_dir = root / "map"
            map_dir.mkdir()
            (root / "unitree_mujoco/unitree_robots/go2").mkdir(parents=True)
            environment = ET.fromstring(
                '<mujoco><worldbody><geom type="plane" size="20 20 .1"/></worldbody></mujoco>'
            )
            house = {
                "objects": [],
                "rooms": [
                    {
                        "id": "room|4",
                        "roomType": "LivingRoom",
                        "floorPolygon": [
                            {"x": 0, "z": 0},
                            {"x": 16, "z": 0},
                            {"x": 16, "z": 16},
                            {"x": 0, "z": 16},
                        ],
                    }
                ],
            }
            with patch.object(import_family_house, "ROBOT", root):
                import_family_house.write_map(map_dir, environment, house, {}, 0.1)
            plan = Planner(map_dir).plan({"x": 0.5, "y": 0}, ["wp_00"])
            route = root / "route.json"
            route.write_text(json.dumps(plan))
            legs = load_legs(
                SimpleNamespace(points=None, path_json=str(route), final_yaw=None),
                {"map_id": "family_house", "version": "v1"},
            )
            self.assertEqual(legs[0]["id"], "wp_00")
            self.assertTrue(np.isfinite(legs[0]["yaw"]))

    def test_scene_identity_supports_reconstructed_and_synthetic_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / "scene.xml"
            service = SimService.__new__(SimService)
            service.scene_path = scene
            for resource, expected in (
                (
                    "/local/maps/room1110/2026-11-11.2/mujoco/collision_hfield.bin",
                    "room1110/2026-11-11.2",
                ),
                ("/local/maps/family_house/v1/assets/mesh.obj", "family_house/v1"),
            ):
                with self.subTest(resource=resource):
                    scene.write_text(
                        f'<mujoco><asset><mesh file="{resource}"/></asset></mujoco>'
                    )
                    self.assertEqual(service.scene_map_version(), expected)

    def test_overview_includes_nested_fixed_bodies_and_ignores_movable_robot(self):
        model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
          <geom type="box" pos="10 0 1" size=".5 .5 .5"/>
          <body pos="2 0 0"><body pos="0 1 0">
            <geom type="box" size=".5 .5 .5"/>
          </body></body>
          <body pos="100 0 1"><freejoint/><geom size=".2" mass="1"/></body>
          <geom type="box" pos="-100 0 0" size="1 1 1" group="3"/>
          <geom type="plane" size="1000 1000 .1"/>
        </worldbody></mujoco>""")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        points = map_points(model, data)
        np.testing.assert_allclose(points.min(0), [1.5, -0.5, -0.5])
        np.testing.assert_allclose(points.max(0), [10.5, 1.5, 1.5])

    def test_navigation_blocks_collision_shapes_and_preserves_free_floor(self):
        model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
          <body><geom type="box" pos="0 0 .5" size=".3 .3 .5"/></body>
          <geom type="box" pos="1 0 .5" size=".3 .3 .5" contype="0" conaffinity="0"/>
          <geom type="box" pos="-1 -1 .03" size=".3 .3 .03"/>
          <geom type="box" pos="-1 1 2" size=".3 .3 .1"/>
          <geom type="plane" size="10 10 .1"/>
        </worldbody></mujoco>""")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        rooms = [
            {
                "id": "room|4",
                "floorPolygon": [
                    {"x": -3, "z": -3},
                    {"x": 3, "z": -3},
                    {"x": 3, "z": 3},
                    {"x": -3, "z": 3},
                ],
            }
        ]
        grid, masks, observed, used = navigation_grid(model, data, rooms, 0.05)
        self.assertEqual(used, 1)
        self.assertEqual(grid.clearance_at([0, 0]), 0)
        for point in ([1, 0], [-1, -1], [-1, 1]):
            self.assertGreater(grid.clearance_at(point), 0.3)
        self.assertEqual(grid.clearance_at([4, 4]), 0)
        np.testing.assert_array_equal(masks[4], observed)


if __name__ == "__main__":
    unittest.main()
