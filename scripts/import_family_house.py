"""Adapt the bundled synthetic ProcTHOR home to Go2 and matching A* map data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError
import yaml

PROJECT = Path(__file__).resolve().parents[2]
ROBOT = PROJECT / "mujoco_sim"
sys.path.insert(0, str(ROBOT / "slam_map_demo/scripts"))
from nav_grid import NavGrid, ROBOT_RADIUS, OBSTACLE_HEIGHT

MAP_ID, VERSION = "family_house", "v1"
ROOM_NAMES = {
    2: "厨房",
    3: "客厅西区",
    4: "客厅",
    5: "客厅东区",
    6: "卧室1",
    7: "卫生间1",
    8: "卧室2",
    9: "卫生间2",
    10: "卧室3",
    11: "卫生间3",
}
OBJECT_NAMES = {
    "Fridge": "冰箱前",
    "Sofa": "沙发前",
    "Bed": "床旁",
    "DiningTable": "餐桌旁",
    "Toilet": "马桶前",
}


def polygon_mask(points, shape, lo, resolution):
    """Rasterize world-coordinate floor/obstacle polygons in the planner's grid."""
    mask = np.zeros(shape, np.uint8)
    pixels = np.rint((np.asarray(points) - lo) / resolution).astype(np.int32)
    cv2.fillPoly(mask, [pixels], 1)
    return mask.astype(bool)


def frozen_environment(source, asset_root):
    """Keep the authored open interior doors and freeze furniture at its default pose."""
    root = ET.parse(source).getroot()
    for element in root.iter():
        if element.get("file"):
            resource = (source.parent / element.get("file")).resolve()
            if not resource.is_relative_to(asset_root) or not resource.is_file():
                raise ValueError(f"missing or external scene asset: {resource}")
            element.set("file", str(resource))
    for parent in root.iter():
        for child in tuple(parent):
            if child.tag in {"joint", "freejoint"}:
                parent.remove(child)
    # Fixed environment bodies do not need the original inter-object exclusions.
    for tag in ("contact", "equality", "actuator", "sensor", "keyframe"):
        element = root.find(tag)
        if element is not None:
            root.remove(element)
    names = {e.get("name") for e in root.iter() if e.get("name")}
    names.update(e.get("class") for e in root.iter("default") if e.get("class"))
    references = {
        "name",
        "mesh",
        "material",
        "texture",
        "class",
        "childclass",
        "body1",
        "body2",
        "site",
        "site1",
        "site2",
        "geom",
        "geom1",
        "geom2",
    }
    for element in root.iter():
        for attribute, value in tuple(element.attrib.items()):
            if attribute in references and value in names:
                element.set(attribute, "home_" + value)
        if element.tag == "geom" and "ceiling" in element.get("name", "").lower():
            element.set("group", "5")
    root.set("model", "Go2 in synthetic furnished family home")
    root.find("option").set("timestep", "0.002")
    root.find("option").set("impratio", "100")
    visual = root.find("visual")
    visual.find("quality").set("shadowsize", "2048")
    return root


def geometry_vertices(model, data, geom):
    """Read compiled world vertices, matching MuJoCo's actual static geometry."""
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        local = model.mesh_vert[start : start + count]
    else:
        center, half = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
        local = center + half * np.array(
            [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        )
    return local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def navigation_grid(model, data, rooms, resolution):
    """Project collidable shapes below 1.5 m; this does not use reconstructed scans."""
    polygons = [np.array([[v["x"], v["z"]] for v in r["floorPolygon"]]) for r in rooms]
    all_points = np.concatenate(polygons)
    lo = np.floor((all_points.min(0) - 0.5) / resolution) * resolution
    hi = all_points.max(0) + 0.5
    shape_xy = np.ceil((hi - lo) / resolution).astype(int) + 1
    shape = tuple(shape_xy[::-1])
    room_masks = {
        int(room["id"].split("|")[-1]): polygon_mask(poly, shape, lo, resolution)
        for room, poly in zip(rooms, polygons)
    }
    observed = np.logical_or.reduce(list(room_masks.values()))
    blocked = np.zeros(shape, bool)
    used = 0
    for geom in range(model.ngeom):
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        vertices = geometry_vertices(model, data, geom)
        if vertices[:, 2].max() <= OBSTACLE_HEIGHT or vertices[:, 2].min() >= 1.5:
            continue
        xy = vertices[:, :2]
        try:
            hull = ConvexHull(xy)
            footprint = xy[hull.vertices]
        except QhullError:
            low, high = xy.min(0) - resolution / 2, xy.max(0) + resolution / 2
            footprint = [
                [low[0], low[1]],
                [high[0], low[1]],
                [high[0], high[1]],
                [low[0], high[1]],
            ]
        blocked |= polygon_mask(footprint, shape, lo, resolution)
        used += 1
    heights = np.where(blocked, 1.5, 0.0)
    grid = NavGrid(heights, lo, resolution, observed)
    return grid, room_masks, observed, used


def choose_point(grid, mask, target, minimum_clearance):
    rows, cols = np.nonzero(mask & (grid.clearance >= minimum_clearance))
    if not len(rows):
        raise ValueError("no safe navigable point exists in a room")
    positions = grid.lo + np.column_stack((cols, rows)) * grid.res
    index = np.argmin(np.linalg.norm(positions - target, axis=1))
    return positions[index]


def write_map(map_dir, root, house, object_metadata, resolution):
    """Generate one coherent scene, navigation package, semantic goals and tasks."""
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    grid, masks, observed, used = navigation_grid(
        model, data, house["rooms"], resolution
    )
    spawn = choose_point(grid, masks[4], np.array([6.5, 13.8]), 0.7)
    labels, _ = ndimage.label(grid.clearance > ROBOT_RADIUS)
    origin_label = labels[grid.to_cell(spawn)]
    connected = labels == origin_label
    if origin_label == 0:
        raise ValueError("spawn is not in free space")
    unreachable = [room for room, mask in masks.items() if not np.any(mask & connected)]
    if unreachable:
        raise ValueError(f"rooms are not connected at the Go2 clearance: {unreachable}")
    targets = []
    room_polygons = []
    for room in house["rooms"]:
        room_id = int(room["id"].split("|")[-1])
        poly = np.array([[v["x"], v["z"]] for v in room["floorPolygon"]])
        target = (
            spawn
            if room_id == 4
            else choose_point(grid, masks[room_id] & connected, poly.mean(0), 0.4)
        )
        relative = target - spawn
        targets.append(
            {
                "id": "wp_00" if room_id == 4 else f"wp_room_{room_id}",
                "name": ROOM_NAMES[room_id],
                "semantic_tags": [room["roomType"].lower(), f"room_{room_id}"],
                "stand_pose": {
                    "x": round(float(relative[0]), 3),
                    "y": round(float(relative[1]), 3),
                    "z": 0.0,
                    # The existing route executor requires a final heading.
                    "yaw": 0.0,
                },
                "room_id": room_id,
                "neighbors": [],
            }
        )
        room_polygons.append(
            {
                "id": room_id,
                "name": ROOM_NAMES[room_id],
                "type": room["roomType"],
                "polygon": (poly - spawn).tolist(),
            }
        )
    objects = []
    for body_name, item in object_metadata.items():
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "home_" + body_name)
        if body < 0:
            continue
        position = data.xpos[body]
        record = {
            "id": item["object_id"],
            "category": item["category"],
            "asset_id": item["asset_id"],
            "room_id": item.get("room_id"),
            "body_name": "home_" + body_name,
            "position": (position - np.r_[spawn, 0.0]).tolist(),
        }
        objects.append(record)
        category, room_id = item["category"], item.get("room_id")
        if category not in OBJECT_NAMES or room_id not in masks or item.get("parent"):
            continue
        # Leave space for a useful furniture view from the low Go2 camera.
        rows, cols = np.nonzero(masks[room_id] & connected & (grid.clearance >= 0.5))
        points = grid.lo + np.column_stack((cols, rows)) * resolution
        distances = np.linalg.norm(points - position[:2], axis=1)
        if not len(points):
            raise ValueError(f"no safe viewpoint for {item['object_id']}")
        standoff = 1.2 if category == "Toilet" else 1.8
        target = points[np.argmin(abs(distances - standoff))]
        relative = target - spawn
        identifier = f"wp_{category.lower()}_{room_id}"
        if any(t["id"] == identifier for t in targets):
            continue
        targets.append(
            {
                "id": identifier,
                "name": ROOM_NAMES[room_id] + OBJECT_NAMES[category],
                "semantic_tags": [f"{category.lower()}_{room_id}", item["object_id"]],
                "stand_pose": {
                    "x": round(float(relative[0]), 3),
                    "y": round(float(relative[1]), 3),
                    "z": 0.0,
                    "yaw": float(
                        math.atan2(position[1] - target[1], position[0] - target[0])
                    ),
                },
                "lookat": {
                    "x": float(position[0] - spawn[0]),
                    "y": float(position[1] - spawn[1]),
                    "z": float(position[2]),
                },
                "room_id": room_id,
                "neighbors": [],
            }
        )
    grid.lo = grid.lo - spawn
    for waypoint in targets:
        goal = waypoint["stand_pose"]
        path = grid.plan([0, 0], [goal["x"], goal["y"]])
        if path is None:
            raise ValueError(f"unreachable semantic waypoint: {waypoint['name']}")
    generated = map_dir / "mujoco"
    generated.mkdir(exist_ok=True)
    np.savez_compressed(
        generated / "collision_hfield.npz",
        heights=grid.heights,
        lo=grid.lo,
        resolution=resolution,
        observed=observed,
    )
    metadata = {
        "map_id": MAP_ID,
        "version": VERSION,
        "up_axis": "z",
        "units": "meter",
        "source_type": "procedural_furnished_house",
        "source": "Ai2 MolmoSpaces ProcTHOR val_2 (CC BY 4.0)",
        "source_spawn_xy": spawn.tolist(),
        "room_count": len(house["rooms"]),
        "furniture_groups": len(house["objects"]),
        "front_visible_geom_groups": [5],
        "overview_elevation_deg": -65,
        "navigation": {
            "source": "compiled static collision footprints",
            "resolution_m": resolution,
            "robot_radius_m": ROBOT_RADIUS,
            "vertical_band_m": [OBSTACLE_HEIGHT, 1.5],
        },
    }
    (map_dir / "meta.yaml").write_text(
        yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False)
    )
    (map_dir / "waypoints.yaml").write_text(
        yaml.safe_dump(
            {"map_id": MAP_ID, "map_version": VERSION, "waypoints": targets},
            allow_unicode=True,
            sort_keys=False,
        )
    )
    (map_dir / "rooms.json").write_text(
        json.dumps(room_polygons, ensure_ascii=False, indent=2) + "\n"
    )
    (map_dir / "objects.json").write_text(
        json.dumps(objects, ensure_ascii=False, indent=2) + "\n"
    )
    world = root.find("worldbody")
    environment = ET.Element(
        "body", name="home_environment", pos=f"{-spawn[0]} {-spawn[1]} 0"
    )
    for child in tuple(world):
        world.remove(child)
        environment.append(child)
    world.append(environment)
    root.insert(0, ET.Element("include", file="go2.xml"))
    statistic = ET.Element(
        "statistic", center=f"{7.35 - spawn[0]} {9.2 - spawn[1]} 1", extent="12"
    )
    root.insert(1, statistic)
    scene = ROBOT / f"unitree_mujoco/unitree_robots/go2/scene_slam_{MAP_ID}.xml"
    ET.indent(root)
    ET.ElementTree(root).write(scene, encoding="unicode")
    tasks = []
    instructions = {
        "Fridge": "Leave the living room, enter the kitchen, and stop in front of the refrigerator.",
        "Bed": "Go to the bedroom and stop next to the bed.",
        "Sofa": "Find the sofa in the living room and stop in front of it.",
    }
    for category in instructions:
        candidates = [
            w for w in targets if w["id"].startswith("wp_" + category.lower() + "_")
        ]
        for waypoint in candidates:
            tasks.append(
                {
                    "id": waypoint["id"],
                    "start_waypoint": "wp_00",
                    "goal_waypoint": waypoint["id"],
                    "instruction": instructions[category],
                    "expected_goal": {
                        k: v for k, v in waypoint["stand_pose"].items() if k != "z"
                    },
                    "task_zh": f"使用 VLN 从客厅前往{waypoint['name']}，并核对停车和地图位姿。",
                }
            )
    (map_dir / "vln_tasks.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2) + "\n"
    )
    report = {
        "map_id": MAP_ID,
        "version": VERSION,
        "rooms": len(house["rooms"]),
        "furniture_groups": len(house["objects"]),
        "semantic_objects": len(objects),
        "waypoints": len(targets),
        "spawn_source_xy": spawn.tolist(),
        "projected_collision_geometries": used,
        "all_rooms_reachable": True,
        "scene": str(scene),
        "grid_shape": list(grid.heights.shape),
        "grid_resolution_m": resolution,
    }
    (map_dir / "generation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=float, default=0.05)
    args = parser.parse_args()
    if not math.isfinite(args.resolution) or args.resolution <= 0:
        parser.error("--resolution must be finite and positive")
    map_dir = ROBOT / f"slam_map_demo/maps/{MAP_ID}/{VERSION}"
    assets = (map_dir / "assets").resolve()
    source = assets / "scenes/procthor-10k-val/val_2_ceiling.xml"
    house = json.loads((source.parent / "val_2.json").read_text())
    metadata = json.loads((source.parent / "val_2_metadata.json").read_text())[
        "objects"
    ]
    for door in house["doors"]:
        if door.get("openable") and door.get("openness") != 1:
            raise ValueError(
                "all navigable interior doors must be authored open before freezing"
            )
    root = frozen_environment(source, assets)
    report = write_map(map_dir, root, house, metadata, args.resolution)
    astar = PROJECT / f"a/slam_map_demo/maps/{MAP_ID}/{VERSION}"
    astar.mkdir(parents=True, exist_ok=True)
    for name in (
        "meta.yaml",
        "waypoints.yaml",
        "rooms.json",
        "objects.json",
        "vln_tasks.json",
        "generation.json",
        "THIRD_PARTY_NOTICES.md",
    ):
        shutil.copy2(map_dir / name, astar / name)
    (astar / "mujoco").mkdir(exist_ok=True)
    shutil.copy2(
        map_dir / "mujoco/collision_hfield.npz", astar / "mujoco/collision_hfield.npz"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
