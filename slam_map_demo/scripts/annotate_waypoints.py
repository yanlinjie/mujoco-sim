"""为地图包标注点位库 waypoints.yaml。

真实流程中，点位的语义（“餐桌”“沙发”）由人在地图上标注；站位则应保证机器人能安全到达。
这里对每个目标给出要看向的物体位置 lookat 和一个大致站位 hint，脚本在 hint 附近
自动寻找四周留有足够空间（可原地转身）的站位，朝向对准 lookat，
并计算点位之间的相邻关系 neighbors（规划路线较短的点位对），供上层做路径规划。

用法：
    python -I annotate_waypoints.py <地图包目录>
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I 模式下需显式加入本目录
from map_to_mujoco import build_heightfield, read_pcd_xyz, RESOLUTION
from nav_grid import NavGrid, ROBOT_RADIUS

STAND_CLEARANCE = 0.45   # 站位离障碍的最小距离，Go2 半长约 0.35 m，留出原地转身空间
SEARCH_RADIUS = 1.2      # 在 hint 周围多大范围内找站位
NEIGHBOR_MAX_PATH = 4.5  # 规划路线不超过此长度的两个点位记为相邻

# (名称, 语义标签, 要看向的位置 lookat, 大致站位 hint)，map 坐标系，单位米。
# 顺序即默认巡航顺序：餐厅 → 左侧区域 → 客厅。
TARGETS = [
    ("餐桌北侧", ["dining_table"], (1.5, -2.0), (1.5, 0.2)),
    ("餐厅东侧窗边", ["window", "corner"], (2.9, -2.0), (2.4, -2.0)),
    ("餐边柜", ["sideboard"], (1.6, -4.7), (1.6, -3.9)),
    ("餐厅西南角", ["corner"], (-0.8, -5.0), (-0.2, -4.3)),
    ("餐厅与左侧区域之间的通道口", ["doorway"], (-2.0, 0.45), (-0.8, 0.45)),
    ("左侧区域西南角", ["corner"], (-4.8, -3.9), (-4.1, -3.2)),
    ("左侧区域西北角", ["corner"], (-4.8, 0.9), (-4.1, 0.3)),
    ("客厅西北角", ["living_room", "corner", "plant"], (-4.8, 5.5), (-4.0, 4.4)),
    ("沙发前", ["living_room", "sofa"], (-1.2, 4.5), (-1.6, 3.7)),
    ("茶几西侧", ["living_room", "coffee_table"], (-0.6, 2.8), (-1.9, 2.8)),
    ("电视柜前", ["living_room", "tv_stand"], (-0.6, 1.2), (-0.6, 1.9)),
    ("客厅东侧（沙发后）", ["living_room", "corner"], (2.9, 3.0), (2.2, 3.0)),
]


def find_stand_xy(grid, hint, reachable):
    """在 hint 附近找离障碍足够远、且与充电桩连通的栅格，取离 hint 最近的一个。"""
    ok = (grid.clearance >= STAND_CLEARANCE) & reachable
    rows, cols = np.nonzero(ok)
    xy = grid.lo + np.stack([cols, rows], axis=1) * grid.res
    d = np.linalg.norm(xy - np.asarray(hint), axis=1)
    if d.size == 0 or d.min() > SEARCH_RADIUS:
        return None
    return xy[d.argmin()]


def path_length(path):
    return float(sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:])))


def main(map_dir):
    map_dir = Path(map_dir)
    meta = yaml.safe_load(open(map_dir / "meta.yaml"))
    heights, lo, observed = build_heightfield(read_pcd_xyz(map_dir / "dense_map.pcd"))
    grid = NavGrid(heights, lo, RESOLUTION, observed)

    labels, _ = ndimage.label(grid.clearance > ROBOT_RADIUS)
    reachable = labels == labels[grid.to_cell((0.0, 0.0))]

    waypoints = [{
        "id": "wp_0000",
        "name": "充电桩（地图原点）",
        "semantic_tags": ["dock"],
        "stand_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
    }]
    for name, tags, lookat, hint in TARGETS:
        xy = find_stand_xy(grid, hint, reachable)
        if xy is None:
            print(f"跳过 {name}：hint 附近 {SEARCH_RADIUS} m 内找不到合适站位")
            continue
        yaw = float(np.arctan2(lookat[1] - xy[1], lookat[0] - xy[0]))
        waypoints.append({
            "id": f"wp_{len(waypoints):04d}",
            "name": name,
            "semantic_tags": tags,
            "stand_pose": {"x": round(float(xy[0]), 3), "y": round(float(xy[1]), 3),
                           "z": 0.0, "yaw": round(yaw, 3)},
            "lookat": {"x": lookat[0], "y": lookat[1]},
        })

    for wp in waypoints:
        wp["neighbors"] = []
        wp["arrive_tolerance"] = {"xy": 0.15, "yaw": 0.15}
    for i, wa in enumerate(waypoints):
        for wb in waypoints[i + 1:]:
            pa = (wa["stand_pose"]["x"], wa["stand_pose"]["y"])
            pb = (wb["stand_pose"]["x"], wb["stand_pose"]["y"])
            if np.hypot(pb[0] - pa[0], pb[1] - pa[1]) > NEIGHBOR_MAX_PATH:
                continue
            path = grid.plan(pa, pb)
            if path is not None and path_length(path) <= NEIGHBOR_MAX_PATH:
                wa["neighbors"].append(wb["id"])
                wb["neighbors"].append(wa["id"])

    with open(map_dir / "waypoints.yaml", "w") as f:
        yaml.safe_dump({"map_id": meta["map_id"], "map_version": meta["version"], "waypoints": waypoints},
                       f, allow_unicode=True, sort_keys=False)
    for wp in waypoints:
        p = wp["stand_pose"]
        print(f"{wp['id']} {wp['name']}: ({p['x']:+.2f}, {p['y']:+.2f}) 朝向 {np.degrees(p['yaw']):+.0f}°，"
              f"离障碍 {grid.clearance_at((p['x'], p['y'])):.2f} m，相邻 {wp['neighbors']}")


if __name__ == "__main__":
    main(sys.argv[1])
