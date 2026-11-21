"""把 room01 激光-双目-3DGS 耦合重建结果打包成地图包（格式同 make_mesh_map_package.py）。

输入（slam_data/scripts/room01_fusion 流程的输出）:
- <补全目录>/scene.obj + scene_texture.png：3DGS 纹理网格 + 激光补面（地面、墙根），已调平、地面 z≈0；
- <补全目录>/collision_points.ply：激光网格表面采样点，作碰撞高度场（激光几何完整，无 3DGS 悬空碎片）；
- <lio 目录>/poses_level.npz、calib_base.json：XT16 轨迹与雷达→机身外参，用于原点和点位。

处理:
1. 重力微调：碰撞点 RANSAC 拟合地面，转平并把地面归零（激光地图已调平，这里只是复核）；
2. 原点：机器狗采集起点，若离障碍不足 ORIGIN_CLEARANCE，取轨迹上最早满足条件的位置（仿真器总把机器狗放在原点、朝 +x）；
3. 点位：沿采集轨迹每走 WAYPOINT_STEP 米取一个，在附近 WAYPOINT_SNAP 内找离障碍最远的栅格，
   离障碍不足 WAYPOINT_CLEARANCE 或离已有点位太近的跳过；朝向取机身当时的朝向；
4. 写 dense_map.pcd、mesh/、trajectory.txt、waypoints.yaml、meta.yaml、map_from_slam.txt。

用法: make_room01_fusion_map_package.py <补全目录> <lio 目录> <地图包目录> [map_id, 默认 room1110]
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_hilti_map_package import rotation_to_up, write_pcd  # noqa: E402
from map_to_mujoco import RESOLUTION, build_heightfield  # noqa: E402
from nav_grid import NavGrid  # noqa: E402

ORIGIN_CLEARANCE = 0.5
WAYPOINT_STEP = 3.0
WAYPOINT_SNAP = 0.3
WAYPOINT_CLEARANCE = 0.45
WAYPOINT_MIN_GAP = 1.5


def snap(grid, xy, radius):
    """在 xy 周围 radius 内找离障碍最远的栅格，返回 (xy, clearance)。"""
    r0, c0 = grid.to_cell(xy)
    k = int(np.ceil(radius / RESOLUTION))
    best = (None, -1.0)
    for r in range(r0 - k, r0 + k + 1):
        for c in range(c0 - k, c0 + k + 1):
            if not grid.inside((r, c)) or (r - r0) ** 2 + (c - c0) ** 2 > k * k:
                continue
            if grid.clearance[r, c] > best[1]:
                best = (np.array(grid.to_xy((r, c))), float(grid.clearance[r, c]))
    return best


def main(fill_dir, lio_dir, map_dir, map_id="room1110"):
    fill_dir, lio_dir, map_dir = Path(fill_dir), Path(lio_dir), Path(map_dir)
    (map_dir / "mesh").mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(o3d.io.read_point_cloud(str(fill_dir / "collision_points.ply")).points)
    # 激光地图已调平、地面在 z≈0；天花板点更多，不能用最密水平层找地面，只在 0.15 m 以下拟合
    low = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[np.abs(xyz[:, 2]) < 0.15]))
    pl, _ = low.segment_plane(0.02, 3, 2000)
    nrm = np.array(pl[:3]) * np.sign(pl[2])
    Rg = rotation_to_up(nrm)
    xyz = xyz @ Rg.T
    z0 = float(np.median(xyz[np.abs(xyz[:, 2]) < 0.05, 2]))
    tilt = float(np.degrees(np.arccos(abs(nrm[2]))))
    xyz[:, 2] -= z0
    print(f"重力微调 {tilt:.2f}°，地面高度 {z0:+.3f} m 归零；碰撞点 {len(xyz)}")

    # 机身轨迹（雷达位姿 × 雷达→机身外参）
    P = np.load(lio_dir / "poses_level.npz")
    Rbl = np.array(json.load(open(lio_dir / "calib_base.json"))["R_base_lidar"])
    Tl = P["T"]
    pos = Tl[:, :3, 3] @ Rg.T
    pos[:, 2] -= z0
    fwd = (Rg @ (Tl[:, :3, :3] @ Rbl.T)[:, :, 0].T).T      # 机身 +x 方向在世界系
    yaw = np.arctan2(fwd[:, 1], fwd[:, 0])

    heights, lo, observed = build_heightfield(xyz)
    grid = NavGrid(heights, lo, RESOLUTION, observed)
    oi = None
    for i in range(len(pos)):
        if grid.clearance_at(pos[i, :2]) >= ORIGIN_CLEARANCE:
            oi = i
            break
    o, cl = snap(grid, pos[oi, :2], 0.1)
    print(f"原点取轨迹第 {oi} 帧附近 ({o[0]:+.2f}, {o[1]:+.2f})，离障碍 {cl:.2f} m（起点离障碍 {grid.clearance_at(pos[0, :2]):.2f} m）")

    T = np.eye(4)
    T[:3, :3] = Rg
    T[:3, 3] = [-o[0], -o[1], -z0]
    np.savetxt(map_dir / "map_from_slam.txt", T, fmt="%.9f", header="4x4 齐次变换：地图坐标 = T @ 调平后激光/重建坐标")
    xyz[:, :2] -= o
    pos[:, :2] -= o
    write_pcd(map_dir / "dense_map.pcd", [("x", "F"), ("y", "F"), ("z", "F")],
              [xyz[:, i].astype(np.float32) for i in range(3)])

    # 视觉网格：只变换顶点
    out = []
    for l in open(fill_dir / "scene.obj"):
        if l.startswith("v "):
            v = np.array(l.split()[1:4], float) @ Rg.T
            out.append(f"v {v[0] - o[0]:.4f} {v[1] - o[1]:.4f} {v[2] - z0:.4f}\n")
        elif not l.startswith(("mtllib", "usemtl")):
            out.append(l)
    (map_dir / "mesh" / "scene.obj").write_text("".join(out))
    shutil.copy(fill_dir / "scene_texture.png", map_dir / "mesh" / "scene_texture.png")

    np.savetxt(map_dir / "trajectory.txt", np.column_stack([P["t"], pos, yaw]), fmt="%.4f",
               header="机身轨迹（地图坐标）：时间戳 x y z yaw")

    # 点位
    grid2 = NavGrid(heights, lo - o, RESOLUTION, observed)
    wps = [{"id": "wp_01", "name": "原点（采集起点）",
            "stand_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "lookat": {"x": 1.0, "y": 0.0, "z": 0.5},
            "arrive_tolerance": {"xy": 0.15, "yaw": 0.2}, "safety_distance": 0.3}]
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1))])
    taken = [np.zeros(2)]
    for target in np.arange(WAYPOINT_STEP, s[-1], WAYPOINT_STEP):
        i = int(np.searchsorted(s, target))
        xy, cl = snap(grid2, pos[i, :2], WAYPOINT_SNAP)
        if xy is None or cl < WAYPOINT_CLEARANCE or min(np.linalg.norm(xy - t) for t in taken) < WAYPOINT_MIN_GAP:
            continue
        taken.append(xy)
        k = len(wps) + 1
        wps.append({"id": f"wp_{k:02d}", "name": f"轨迹点 {k}",
                    "stand_pose": {"x": round(float(xy[0]), 3), "y": round(float(xy[1]), 3), "z": 0.0,
                                   "yaw": round(float(yaw[i]), 3)},
                    "lookat": {"x": round(float(xy[0] + np.cos(yaw[i])), 3), "y": round(float(xy[1] + np.sin(yaw[i])), 3), "z": 0.5},
                    "arrive_tolerance": {"xy": 0.15, "yaw": 0.2}, "safety_distance": 0.3})
    (map_dir / "waypoints.yaml").write_text(yaml.safe_dump({"waypoints": wps}, sort_keys=False, allow_unicode=True))
    print(f"点位 {len(wps)} 个")
    meta = {"map_id": map_id, "version": map_dir.name, "up_axis": "z", "units": "meter",
            "source": "room01：XT16 激光几何先验 + D435i 双目深度 + 3DGS 外观学习耦合重建；地面与墙根由激光网格补全",
            "gravity_correction_deg": round(tilt, 2), "hfield_margin": 40}
    (map_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
    print(f"地图范围 x[{xyz[:, 0].min():.1f},{xyz[:, 0].max():.1f}] y[{xyz[:, 1].min():.1f},{xyz[:, 1].max():.1f}]")


if __name__ == "__main__":
    main(*sys.argv[1:5])
