"""把自建 SLAM（slam_data/scripts/go2_room/）的地图转成地图包，格式与 make_r2b_map_package.py 相同。

输入：refine_backend.py 输出的 go2_room_selfbuilt_map.pcd（XYZ + 强度存在颜色通道）与
go2_room_selfbuilt_poses_tum.txt（每帧雷达位姿），以及雷达 -> 机身外参 T_base_lidar.txt（calib_lidar_base.py）。

1. 重力对齐：拟合主地面（沿用 make_hilti_map_package 的 RANSAC 与地面层识别），地面归零；
2. 原点与朝向：机器狗起点（机身中心）投影到地面为原点，机身起始朝向为 +x；
3. 范围裁剪：按 0.5 m 栅格统计点数，只保留点数足够的栅格的包围盒（去掉门窗外漏出的稀疏远点）；
   并清空机器狗自己走过的空间（机身中心 0.4 m 内、离地 0.08~0.6 m）；
4. 点位：沿机身轨迹每 1.5 m 取一个，朝向为机身朝向；0.45 m 内有障碍时沿轨迹顺延；
5. 输出 dense_map.pcd（xyz）+ mesh_input.pcd（XYZRGB，灰度为激光强度，3 cm 下采样）+ meta.yaml
   + waypoints.yaml + trajectory.txt（雷达轨迹，供 mesh_map_trim.py 判断法向与天花板）。

用法: make_go2_room_selfbuilt_package.py <地图 pcd> <位姿 TUM> <T_base_lidar.txt> <地图包目录> [--frame <yaml>]
    --frame：自定原点与 +x 方向（pick_map_frame.py 在默认坐标系的地图包上点选得到的 yaml），
             在第 2 步的默认坐标系上再做一次平移 + 绕 z 旋转；z 轴与地面归零不变。
"""
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_hilti_map_package import floor_level, ransac_floor, rotation_to_up, write_pcd  # noqa: E402

SWEEP_R = 0.4          # 机身扫过空间的半径（米），Go2 机身长约 0.7 m（半长 0.35 m）
SWEEP_Z = (0.08, 0.6)  # 清空的高度范围（米）


def main(map_pcd, pose_file, tbl_file, map_dir, frame_yaml=None):
    map_dir = Path(map_dir)
    map_dir.mkdir(parents=True, exist_ok=True)
    pc = o3d.io.read_point_cloud(str(map_pcd))
    xyz = np.asarray(pc.points)
    inten = np.asarray(pc.colors)[:, 0] if pc.has_colors() else np.full(len(xyz), 0.5)
    P = np.loadtxt(pose_file, ndmin=2)
    T_BL = np.loadtxt(tbl_file)
    T_WL = np.array([np.vstack([np.c_[R.from_quat(p[4:8]).as_matrix(), p[1:4]], [0, 0, 0, 1]]) for p in P])
    T_WB = T_WL @ np.linalg.inv(T_BL)

    # 1) 重力对齐 + 地面归零
    nrm, _ = ransac_floor(xyz)
    tilt = np.degrees(np.arccos(abs(nrm[2])))
    Rg = rotation_to_up(nrm)
    xyz = xyz @ Rg.T
    level = floor_level(xyz)
    z0 = np.median(xyz[np.abs(xyz[:, 2] - level) < 0.1, 2])
    xyz[:, 2] -= z0

    # 2) 原点 = 机身起点投影到地面，+x = 机身起始朝向
    base_p = (T_WB[:, :3, 3] @ Rg.T) - [0, 0, z0]
    fwd = Rg @ T_WB[0, :3, 0]
    yaw0 = np.arctan2(fwd[1], fwd[0])
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    shift = np.array([base_p[0, 0], base_p[0, 1], 0.0])
    xyz = (xyz - shift) @ Rz.T
    A = Rz @ Rg
    T_map_slam = np.eye(4)
    T_map_slam[:3, :3] = A
    T_map_slam[:3, 3] = -Rz @ np.array([shift[0], shift[1], z0])
    frame_desc = "原点 = 机身起点投影到地面，+x = 机身起始朝向"
    if frame_yaml:
        # 自定坐标系：在上面的默认坐标系里再平移到 origin、绕 z 转 yaw（pick_map_frame.py 的输出）
        f = yaml.safe_load(Path(frame_yaml).read_text())
        o = np.array([*f["origin"], 0.0])
        d = np.array(f["x_toward"]) - o[:2]
        yu = np.arctan2(d[1], d[0])
        cu, su = np.cos(-yu), np.sin(-yu)
        Ru = np.array([[cu, -su, 0], [su, cu, 0], [0, 0, 1]])
        T_user = np.eye(4)
        T_user[:3, :3] = Ru
        T_user[:3, 3] = -Ru @ o
        xyz = (xyz - o) @ Ru.T
        T_map_slam = T_user @ T_map_slam
        frame_desc = (f"自定：默认坐标系（机身起点、起始朝向）下的点 ({o[0]:.4f}, {o[1]:.4f}) 为原点，"
                      f"指向 ({f['x_toward'][0]:.4f}, {f['x_toward'][1]:.4f}) 为 +x（转 {np.degrees(yu):.3f}°）")
        print(frame_desc)
    np.savetxt(map_dir / "map_from_slam.txt", T_map_slam, fmt="%.9f",
               header="4x4 齐次变换：地图坐标 = T @ 自建 SLAM 坐标（go2_room_selfbuilt_poses_tum.txt 所在坐标系）")
    TB = T_map_slam @ T_WB
    TL = T_map_slam @ T_WL
    base_yaw = np.arctan2(TB[:, 1, 0], TB[:, 0, 0])
    print(f"地面倾角 {tilt:.2f}°，地面高度 {z0:.3f} m；起始朝向修正 {np.degrees(yaw0):.1f}°；"
          f"机身离地 中位数 {np.median(TB[:, 2, 3]):.3f} m，雷达离地 中位数 {np.median(TL[:, 2, 3]):.3f} m")

    # 3) 裁剪
    cell = np.floor(xyz[:, :2] / 0.5).astype(np.int64)
    uc, cnt = np.unique(cell, axis=0, return_counts=True)
    dense = uc[cnt >= 2000]  # 与 make_r2b_map_package.py 相同的阈值
    lo, hi = dense.min(0) * 0.5 - 0.3, (dense.max(0) + 1) * 0.5 + 0.3
    inside = ((xyz[:, :2] >= lo) & (xyz[:, :2] <= hi)).all(1) & (xyz[:, 2] < 3.5) & (xyz[:, 2] > -0.5)
    print(f"裁剪到 x[{lo[0]:.1f},{hi[0]:.1f}] y[{lo[1]:.1f},{hi[1]:.1f}]，保留 {inside.sum()} / {len(xyz)} 点")
    xyz, inten = xyz[inside], inten[inside]
    # 清空机器狗自己走过的空间：机身中心 SWEEP_R 内、离地 SWEEP_Z 的点必然不是静止障碍，
    # 来自开头站起时的腿、走动的人、后来关上的门等（与 make_r2b_ns_map_package.py 的做法相同）
    dist, _ = cKDTree(TB[:, :2, 3]).query(xyz[:, :2])
    swept = (dist < SWEEP_R) & (xyz[:, 2] > SWEEP_Z[0]) & (xyz[:, 2] < SWEEP_Z[1])
    print(f"机身扫过空间内的点 {swept.sum()} 个，已删除")
    xyz, inten = xyz[~swept], inten[~swept]

    # 4) 点云输出
    f32 = [xyz[:, i].astype(np.float32) for i in range(3)]
    write_pcd(map_dir / "dense_map.pcd", [("x", "F"), ("y", "F"), ("z", "F")], f32)
    key = np.floor(xyz / 0.03).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx = np.sort(idx)
    g = (np.clip(0.35 + 0.6 * inten[idx] / max(np.percentile(inten, 99), 1e-6), 0, 1) * 255).astype(np.uint32)
    packed = ((g << 16) | (g << 8) | g).view(np.float32)
    write_pcd(map_dir / "mesh_input.pcd", [("x", "F"), ("y", "F"), ("z", "F"), ("rgb", "F")],
              [f[idx] for f in f32] + [packed])
    print(f"dense_map.pcd {len(xyz)} 点，mesh_input.pcd {len(idx)} 点")

    # 5) 点位（机身轨迹）
    # 净空检查（与 make_hilti_map_package.py 相同的规则）：点位 0.45 m 内不能有离地 0.12~1.5 m 的点，
    # 否则机器狗站立时腿会碰到旁边的物体；不满足就沿轨迹顺延
    obst = xyz[(xyz[:, 2] > 0.12) & (xyz[:, 2] < 1.5)]
    obst_tree = cKDTree(obst[:, :2])
    wps, last, n_skip = [], None, 0
    for p, yaw in zip(TB[:, :3, 3], base_yaw):
        if last is not None and np.linalg.norm(p[:2] - last) < 1.5:
            continue
        if len(obst_tree.query_ball_point(p[:2], 0.45)) > 20:
            n_skip += 1
            continue
        last = p[:2]
        n = len(wps) + 1
        wps.append({"id": f"wp_{n:02d}", "name": f"room {n}",
                    "stand_pose": {"x": round(float(p[0]), 3), "y": round(float(p[1]), 3), "z": 0.0,
                                   "yaw": round(float(yaw), 3)},
                    "lookat": {"x": round(float(p[0] + np.cos(yaw)), 3),
                               "y": round(float(p[1] + np.sin(yaw)), 3), "z": 0.5},
                    "arrive_tolerance": {"xy": 0.15, "yaw": 0.2},
                    "safety_distance": 0.3})
    (map_dir / "waypoints.yaml").write_text(yaml.safe_dump({"waypoints": wps}, sort_keys=False))
    meta = {"map_id": "go2_room_selfbuilt", "version": map_dir.name, "up_axis": "z", "units": "meter",
            "source": "Go2 bag room-19700102-135516-vpn1 (Unitree L1 lidar + /utlidar/imu gyro + /utlidar/robot_odom "
                      "leg odometry); self-built SLAM: gyro + leg-odometry aided scan-to-map ICP front end, "
                      "loop closure + pose graph + global refinement back end (slam_data/scripts/go2_room). "
                      "Unitree onboard SLAM (/uslam/*) NOT used for mapping.",
            "gravity_correction_deg": float(round(tilt, 2)),
            "frame": frame_desc,
            "hfield_margin": 40,
            # L1 测距噪声使地面层厚约 20 cm，高度场按 60 分位数取栅格高度（墙体栅格仍接近顶部高度；
            # 80 分位数时地面噪声在 5 cm 左右，会被识别成一片片“地毯”台阶）
            "hfield_height_percentile": 60}
    (map_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    np.savetxt(map_dir / "trajectory.txt", np.column_stack([TL[:, :3, 3], np.arctan2(TL[:, 1, 0], TL[:, 0, 0])]),
               fmt="%.4f", header="建图雷达轨迹（地图坐标系）: x y z yaw")
    np.savetxt(map_dir / "base_trajectory.txt", np.column_stack([TB[:, :3, 3], base_yaw]), fmt="%.4f",
               header="机身轨迹（地图坐标系）: x y z yaw")
    print(f"点位 {len(wps)} 个（净空不足顺延 {n_skip} 次）；地图范围 x[{xyz[:, 0].min():.1f},{xyz[:, 0].max():.1f}] "
          f"y[{xyz[:, 1].min():.1f},{xyz[:, 1].max():.1f}] z[{xyz[:, 2].min():.1f},{xyz[:, 2].max():.1f}]")


if __name__ == "__main__":
    args = sys.argv[1:]
    frame = None
    if "--frame" in args:
        i = args.index("--frame")
        frame = args[i + 1]
        del args[i:i + 2]
    main(*args[:4], frame_yaml=frame)
