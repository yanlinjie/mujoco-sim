"""把 r2b_storage 静态地图（KISS-ICP 激光里程计 + 全局精修 + 动态剔除）转成地图包。

- 重力校正：沿用 make_hilti_map_package 的地面拟合，地面归零
- 原点与朝向：机器人起点投影到地面为原点，起始朝向为 +x。
  起点所在的通道前方约 6 m 都是机器人实际开过的地面，正好满足仿真器出生要求
- 范围裁剪：去掉从门口漏出去的稀疏远点，避免高度场被拉得很大
- 点位：沿机器人实际轨迹每隔约 1.5 m 取一个（必然可通行），朝向为行驶方向
- 输出 dense_map.pcd（xyz）+ mesh_input.pcd（XYZRGB，0.03 m 下采样）+ meta.yaml + waypoints.yaml

用法: make_r2b_map_package.py <static_map.npz> <地图包目录>
"""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_hilti_map_package import floor_level, ransac_floor, rotation_to_up, write_pcd  # noqa: E402


def main(src, map_dir):
    map_dir = Path(map_dir)
    map_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(src)
    xyz, rgb, poses = d["xyz"].astype(np.float64), d["rgb"], d["poses"]

    # 1) 重力校正 + 地面归零
    nrm, _ = ransac_floor(xyz)
    tilt = np.degrees(np.arccos(abs(nrm[2])))
    Rg = rotation_to_up(nrm)
    xyz = xyz @ Rg.T
    traj = poses[:, :3, 3] @ Rg.T
    level = floor_level(xyz)
    z0 = np.median(xyz[np.abs(xyz[:, 2] - level) < 0.1, 2])
    xyz[:, 2] -= z0
    traj[:, 2] -= z0
    print(f"地面倾角 {tilt:.2f}°，地面高度 {z0:.3f} m，雷达离地 {traj[0, 2]:.3f} m")

    # 2) 原点 = 起点投影到地面，+x = 起始朝向（水平分量）
    fwd = Rg @ poses[0, :3, 0]
    yaw0 = np.arctan2(fwd[1], fwd[0])
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    shift = np.array([traj[0, 0], traj[0, 1], 0.0])
    xyz = (xyz - shift) @ Rz.T
    traj = (traj - shift) @ Rz.T
    heading = np.array([np.arctan2(*(Rz @ Rg @ T[:3, 0])[[1, 0]]) for T in poses])
    # 建图坐标系（第 0 帧雷达）-> 地图坐标系的刚体变换，供图像库等后续模块换算位姿
    T_map_slam = np.eye(4)
    T_map_slam[:3, :3] = Rz @ Rg
    T_map_slam[:3, 3] = -Rz @ np.array([shift[0], shift[1], z0])
    np.savetxt(map_dir / "map_from_slam.txt", T_map_slam, fmt="%.9f",
               header="4x4 齐次变换：地图坐标 = T @ 建图坐标（第 0 帧雷达坐标系）")
    patch = xyz[(np.hypot(xyz[:, 0], xyz[:, 1]) < 0.5) & (np.abs(xyz[:, 2]) < 0.15)]
    print(f"起始朝向修正 {np.degrees(yaw0):.1f}°；原点 0.5 m 内地面点 {len(patch)}，高度标准差 {patch[:, 2].std() * 100:.1f} cm")

    # 3) 裁剪稀疏远点：按 0.5 m 栅格统计，只保留点数足够的栅格的包围盒
    cell = np.floor(xyz[:, :2] / 0.5).astype(np.int64)
    uc, cnt = np.unique(cell, axis=0, return_counts=True)
    dense = uc[cnt >= 2000]
    lo, hi = dense.min(0) * 0.5 - 0.3, (dense.max(0) + 1) * 0.5 + 0.3
    inside = ((xyz[:, :2] >= lo) & (xyz[:, :2] <= hi)).all(1) & (xyz[:, 2] < 4.0)
    print(f"裁剪到 x[{lo[0]:.1f},{hi[0]:.1f}] y[{lo[1]:.1f},{hi[1]:.1f}]，保留 {inside.sum()} / {len(xyz)} 点")
    xyz, rgb = xyz[inside], rgb[inside]

    # 4) 点云输出
    f32 = [xyz[:, i].astype(np.float32) for i in range(3)]
    write_pcd(map_dir / "dense_map.pcd", [("x", "F"), ("y", "F"), ("z", "F")], f32)
    key = np.floor(xyz / 0.03).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx = np.sort(idx)
    packed = ((rgb[idx, 0].astype(np.uint32) << 16) | (rgb[idx, 1].astype(np.uint32) << 8)
              | rgb[idx, 2].astype(np.uint32)).view(np.float32)
    write_pcd(map_dir / "mesh_input.pcd", [("x", "F"), ("y", "F"), ("z", "F"), ("rgb", "F")],
              [f[idx] for f in f32] + [packed])
    print(f"dense_map.pcd {len(xyz)} 点，mesh_input.pcd {len(idx)} 点")

    # 5) 点位：沿轨迹按 1.5 m 间隔取点
    wps, last = [], None
    for p, yaw in zip(traj, heading):
        if last is not None and np.linalg.norm(p[:2] - last) < 1.5:
            continue
        last = p[:2]
        n = len(wps) + 1
        wps.append({"id": f"wp_{n:02d}", "name": f"corridor {n}",
                    "stand_pose": {"x": round(float(p[0]), 3), "y": round(float(p[1]), 3), "z": 0.0,
                                   "yaw": round(float(yaw), 3)},
                    "lookat": {"x": round(float(p[0] + np.cos(yaw)), 3),
                               "y": round(float(p[1] + np.sin(yaw)), 3), "z": 0.5},
                    "arrive_tolerance": {"xy": 0.15, "yaw": 0.2},
                    "safety_distance": 0.3})
    (map_dir / "waypoints.yaml").write_text(yaml.safe_dump({"waypoints": wps}, sort_keys=False))

    meta = {"map_id": "r2b_storage", "version": map_dir.name, "up_axis": "z", "units": "meter",
            "source": "NVIDIA r2b_storage (ROS2 bag, PandarXT-32 + D455), "
                      "KISS-ICP lidar odometry + global refinement + dynamic removal",
            "gravity_correction_deg": float(round(tilt, 2)),
            "hfield_margin": 40}
    (map_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    np.savetxt(map_dir / "trajectory.txt", np.column_stack([traj, heading]), fmt="%.4f",
               header="建图传感器轨迹（地图坐标系）: x y z yaw")
    print(f"点位 {len(wps)} 个；地图范围 x[{xyz[:, 0].min():.1f},{xyz[:, 0].max():.1f}] "
          f"y[{xyz[:, 1].min():.1f},{xyz[:, 1].max():.1f}] z[{xyz[:, 2].min():.1f},{xyz[:, 2].max():.1f}]")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
