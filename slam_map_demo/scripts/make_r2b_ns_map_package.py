"""把 nerfstudio（splatfacto/3DGS）重建结果转成地图包，格式与 make_r2b_map_package.py 相同。

输入均来自 slam_data/scripts/r2b_ns/ 流程：
- mesh_colored.ply：TSDF 融合网格（全分辨率，地图坐标：第 0 帧 base_link 为原点、+x 朝前、z 朝上、米）
- scene.obj / scene.mtl / scene_texture.png：简化后带纹理的视觉网格（同一坐标系）
- base_trajectory.txt：各帧 base_link 位姿（帧号 x y z yaw）

处理：
1. 地面拟合（沿用 make_hilti_map_package 的 RANSAC 与地面层识别），重力微调并把地面归零；
   再用二次曲面拟合地面并从全部几何中减掉（地面展平），消除视觉位姿高度漂移造成的地面弯曲；
   朝向不变；
2. 原点平移到轨迹上第一个“四周都已重建”的位姿（见 pick_origin）：仿真器总把机器狗放在世界原点，
   而前向相机看不到起点前约 0.8 m 的地面，以起点为原点会让机器狗出生在地图外；
3. dense_map.pcd：在全分辨率网格表面按 1 cm 间距均匀采样，供 map_to_mujoco.py 生成碰撞高度场；
4. 视觉网格与纹理做同样的变换后写入 mesh/；
   采集机器人行驶扫过的空间（轨迹两侧 0.3 m、离地 0.08~1.5 m）内残留的点和三角形一律删除；
5. 点位：第一个即原点（充电桩），其余按 WAYPOINT_HINTS 给出的大致位置和朝向，在附近找离障碍最远的栅格
   作站位；离障碍或未重建区域不足 WAYPOINT_CLEARANCE 时报错。覆盖整条走廊和尽头的丁字路口（含拐弯）。

6. 可选：3DGS 高斯模型（splat_to_map_frame.py 的输出）做同样的变换，写入 gaussians/splat.ply。

用法: make_r2b_ns_map_package.py <mesh 目录> <base_trajectory.txt> <地图包目录> [高斯模型 .ply]
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_hilti_map_package import floor_level, ransac_floor, rotation_to_up, write_pcd  # noqa: E402
from map_to_mujoco import RESOLUTION, build_heightfield  # noqa: E402
from nav_grid import NavGrid  # noqa: E402


SWEEP_RADIUS = 0.3   # 采集机器人车身半宽（米）
SWEEP_Z = (0.08, 1.5)  # 清空的高度范围：地面以上到 map_to_mujoco.py 的天花板截断高度
ORIGIN_CLEARANCE = 0.5     # 原点（出生点）离障碍或未重建区域的最小距离，米
# 点位离障碍或未重建区域的最小距离：Go2 机身半长约 0.35 m，check_scene.py 在机身四周 ±0.3 m（对角 0.42 m）
# 取地面高度，走廊中段一根离路线约 0.4 m 的残留横杆会被误当成脚下地面，故取 0.45 m
WAYPOINT_CLEARANCE = 0.45
WAYPOINT_SNAP = 0.15  # 在大致位置周围多大范围内找站位
# (名称, 大致位置 x y, 朝向°)。位置写在重建坐标系（第 0 帧 base_link，未平移原点）下，原点选取规则改变时不用跟着改。
# 顺序即巡航顺序：沿走廊向前，到丁字路口左转进北侧通道，再折回路口右转到东侧通道尽头。
WAYPOINT_HINTS = [
    ("走廊 2", (2.74, 0.0), 0),
    ("走廊 3", (4.29, -0.1), 0),
    ("走廊 4", (5.54, 0.1), 0),
    ("丁字路口", (6.54, 0.15), 90),
    ("北侧通道尽头", (6.74, 1.4), 90),
    ("东侧通道尽头", (7.54, 0.2), 0),
]


def quadratic(c):
    return lambda px, py: c[0] + c[1] * px + c[2] * py + c[3] * px * px + c[4] * px * py + c[5] * py * py


def shift_quadratic(c, ox, oy):
    """坐标平移 x = x' + ox、y = y' + oy 后，二次曲面在新坐标下的系数。"""
    return np.array([quadratic(c)(ox, oy), c[1] + 2 * c[3] * ox + c[4] * oy,
                     c[2] + c[4] * ox + 2 * c[5] * oy, c[3], c[4], c[5]])


def nav_grid(xyz):
    """按与 map_to_mujoco.py / nav_grid.py 完全相同的规则生成高度场和可通行栅格
    （未重建区域在导航中一律当作障碍）。"""
    heights, lo, observed = build_heightfield(xyz)
    return NavGrid(heights, lo, RESOLUTION, observed)


def snap_stand(grid, hint):
    """hint 周围 WAYPOINT_SNAP 内离障碍最远的栅格，返回 (xy, 离障碍距离)。"""
    r0, c0 = grid.to_cell(hint)
    k = int(np.ceil(WAYPOINT_SNAP / grid.res))
    best = (np.asarray(hint, float), grid.clearance_at(hint))
    for r in range(r0 - k, r0 + k + 1):
        for c in range(c0 - k, c0 + k + 1):
            xy = grid.to_xy((r, c))
            if (np.linalg.norm(xy - hint) <= WAYPOINT_SNAP and grid.inside((r, c))
                    and grid.clearance[r, c] > best[1]):
                best = (xy, grid.clearance[r, c])
    return best


def waypoint(n, name, x, y, yaw):
    return {"id": f"wp_{n:02d}", "name": name,
            "stand_pose": {"x": round(float(x), 3), "y": round(float(y), 3), "z": 0.0, "yaw": round(float(yaw), 3)},
            "lookat": {"x": round(float(x + np.cos(yaw)), 3), "y": round(float(y + np.sin(yaw)), 3), "z": 0.5},
            "arrive_tolerance": {"xy": 0.15, "yaw": 0.2},
            "safety_distance": 0.3}


def pick_origin(clear):
    """轨迹上第一个离障碍和未重建区域都不少于 ORIGIN_CLEARANCE 的位姿的序号。"""
    ok = np.flatnonzero(clear >= ORIGIN_CLEARANCE)
    if not len(ok):
        raise RuntimeError(f"轨迹上没有离障碍 {ORIGIN_CLEARANCE} m 以上的位姿，无法选原点")
    return int(ok[0])


def swept_mask(xyz, traj):
    """落在采集机器人行驶扫过空间内的点。机器人开过的地方必然是空的，这里残留的表面
    来自掩码没覆盖到的运动物体（例如行人抱着的纸箱），会在高度场里变成路中间的障碍。"""
    d, _ = cKDTree(traj[:, :2]).query(xyz[:, :2])
    return (d < SWEEP_RADIUS) & (xyz[:, 2] > SWEEP_Z[0]) & (xyz[:, 2] < SWEEP_Z[1])


def fit_floor_surface(xyz, nz):
    """用水平朝向的地面点拟合二次曲面 z = h(x, y)（Huber 加权迭代最小二乘），返回 h 与系数。

    纯视觉位姿有低频的高度漂移（本数据中视觉轨迹 6.5 m 内抬升约 10 cm，激光轨迹只有 2 cm），
    刚性调平只能去掉线性倾斜，重建出的地面仍会微弯；仿真高度场对地面平整度很敏感（几厘米的台阶
    就会让机器狗打滑），因此把这一低频弯曲从全部几何中减掉。"""
    sel = (np.abs(xyz[:, 2]) < 0.15) & (np.abs(nz) > 0.95)
    x, y, z = xyz[sel, 0], xyz[sel, 1], xyz[sel, 2]
    A = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    w = np.ones_like(z)
    for _ in range(8):
        c, *_ = np.linalg.lstsq(A * w[:, None], z * w, rcond=None)
        r = np.abs(A @ c - z)
        w = np.where(r < 0.02, 1.0, 0.02 / r)

    return quadratic(c), c, sel


def transform_obj(src, dst, T, traj, flatten):
    """变换 OBJ 顶点坐标（刚体变换 + 地面展平），并删除任一顶点落在行驶扫过空间内的三角形。
    只支持本流程 splat_to_mesh.py 写出的格式（v / vt / f a/a b/b c/c）。"""
    v, vt, f, head = [], [], [], []
    for line in open(src):
        if line.startswith("v "):
            v.append(line.split()[1:4])
        elif line.startswith("vt "):
            vt.append(line.split()[1:3])
        elif line.startswith("f "):
            f.append([int(t.split("/")[0]) for t in line.split()[1:4]])
        else:
            head.append(line)
    v = np.array(v, float) @ T[:3, :3].T + T[:3, 3]
    v[:, 2] -= flatten(v[:, 0], v[:, 1])
    f = np.array(f) - 1
    bad = swept_mask(v, traj)[f].any(1)
    f = f[~bad]
    with open(dst, "w") as fo:
        fo.writelines(head)
        fo.write("".join(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in v))
        fo.write("".join(f"vt {a} {b}\n" for a, b in vt))
        fo.write("".join(f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}\n" for a, b, c in f))
    return int(bad.sum())


def transform_splat(src, dst, T, flatten):
    """把地图坐标系（未展平）下的 3DGS 高斯模型做同样的刚体变换与地面展平。
    高斯朝向随刚体旋转一起转；展平只是几厘米的竖直平移，尺度和朝向不受影响。"""
    from plyfile import PlyData, PlyElement
    from scipy.spatial.transform import Rotation

    v = PlyData.read(str(src))["vertex"].data.copy()
    p = np.stack([v["x"], v["y"], v["z"]], -1).astype(float) @ T[:3, :3].T + T[:3, 3]
    p[:, 2] -= flatten(p[:, 0], p[:, 1])
    q = np.stack([v["rot_1"], v["rot_2"], v["rot_3"], v["rot_0"]], -1)  # 文件中顺序为 w x y z
    q = (Rotation.from_matrix(T[:3, :3]) * Rotation.from_quat(q / np.linalg.norm(q, axis=1, keepdims=True))).as_quat()
    v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"] = q[:, 3], q[:, 0], q[:, 1], q[:, 2]
    v["x"], v["y"], v["z"] = p.T.astype(np.float32)
    PlyData([PlyElement.describe(v, "vertex")]).write(str(dst))
    return len(v)


def main(mesh_dir, traj_path, map_dir, splat=None):
    mesh_dir, map_dir = Path(mesh_dir), Path(map_dir)
    (map_dir / "mesh").mkdir(parents=True, exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(str(mesh_dir / "mesh_colored.ply"))
    area = mesh.get_surface_area()
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(int(area / 0.01 ** 2))
    xyz = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    traj_raw = np.loadtxt(traj_path, ndmin=2)
    traj, yaw = traj_raw[:, 1:4], traj_raw[:, 4]
    print(f"网格表面积 {area:.1f} m²，采样 {len(xyz)} 点")

    # 1) 重力微调 + 地面归零（align_metric.py 已按 base_link 做过重力对齐，这里只修残差）
    nrm, _ = ransac_floor(xyz)
    tilt = np.degrees(np.arccos(abs(nrm[2])))
    Rg = rotation_to_up(nrm)
    xyz_r = xyz @ Rg.T
    level = floor_level(xyz_r)
    z0 = np.median(xyz_r[np.abs(xyz_r[:, 2] - level) < 0.1, 2])
    T = np.eye(4)
    T[:3, :3] = Rg
    T[:3, 3] = [0, 0, -z0]
    xyz = xyz_r - [0, 0, z0]
    traj = traj @ Rg.T - [0, 0, z0]
    print(f"地面残余倾角 {tilt:.2f}°，地面高度 {z0:.3f} m → 归零后 base_link 离地 {traj[0, 2]:.3f} m")
    # 地面展平：去掉视觉位姿高度漂移造成的地面低频弯曲
    flatten, coef, floor_sel = fit_floor_surface(xyz, (normals @ Rg.T)[:, 2])
    before = xyz[floor_sel, 2].copy()
    corr = flatten(xyz[:, 0], xyz[:, 1])
    xyz[:, 2] -= corr
    traj[:, 2] -= flatten(traj[:, 0], traj[:, 1])
    after = xyz[floor_sel, 2]
    span = flatten(xyz[floor_sel, 0], xyz[floor_sel, 1])
    print(f"地面展平：修正量范围 {span.min() * 100:+.1f} ~ {span.max() * 100:+.1f} cm；"
          f"地面点高度 5~95% 分位 {np.percentile(before, 5) * 100:+.1f}~{np.percentile(before, 95) * 100:+.1f} cm"
          f" → {np.percentile(after, 5) * 100:+.1f}~{np.percentile(after, 95) * 100:+.1f} cm")

    # 删除行驶扫过空间内的残留表面（平移不影响该判定，先删再选原点，免得残影影响离障碍距离）
    sw = swept_mask(xyz, traj)
    print(f"行驶扫过空间内的残留点 {sw.sum()} 个（运动物体残影），已删除")
    xyz = xyz[~sw & (xyz[:, 2] < 4.0)]

    # 2) 原点平移到四周都已重建的位姿；展平曲面系数随之换算到新坐标
    grid = nav_grid(xyz)
    clear = np.array([grid.clearance_at(p[:2]) for p in traj])
    i0 = pick_origin(clear)
    o = traj[i0, :2].copy()
    xyz[:, :2] -= o
    traj[:, :2] -= o
    T[:2, 3] -= o
    coef = shift_quadratic(coef, *o)
    flatten = quadratic(coef)
    print(f"原点平移到轨迹第 {i0} 帧（原坐标 x={o[0]:+.3f} y={o[1]:+.3f}，离障碍/未重建区域 {clear[i0]:.2f} m）")
    np.savetxt(map_dir / "map_from_slam.txt", T, fmt="%.9f",
               header="4x4 齐次变换：地图坐标 = T @ 重建坐标（第 0 帧 base_link 系，见 align_metric.py），"
                      "之后再按 floor_flatten.txt 做地面展平")
    np.savetxt(map_dir / "floor_flatten.txt", coef[None], fmt="%.9f",
               header="地面展平系数 c0..c5：地图 z = 刚体变换后 z - (c0 + c1 x + c2 y + c3 x² + c4 xy + c5 y²)")
    patch = xyz[(np.hypot(xyz[:, 0], xyz[:, 1]) < 0.5) & (np.abs(xyz[:, 2]) < 0.15)]
    print(f"原点 0.5 m 内地面点 {len(patch)}，高度标准差 {patch[:, 2].std() * 100:.1f} cm")

    # 3) 碰撞用点云
    write_pcd(map_dir / "dense_map.pcd", [("x", "F"), ("y", "F"), ("z", "F")],
              [xyz[:, i].astype(np.float32) for i in range(3)])

    # 4) 视觉网格
    n_bad = transform_obj(mesh_dir / "scene.obj", map_dir / "mesh" / "scene.obj", T, traj, flatten)
    print(f"视觉网格删除行驶扫过空间内的三角形 {n_bad} 个")
    for name in ("scene.mtl", "scene_texture.png"):
        shutil.copy(mesh_dir / name, map_dir / "mesh" / name)

    if splat:
        (map_dir / "gaussians").mkdir(exist_ok=True)
        n = transform_splat(splat, map_dir / "gaussians" / "splat.ply", T, flatten)
        print(f"3DGS 高斯模型 {n} 个已写入 gaussians/splat.ply")

    # 5) 点位：原点 + WAYPOINT_HINTS
    grid = nav_grid(xyz)
    wps = [waypoint(1, "走廊 1（充电桩）", 0.0, 0.0, yaw[i0])]
    for name, hint, yaw_deg in WAYPOINT_HINTS:
        # 重建坐标 → 地图坐标：刚体变换（重力微调只有约 1.5°，对水平位置的影响可忽略）+ 原点平移
        h = (Rg @ [hint[0], hint[1], 0.0])[:2] - o
        xy, c = snap_stand(grid, h)
        if c < WAYPOINT_CLEARANCE:
            raise RuntimeError(f"点位 {name} 附近找不到离障碍 {WAYPOINT_CLEARANCE} m 以上的站位（最多 {c:.2f} m）")
        wps.append(waypoint(len(wps) + 1, name, xy[0], xy[1], np.radians(yaw_deg)))
        print(f"点位 {wps[-1]['id']} {name}: ({xy[0]:+.2f}, {xy[1]:+.2f}) 朝向 {yaw_deg}°，离障碍 {c:.2f} m")
    (map_dir / "waypoints.yaml").write_text(yaml.safe_dump({"waypoints": wps}, sort_keys=False, allow_unicode=True))
    meta = {"map_id": "r2b_storage_ns", "version": map_dir.name, "up_axis": "z", "units": "meter",
            "source": "NVIDIA r2b_storage (ROS2 bag, Hawk stereo + D455 depth), nerfstudio: COLMAP poses (metric scale from stereo baseline, gravity from tf_static) + splatfacto (3DGS) with D455 depth supervision and person masks + TSDF fusion of rendered depth + floor flattening",
            "gravity_correction_deg": float(round(tilt, 2)),
            "hfield_margin": 40}
    if splat:
        meta["gaussians"] = "gaussians/splat.ply（3DGS 高斯模型，地图坐标系，可用 3DGS 查看器打开；仿真中未使用）"
    (map_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
    np.savetxt(map_dir / "trajectory.txt", np.column_stack([traj, yaw]), fmt="%.4f",
               header="建图轨迹（地图坐标系，base_link）: x y z yaw")
    print(f"点位 {len(wps)} 个；dense_map.pcd {len(xyz)} 点；地图范围 x[{xyz[:, 0].min():.1f},{xyz[:, 0].max():.1f}] "
          f"y[{xyz[:, 1].min():.1f},{xyz[:, 1].max():.1f}] z[{xyz[:, 2].min():.1f},{xyz[:, 2].max():.1f}]")


if __name__ == "__main__":
    main(*sys.argv[1:5])
