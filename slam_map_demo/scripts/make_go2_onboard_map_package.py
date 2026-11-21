"""把 Go2 机载建图（uslam）发布的 /uslam/cloud_map 转成 slam_map_demo 地图包。

输入是 extract 出来的 cloud_map.npz（x y z intensity，odom 坐标系）和 frontend_odom.npy
（t x y z qx qy qz qw，机身位姿）。机载地图约 6 万点，过于稀疏，不适合泊松重建，
因此可视化 Mesh 直接用 5 cm 体素方块拼成，只保留外露面，并按高度贴色带纹理；
地面点本身很稀疏，直接画成体素会满地黑缝，改为把扫描到的地面区域填实，画成一层灰色平面。

- 重力校正：RANSAC 拟合地面平面，把法向转到 +z，地面归零
- 原点与朝向：机身起点投影到地面为原点，起始朝向为 +x
- 去噪：去掉地面以下的点（玻璃/镜面反射造成的虚像）和孤立体素
- 点位：沿实际轨迹每隔约 1.5 m 取一个

用法: make_go2_onboard_map_package.py <cloud_map.npz> <frontend_odom.npy> <地图包目录>
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_hilti_map_package as hilti  # noqa: E402
from make_hilti_map_package import floor_level, rotation_to_up, write_pcd  # noqa: E402

VOXEL = 0.05
BELOW_FLOOR = -0.08   # 低于此高度的点视为反射虚像
TOP = 3.0
FLOOR_TOP = 0.08     # 低于此高度的点视为地面，画成灰色平面而不是体素
COLOR_LO, COLOR_HI = 0.0, 2.2  # 色带覆盖的高度范围
GREY_U = 0.02        # 贴图最左侧一段是灰色，供地面使用
COLOR_U0 = 0.08      # 色带从这里开始


def quat_to_yaw(q):
    x, y, z, w = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def colormap_texture(path, n=256):
    """左侧一小段灰色（地面），其余为 蓝→青→绿→黄→红 的高度色带，写成 n x 8 的贴图。"""
    stops = np.array([[0.25, 0.30, 0.85], [0.10, 0.75, 0.90], [0.20, 0.80, 0.30],
                      [0.95, 0.85, 0.15], [0.90, 0.25, 0.15]])
    t = np.linspace(0, 1, n) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - i)[:, None]
    rgb = stops[i] * (1 - f) + stops[i + 1] * f
    rgb[:int(COLOR_U0 * n)] = 0.62
    img = np.repeat((rgb * 255).astype(np.uint8)[None], 8, axis=0)
    Image.fromarray(img).save(path)


def voxel_mesh_obj(path, occ, lo, floor):
    """occ 为 [nx, ny, nz] 布尔体素，只输出与空体素相邻的面；每个面 4 个独立顶点，保证法向为平面法向。
    floor 为 [nx, ny] 布尔地面掩码，与 occ 共用 xy 栅格，每格输出一个 z=0 处朝上的方片。"""
    pad = np.pad(occ, 1)
    faces_v, faces_uv = [], []
    # (轴, 方向, 该面四个角相对体素最小角的偏移，逆时针朝外)
    corners = {
        (0, 1): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
        (0, -1): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
        (1, 1): [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
        (1, -1): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        (2, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
        (2, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
    }
    idx = np.argwhere(occ)
    for (axis, sign), cs in corners.items():
        nb = idx + 1
        nb[:, axis] += sign
        exposed = idx[~pad[nb[:, 0], nb[:, 1], nb[:, 2]]]
        zc = lo[2] + (exposed[:, 2] + 0.5) * VOXEL
        u = COLOR_U0 + np.clip((zc - COLOR_LO) / (COLOR_HI - COLOR_LO), 0.01, 0.99) * (0.99 - COLOR_U0)
        for c in cs:
            faces_v.append((exposed + np.array(c)) * VOXEL + lo)
        faces_uv.append(u)
    # 重新排列成 [面, 4, 3]
    verts, uvs = [], []
    k = 0
    for f in range(6):
        group = np.stack(faces_v[k:k + 4], axis=1)
        verts.append(group.reshape(-1, 3))
        uvs.append(np.repeat(faces_uv[f], 4))
        k += 4
    cells = np.argwhere(floor)
    quad = np.array([(0, 0), (1, 0), (1, 1), (0, 1)])
    fv = (cells[:, None, :] + quad[None]) * VOXEL + lo[:2]
    verts.append(np.concatenate([fv, np.full(fv.shape[:2] + (1,), 0.002)], axis=2).reshape(-1, 3))
    uvs.append(np.full(len(cells) * 4, GREY_U))
    verts = np.concatenate(verts)
    uvs = np.concatenate(uvs)
    nface = len(verts) // 4
    with open(path, "w") as fo:
        fo.write("# Go2 机载地图体素网格\n")
        np.savetxt(fo, verts, fmt="v %.4f %.4f %.4f")
        np.savetxt(fo, np.column_stack([uvs, np.full_like(uvs, 0.5)]), fmt="vt %.4f %.4f")
        base = np.arange(nface)[:, None] * 4 + np.arange(4) + 1
        # 拆成两个三角形
        tri = np.concatenate([base[:, [0, 1, 2]], base[:, [0, 2, 3]]])
        np.savetxt(fo, np.column_stack([tri[:, 0], tri[:, 0], tri[:, 1], tri[:, 1], tri[:, 2], tri[:, 2]]),
                   fmt="f %d/%d %d/%d %d/%d")
    return nface


def ransac_floor(xyz):
    """沿用 make_hilti_map_package.ransac_floor；它固定抽 5 万个地面点，
    机载地图点数不够，这里把地面带内的点重复到足够数量再交给它（重复点不影响平面拟合结果）。"""
    level = floor_level(xyz)
    band = (xyz[:, 2] > level - 0.25) & (xyz[:, 2] < level + 0.25)
    reps = int(np.ceil(50000 / band.sum()))
    return hilti.ransac_floor(np.tile(xyz, (reps, 1)) if reps > 1 else xyz)


def main(cloud_npz, odom_npy, map_dir):
    map_dir = Path(map_dir)
    (map_dir / "mesh").mkdir(parents=True, exist_ok=True)
    d = np.load(cloud_npz)
    xyz = np.column_stack([d["x"], d["y"], d["z"]]).astype(np.float64)
    odom = np.load(odom_npy)
    traj = odom[:, 1:4].copy()
    yaw = quat_to_yaw(odom[:, 4:8])

    # 1) 重力校正 + 地面归零
    nrm, _ = ransac_floor(xyz)
    tilt = np.degrees(np.arccos(abs(nrm[2])))
    Rg = rotation_to_up(nrm)
    xyz = xyz @ Rg.T
    traj = traj @ Rg.T
    level = floor_level(xyz)
    z0 = np.median(xyz[np.abs(xyz[:, 2] - level) < 0.1, 2])
    xyz[:, 2] -= z0
    traj[:, 2] -= z0
    print(f"地面倾角 {tilt:.2f}°，地面高度 {z0:.3f} m，机身离地 {np.median(traj[:, 2]):.3f} m")

    # 2) 原点 = 起点投影到地面，+x = 起始朝向
    yaw0 = yaw[0]
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    shift = np.array([traj[0, 0], traj[0, 1], 0.0])
    xyz = (xyz - shift) @ Rz.T
    traj = (traj - shift) @ Rz.T
    heading = yaw - yaw0
    T = np.eye(4)
    T[:3, :3] = Rz @ Rg
    T[:3, 3] = -Rz @ np.array([shift[0], shift[1], z0])
    np.savetxt(map_dir / "map_from_slam.txt", T, fmt="%.9f",
               header="4x4 齐次变换：地图坐标 = T @ 机载 odom 坐标")

    # 3) 去掉地面以下的反射虚像和过高的点
    keep = (xyz[:, 2] > BELOW_FLOOR) & (xyz[:, 2] < TOP)
    print(f"去掉地面以下/过高的点 {(~keep).sum()} 个，保留 {keep.sum()}")
    xyz = xyz[keep]

    # 4) 体素化并去掉孤立体素（26 邻域内没有其他体素）
    lo = np.floor(xyz.min(0) / VOXEL) * VOXEL
    ijk = np.floor((xyz - lo) / VOXEL).astype(int)
    occ = np.zeros(ijk.max(0) + 1, dtype=bool)
    occ[tuple(ijk.T)] = True
    nb = ndimage.convolve(occ.astype(np.uint8), np.ones((3, 3, 3), np.uint8), mode="constant") - occ
    isolated = occ & (nb == 0)
    occ &= ~isolated
    print(f"体素 {occ.sum()} 个（去掉孤立体素 {isolated.sum()} 个）")
    good = occ[tuple(ijk.T)]
    xyz = xyz[good]

    write_pcd(map_dir / "dense_map.pcd", [("x", "F"), ("y", "F"), ("z", "F")],
              [xyz[:, i].astype(np.float32) for i in range(3)])
    # 地面：有地面点的栅格，闭运算补缝并填洞，去掉零碎小块
    zc = lo[2] + (np.arange(occ.shape[2]) + 0.5) * VOXEL
    floor = occ[:, :, zc < FLOOR_TOP].any(2)
    floor = ndimage.binary_fill_holes(ndimage.binary_closing(floor, iterations=4) | floor)
    lab, n = ndimage.label(floor)
    sizes = ndimage.sum(floor, lab, range(1, n + 1))
    floor = np.isin(lab, 1 + np.flatnonzero(sizes * VOXEL ** 2 >= 0.5))
    obstacle = occ.copy()
    obstacle[:, :, zc < FLOOR_TOP] = False
    print(f"地面面积 {floor.sum() * VOXEL ** 2:.1f} m²，障碍体素 {obstacle.sum()} 个")
    nface = voxel_mesh_obj(map_dir / "mesh" / "scene.obj", obstacle, lo, floor)
    colormap_texture(map_dir / "mesh" / "scene_texture.png")
    print(f"dense_map.pcd {len(xyz)} 点；体素网格外露面 {nface} 个")

    # 5) 点位：沿轨迹按 1.5 m 间隔取点
    wps, last = [], None
    for p, h in zip(traj, heading):
        if last is not None and np.linalg.norm(p[:2] - last) < 1.5:
            continue
        last = p[:2]
        n = len(wps) + 1
        wps.append({"id": f"wp_{n:02d}", "name": f"room {n}",
                    "stand_pose": {"x": round(float(p[0]), 3), "y": round(float(p[1]), 3), "z": 0.0,
                                   "yaw": round(float(h), 3)},
                    "lookat": {"x": round(float(p[0] + np.cos(h)), 3),
                               "y": round(float(p[1] + np.sin(h)), 3), "z": 0.5},
                    "arrive_tolerance": {"xy": 0.15, "yaw": 0.2},
                    "safety_distance": 0.3})
    (map_dir / "waypoints.yaml").write_text(yaml.safe_dump({"waypoints": wps}, sort_keys=False))
    meta = {"map_id": "go2_room_onboard", "version": map_dir.name, "up_axis": "z", "units": "meter",
            "source": "Unitree Go2 onboard uslam /uslam/cloud_map (bag room-19700102-135516-vpn1)",
            "gravity_correction_deg": float(round(tilt, 2)),
            "hfield_margin": 20}
    (map_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    np.savetxt(map_dir / "trajectory.txt", np.column_stack([traj, heading]), fmt="%.4f",
               header="机身轨迹（地图坐标系）: x y z yaw")
    print(f"点位 {len(wps)} 个；地图范围 x[{xyz[:, 0].min():.1f},{xyz[:, 0].max():.1f}] "
          f"y[{xyz[:, 1].min():.1f},{xyz[:, 1].max():.1f}] z[{xyz[:, 2].min():.1f},{xyz[:, 2].max():.1f}]")


if __name__ == "__main__":
    main(*sys.argv[1:4])
