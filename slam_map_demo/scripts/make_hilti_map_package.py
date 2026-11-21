"""把 HILTI exp14 的 FAST-LIVO2 建图产物转成 slam_map_demo 地图包。

- 重力校正：RANSAC 拟合主地面平面，把法向旋转到 +z（实测倾斜 3.26°）
- 地面识别：只统计局部法向竖直的点，并检查地图是否上下颠倒（颠倒则报错）
- 原点选择：平坦地面点，且 +x 方向有 >=1.5 m 的连续可走地面
  （仿真器把机器狗以伸直腿姿态放在世界原点、朝 +x）
- 输出 dense_map.pcd（xyz，用于高度场）+ mesh_input.pcd（XYZRGB 下采样，用于网格重建）
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree


def read_pcd(path):
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if line.startswith("#") or not line:
                continue
            key, _, value = line.partition(" ")
            header[key] = value
            if key == "DATA":
                break
        fields = header["FIELDS"].split()
        n = int(header["POINTS"])
        buf = f.read()
    sizes = [int(s) for s in header["SIZE"].split()]
    stride = sum(sizes)
    arr = np.frombuffer(buf[: n * stride], dtype=np.float32).reshape(n, stride // 4)
    out = {}
    for i, name in enumerate(fields):
        out[name] = arr[:, i]
    return out


def write_pcd(path, fields, arrays):
    """fields: [(name, type)]，type 为 f 时按 float32 写入；rgb 为打包 uint32 的浮点位模式。"""
    n = len(arrays[0])
    header = "".join([
        "# .PCD 0.7 - Point Cloud Data file format\n",
        "VERSION 0.7\n",
        "FIELDS " + " ".join(f for f, _ in fields) + "\n",
        "SIZE " + " ".join("4" for _ in fields) + "\n",
        "TYPE " + " ".join(t for _, t in fields) + "\n",
        "COUNT " + " ".join("1" for _ in fields) + "\n",
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n",
        f"POINTS {n}\nDATA binary\n",
    ])
    data = np.column_stack(arrays).astype(np.float32).tobytes()
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data)


def floor_level(xyz, bin_size=0.05, n_sample=300000, k=20):
    """主地面高度：水平面点（局部法向接近竖直）最多的高度层。

    不能直接取全部点的最密高度层：雷达 0° 俯仰的那圈激光打在墙上，点都落在传感器高度，
    叠加后比地面还密。这里先用局部 PCA 估计法向，只统计水平面上的点。

    同时检查地图是否上下颠倒：地面应在场景底部，若有大量点（墙、家具）低于它，
    说明找到的是天花板，地图 z 轴朝下（例如建图时没打开重力对齐、IMU 又是倒装的）。
    """
    rng = np.random.default_rng(0)
    sub = xyz[rng.choice(len(xyz), min(n_sample, len(xyz)), replace=False)]
    _, nn = cKDTree(sub).query(sub, k=k)
    nb = sub[nn] - sub[nn].mean(1, keepdims=True)
    _, vecs = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb))
    horizontal = np.abs(vecs[:, 2, 0]) > 0.95  # 最小特征值对应的特征向量即法向
    z = xyz[:, 2]
    zh = sub[horizontal, 2]
    lo, hi = np.percentile(zh, [0.5, 99.5])
    h, e = np.histogram(zh, bins=max(int((hi - lo) / bin_size), 1), range=(lo, hi))
    level = e[np.argmax(h)] + bin_size / 2
    below = np.mean(z < level - 0.3)
    above = np.mean(z > level + 0.3)
    print(f"[floor] 最密水平层 z={level:.2f} m，其下方点占 {below:.1%}，上方点占 {above:.1%}")
    if below > above:
        raise ValueError("找到的主水平面上方点少于下方点，像是天花板：地图可能上下颠倒，"
                         "请检查建图是否已做重力对齐（FAST-LIVO2 的 uav/gravity_align_en）")
    return level


def ransac_floor(xyz, n_iter=500, thr=0.05):
    """拟合主地面平面（迭代重加权），返回法向（朝 +z）与平面偏移。"""
    level = floor_level(xyz)
    band = xyz[(xyz[:, 2] > level - 0.25) & (xyz[:, 2] < level + 0.25)]
    rng = np.random.default_rng(0)
    sub = band[rng.choice(len(band), 50000, replace=False)]
    best = None
    for _ in range(n_iter):
        p1, p2, p3 = sub[rng.choice(len(sub), 3, replace=False)]
        nrm = np.cross(p2 - p1, p3 - p1)
        nn = np.linalg.norm(nrm)
        if nn < 1e-9:
            continue
        nrm /= nn
        if abs(nrm[2]) < 0.9:  # 只接受近水平面
            continue
        d = -nrm @ p1
        inl = (np.abs(sub @ nrm + d) < thr).sum()
        if inl > (best[0] if best else 0):
            best = (inl, nrm, d)
    nrm, d = best[1], best[2]
    if nrm[2] < 0:
        nrm = -nrm
    # IRLS 精修：权重随残差衰减
    for _ in range(5):
        r = band @ nrm + d
        w = 1.0 / (0.05 + np.abs(r))
        A = np.column_stack([band[:, :2], np.ones(len(band))])
        c, _, _, _ = np.linalg.lstsq(A * w[:, None], band[:, 2] * w, rcond=None)
        nrm = np.array([-c[0], -c[1], 1.0])
        nrm /= np.linalg.norm(nrm)
        d = c[2] / np.linalg.norm([-c[0], -c[1], 1.0])
    if nrm[2] < 0:
        nrm, d = -nrm, -d
    return nrm, d


def rotation_to_up(nrm):
    """返回把法向 nrm 转到 +z 的旋转矩阵（罗德里格斯公式，K 由单位旋转轴构造）。"""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(nrm, z)
    s, c = np.linalg.norm(axis), nrm @ z
    if s < 1e-9:
        return np.eye(3)
    axis /= s
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def rotate_to_up(xyz, nrm):
    """绕水平轴旋转点云，使法向对齐 +z。"""
    return xyz @ rotation_to_up(nrm).T


class FloorIndex:
    """地面点与低矮障碍点的二维 KD 树，用于判断某处是否平坦、周围是否无障碍。"""

    def __init__(self, xyz):
        self.floor = xyz[np.abs(xyz[:, 2]) < 0.15]  # 校正后地面带
        self.obst = xyz[(xyz[:, 2] > 0.15) & (xyz[:, 2] < 1.0)]  # 机身高度范围内的障碍
        self.floor_tree = cKDTree(self.floor[:, :2])
        self.obst_tree = cKDTree(self.obst[:, :2])

    def patch(self, p, radius):
        return self.floor[self.floor_tree.query_ball_point(p[:2], radius)]

    def clear(self, p, radius):
        return not self.obst_tree.query_ball_point(p[:2], radius, return_length=True)


def choose_origin(index, radius=0.5, thr=0.04, clearance=0.5):
    """选平坦地面点：取平坦邻域最大、且周围无障碍的候选（地下室地面分段不平，不强求 +x 连续可走）。"""
    floor = index.floor
    rng = np.random.default_rng(1)
    cand = floor[rng.choice(len(floor), 8000, replace=False)]
    best = None
    n_flat = 0
    for p in cand:
        patch = index.patch(p, radius)
        if len(patch) < 500 or patch[:, 2].std() > thr or not index.clear(p, clearance):
            continue
        n_flat += 1
        # 平坦邻域半径：逐级扩大尝试，越大越好
        r_ok = radius
        for r2 in (0.8, 1.0):
            p2 = index.patch(p, r2)
            if len(p2) < 800 or p2[:, 2].std() > thr:
                break
            r_ok = r2
        score = len(patch)
        if best is None or score > best[0]:
            best = (score, p, r_ok)
    print(f"[origin] 候选 {len(cand)}，平坦且无障碍 {n_flat}")
    if best is None:
        return None
    print(f"[origin] 选中 ({best[1][0]:.1f},{best[1][1]:.1f})，平坦邻域半径 {best[2]:.1f} m，点数 {best[0]}")
    return best[1]


def main(src_pcd, map_dir, poses_txt=None):
    map_dir = Path(map_dir)
    map_dir.mkdir(parents=True, exist_ok=True)

    d = read_pcd(src_pcd)
    xyz = np.column_stack([d["x"], d["y"], d["z"]]).astype(np.float64)
    rgb = np.column_stack([d["rgb"]]).astype(np.float32)
    # 传感器轨迹（TUM 格式），与地图做相同变换后写入地图包，供网格重建统一法向朝向
    traj = np.loadtxt(poses_txt)[:, 1:4] if poses_txt else np.zeros((0, 3))

    # 1) 重力校正
    nrm, _ = ransac_floor(xyz)
    tilt = np.degrees(np.arccos(abs(nrm[2])))
    print(f"地面倾角 {tilt:.2f}°，旋转校正")
    xyz = rotate_to_up(xyz, nrm)
    traj = rotate_to_up(traj, nrm)

    # 2) 地面归零
    level = floor_level(xyz)
    floor_band = xyz[np.abs(xyz[:, 2] - level) < 0.1]
    z0 = np.median(floor_band[:, 2])
    xyz[:, 2] -= z0
    traj[:, 2] -= z0
    print(f"地面中位高度 {z0:.3f} m → 平移归零")

    # 3) 原点选择
    origin = choose_origin(FloorIndex(xyz))
    xyz[:, :2] -= origin[:2]
    traj[:, :2] -= origin[:2]
    if len(traj):
        np.savetxt(map_dir / "trajectory.txt", traj, fmt="%.4f", header="建图传感器轨迹（地图坐标系）: x y z")
    print(f"原点选在 ({origin[0]:.2f}, {origin[1]:.2f})（校正前坐标），已平移")

    # 4) dense_map.pcd（全量 xyz，供高度场）
    write_pcd(map_dir / "dense_map.pcd",
              [("x", "F"), ("y", "F"), ("z", "F")],
              [xyz[:, 0].astype(np.float32), xyz[:, 1].astype(np.float32), xyz[:, 2].astype(np.float32)])
    print(f"dense_map.pcd: {len(xyz)} 点")

    # 5) mesh_input.pcd（0.03 m 体素下采样 XYZRGB，供网格重建）
    v = 0.03
    key = np.floor(xyz[:, :3] / v).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    ms = np.sort(idx)
    print(f"mesh 输入下采样: {len(xyz)} -> {len(ms)} 点")
    write_pcd(map_dir / "mesh_input.pcd",
              [("x", "F"), ("y", "F"), ("z", "F"), ("rgb", "F")],
              [xyz[ms, 0].astype(np.float32), xyz[ms, 1].astype(np.float32),
               xyz[ms, 2].astype(np.float32), rgb[ms, 0]])

    # 6) meta.yaml + waypoints.yaml（自动挑 8 个平坦点位作目视标记）
    meta = {"map_id": "hilti_exp14", "version": map_dir.name,
            "up_axis": "z", "units": "meter",
            "source": "HILTI 2022 exp14_basement_2, FAST-LIVO2 (official ROS1, gravity aligned)",
            "gravity_correction_deg": float(round(tilt, 2))}
    (map_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))

    # 点位要求：0.6 m 内地面平整，0.45 m 内（机身半长）没有障碍
    index = FloorIndex(xyz)
    rng = np.random.default_rng(2)
    cand = index.floor[rng.choice(len(index.floor), 800, replace=False)]
    wps, seen = [], set()
    for p in cand:
        patch = index.patch(p, 0.6)
        if len(patch) < 400 or patch[:, 2].std() > 0.05 or not index.clear(p, 0.45):
            continue
        key = tuple(np.round(p[:2] / 1.5).astype(int))
        if key in seen:
            continue
        seen.add(key)
        wps.append({"id": f"wp_{len(wps)+1:02d}",
                    "name": f"flat spot {len(wps)+1}",
                    "stand_pose": {"x": float(p[0]), "y": float(p[1]), "z": 0.0,
                                   "yaw": float(np.arctan2(-p[1], -p[0]))},
                    "lookat": {"x": float(p[0]), "y": float(p[1]), "z": 1.0},
                    "arrive_tolerance": {"xy": 0.15, "yaw": 0.2},
                    "safety_distance": 0.3})
        if len(wps) >= 8:
            break
    (map_dir / "waypoints.yaml").write_text(yaml.safe_dump({"waypoints": wps}, sort_keys=False))
    print(f"waypoints: {len(wps)} 个")
    print(f"地图范围: x[{xyz[:,0].min():.1f},{xyz[:,0].max():.1f}] "
          f"y[{xyz[:,1].min():.1f},{xyz[:,1].max():.1f}] z[{xyz[:,2].min():.1f},{xyz[:,2].max():.1f}]")


if __name__ == "__main__":
    main(*sys.argv[1:4])
