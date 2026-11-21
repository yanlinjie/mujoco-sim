"""把 FAST-LIVO2 建图产物转成 slam_map_demo 地图包（通用版）。

- 重力校正：RANSAC 拟合主地面平面，把法向旋转到 +z（实测倾斜 3.26°）
- 原点选择：平坦地面点，且 +x 方向有 >=1.5 m 的连续可走地面
  （仿真器把机器狗以伸直腿姿态放在世界原点、朝 +x）
- 输出 dense_map.pcd（xyz，用于高度场）+ mesh_input.pcd（XYZRGB 下采样，用于网格重建）
"""
import sys
from pathlib import Path

import numpy as np
import yaml


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


def ransac_floor(xyz, n_iter=500, thr=0.05, gravity=None):
    """拟合主地面平面（迭代重加权），返回法向（朝 +z）与平面偏移。

    gravity: 重力方向（指向地面）在地图坐标系下的向量；不传则默认 -z。
    """
    if gravity is None:
        gravity = np.array([0.0, 0.0, -1.0])
    g = np.asarray(gravity, dtype=np.float64)
    g /= np.linalg.norm(g)
    proj = xyz @ g
    lo_q, hi_q = np.quantile(proj, [0.02, 0.25])
    band = xyz[(proj > lo_q) & (proj < hi_q)]
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
        if abs(nrm @ g) < 0.9:  # 只接受与重力方向接近的平面
            continue
        d = -nrm @ p1
        inl = (np.abs(sub @ nrm + d) < thr).sum()
        if inl > (best[0] if best else 0):
            best = (inl, nrm, d)
    nrm, d = best[1], best[2]
    if nrm[2] < 0:
        nrm = -nrm
    # IRLS 精修：以重力方向为法向先验做带权平面拟合
    for _ in range(5):
        r = band @ nrm + d
        w = 1.0 / (0.05 + np.abs(r))
        c, _, _, _ = np.linalg.lstsq(band * w[:, None], (band @ g)[:, None] * w[:, None], rcond=None)
        nrm = g + c[:, 0]
        nrm /= np.linalg.norm(nrm)
        d = -nrm @ (band * w[:, None]).sum(0) / w.sum()
    if nrm @ g < 0:
        nrm, d = -nrm, -d
    return nrm, d


def rotate_to_up(xyz, nrm):
    """绕水平轴旋转点云，使法向对齐 +z。"""
    z = np.array([0.0, 0.0, 1.0])
    c = nrm @ z
    if c > 0.99999:
        return xyz
    axis = np.cross(nrm, z)
    s = np.linalg.norm(axis)
    axis /= s
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    # 罗德里格斯公式（K 由单位旋转轴构造）：R = I + sinθ·K + (1-cosθ)·K²
    R = np.eye(3) + s * K + (1 - c) * K @ K
    return xyz @ R.T


def choose_origin(xyz, radius=0.5, thr=0.04):
    """选平坦地面点：取平坦邻域最大的候选（地下室地面分段不平，不强求 +x 连续可走）。"""
    floor = xyz[np.abs(xyz[:, 2]) < 0.15]  # 校正后地面带
    rng = np.random.default_rng(1)
    cand = floor[rng.choice(len(floor), 8000, replace=False)]
    best = None
    n_flat = 0
    for p in cand:
        d = np.linalg.norm(floor[:, :2] - p[:2], axis=1)
        patch = floor[d < radius]
        if len(patch) < 500:
            continue
        if patch[:, 2].std() > thr:
            continue
        n_flat += 1
        # 平坦邻域半径：按 0.05 m 递增尝试，越大越好
        for r2 in (0.8, 1.0):
            dd = np.linalg.norm(floor[:, :2] - p[:2], axis=1)
            p2 = floor[dd < r2]
            if len(p2) < 800 or p2[:, 2].std() > thr:
                break
            radius = r2
        score = len(patch)
        if best is None or score > best[0]:
            best = (score, p, radius)
    print(f"[origin] 候选 {len(cand)}，平坦 {n_flat}")
    if best is None:
        return None
    print(f"[origin] 选中 ({best[1][0]:.1f},{best[1][1]:.1f})，平坦邻域半径 {best[2]:.1f} m，点数 {best[0]}")
    return best[1]


def main(src_pcd, map_dir, gravity=None):
    map_dir = Path(map_dir)
    map_dir.mkdir(parents=True, exist_ok=True)

    d = read_pcd(src_pcd)
    xyz = np.column_stack([d["x"], d["y"], d["z"]]).astype(np.float64)
    rgb = np.column_stack([d["rgb"]]).astype(np.float32)

    # 1) 重力校正
    nrm, _ = ransac_floor(xyz, gravity=gravity)
    g = np.array([0.0, 0.0, -1.0]) if gravity is None else np.asarray(gravity, dtype=np.float64)
    g = g / np.linalg.norm(g)
    tilt = np.degrees(np.arccos(abs(nrm[2])))
    print(f"地面法向与 z 轴夹角 {tilt:.2f}°，旋转校正")
    xyz = rotate_to_up(xyz, nrm)

    # 2) 地面归零（沿校正后 z 轴的最低带）
    zq = np.quantile(xyz[:, 2], [0.02, 0.25])
    floor_band = xyz[(xyz[:, 2] > zq[0]) & (xyz[:, 2] < zq[1])]
    z0 = np.median(floor_band[:, 2])
    xyz[:, 2] -= z0
    print(f"地面中位高度 {z0:.3f} m → 平移归零")

    # 3) 原点选择：优先平坦邻域；找不到时用地面带中离世界原点最近的点（机器人从原点出发，那里必是地面）
    origin = choose_origin(xyz)
    if origin is None:
        dist = np.linalg.norm(floor_band[:, :2], axis=1)
        origin = floor_band[np.argmin(dist)]
        print("[origin] 平坦搜索失败，改用离原点最近的地面点")
    xyz[:, :2] -= origin[:2]
    print(f"原点选在 ({origin[0]:.2f}, {origin[1]:.2f})（校正前坐标），已平移")

    # 4) 把较长的水平轴转到 +x（机器狗出生朝 +x，前进方向需要空间）
    ext_x = xyz[:, 0].max() - xyz[:, 0].min()
    ext_y = xyz[:, 1].max() - xyz[:, 1].min()
    if ext_y > ext_x:
        old = xyz[:, :2].copy()
        xyz[:, 0] = old[:, 1]
        xyz[:, 1] = -old[:, 0]
        print(f"水平轴交换：y({ext_y:.1f} m) -> x（原 x({ext_x:.1f} m) -> -y）")

    # 3.5) 按 0.1%-99.9% 分位数裁剪离群点（稀疏扫描的尾点会把包围盒撑得很大）
    def clip_axis(a):
        lo, hi = np.percentile(a, [0.1, 99.9])
        return (a >= lo) & (a <= hi)
    keep = clip_axis(xyz[:, 0]) & clip_axis(xyz[:, 1]) & clip_axis(xyz[:, 2])
    print(f"离群点裁剪: {len(xyz) - keep.sum()} 点被移除")
    xyz = xyz[keep]
    rgb = rgb[keep]

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
    meta = {"map_id": map_dir.parent.name, "version": map_dir.name,
            "up_axis": "z", "units": "meter",
            "source": src_pcd,
            "gravity_correction_deg": float(round(tilt, 2)),
            "hfield_margin": 40}  # 高度场四周 2m 零高度边距，防止机器狗脚踩出场外
    (map_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))

    floor = xyz[np.abs(xyz[:, 2]) < 0.15]
    rng = np.random.default_rng(2)
    cand = floor[rng.choice(len(floor), 800, replace=False)]
    wps, seen = [], set()
    for p in cand:
        dd = np.linalg.norm(floor[:, :2] - p[:2], axis=1)
        patch = floor[dd < 0.6]
        if len(patch) < 400 or patch[:, 2].std() > 0.05:
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
    gravity = None
    if len(sys.argv) > 3:
        gravity = [float(v) for v in sys.argv[3].split(",")]
    main(sys.argv[1], sys.argv[2], gravity)
