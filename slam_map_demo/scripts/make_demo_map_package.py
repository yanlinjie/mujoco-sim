"""用开源扫描 Mesh 伪造一份“SLAM 地图包”，代替真实建图结果做快速验证。

输出目录结构与 SLAM部署方案.md 第七步一致：
    maps/<地图ID>/<版本号>/
    ├── meta.yaml
    ├── dense_map.pcd        # 从 Mesh 表面采样得到，模拟 SLAM 输出的稠密点云
    ├── mesh/scene.obj       # 视觉 Mesh
    └── mesh/texture.png

点位库 waypoints.yaml 由 annotate_waypoints.py 另行标注生成。

用法：
    python -I make_demo_map_package.py <apartment_1.glb> <输出地图包目录>
"""
import sys
from pathlib import Path

import numpy as np
import trimesh
import yaml

MAP_ID = "habitat_apartment_1"
MAP_VERSION = "2026-10-27.1"

# 在原始 Mesh 坐标系中选定的地图原点（餐厅北侧通道的硬地面，模拟充电桩位置）。
# 机器人在 unitree_mujoco 中总是以伸直腿的姿态出生在世界原点，
# 因此原点必须是一块没有地毯等凸起的空地，否则脚会一出生就嵌进碰撞体。
# 原点朝 +x 方向需留出行走空间：此处正前方约 2.8 m 内无障碍，且地面有真实扫描数据。
ORIGIN_IN_SOURCE = np.array([3.8, -3.0])

NUM_CLOUD_POINTS = 3_000_000


def estimate_floor_height(mesh):
    """取朝上面片中面积最大的高度层作为地面高度。"""
    normals = mesh.face_normals
    areas = mesh.area_faces
    up = normals[:, 2] > 0.9
    heights = mesh.triangles_center[up, 2]
    hist, edges = np.histogram(heights, bins=200, weights=areas[up])
    i = hist.argmax()
    return (edges[i] + edges[i + 1]) / 2


def fit_floor_plane(mesh, floor):
    """用地面附近朝上的面片拟合平面 z = a x + b y + c，迭代剔除离群面片（家具顶面、地毯边缘）。"""
    normals, centers, areas = mesh.face_normals, mesh.triangles_center, mesh.area_faces
    sel = (normals[:, 2] > 0.95) & (np.abs(centers[:, 2] - floor) < 0.15)
    pts, w = centers[sel], areas[sel]
    for _ in range(5):
        design = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
        coef = np.linalg.lstsq(design * np.sqrt(w)[:, None], pts[:, 2] * np.sqrt(w), rcond=None)[0]
        res = pts[:, 2] - design @ coef
        keep = np.abs(res) < max(0.01, 2.5 * np.sqrt(np.average(res ** 2, weights=w)))
        pts, w = pts[keep], w[keep]
    return coef


def rotation_to_z(normal):
    """把单位向量 normal 转到 +z 的旋转矩阵（Rodrigues 公式）。"""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, z)
    s, c = np.linalg.norm(axis), normal @ z
    if s < 1e-12:
        return np.eye(3)
    k = axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * kx + (1 - c) * kx @ kx


def write_obj(path, vertices, uvs, faces):
    with open(path, "w") as f:
        f.write("# 视觉 Mesh，map 坐标系，单位米，z 轴朝上\n")
        np.savetxt(f, vertices, fmt="v %.5f %.5f %.5f")
        np.savetxt(f, uvs, fmt="vt %.6f %.6f")
        idx = faces + 1
        np.savetxt(f, np.repeat(idx, 2, axis=1), fmt="f %d/%d %d/%d %d/%d")


def write_pcd(path, points):
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\nDATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(points.astype("<f4").tobytes())


def main(glb_path, out_dir):
    out_dir = Path(out_dir)
    (out_dir / "mesh").mkdir(parents=True, exist_ok=True)

    # trimesh 会应用 glb 节点变换，变换后该场景为 z 轴朝上（已通过地面/天花板面积确认）
    mesh = trimesh.load(glb_path).to_geometry()
    # 重力对齐：扫描场景的地面与水平面有约 1° 倾斜，跨 10 m 就差出近 10 cm。
    # 拟合地面平面，绕原点把场景转平，再把原点处的地面平移到 z=0。
    a, b, c = fit_floor_plane(mesh, estimate_floor_height(mesh))
    normal = np.array([-a, -b, 1.0]) / np.linalg.norm([-a, -b, 1.0])
    rot = rotation_to_z(normal)
    origin = np.array([ORIGIN_IN_SOURCE[0], ORIGIN_IN_SOURCE[1], a * ORIGIN_IN_SOURCE[0] + b * ORIGIN_IN_SOURCE[1] + c])
    print(f"地面倾斜 {np.degrees(np.arccos(normal[2])):.2f}°，原点处地面高度 {origin[2]:.3f} m")

    vertices = (mesh.vertices - origin) @ rot.T
    write_obj(out_dir / "mesh" / "scene.obj", vertices, mesh.visual.uv, mesh.faces)
    mesh.visual.material.baseColorTexture.save(out_dir / "mesh" / "texture.png")

    shifted = trimesh.Trimesh(vertices, mesh.faces, process=False)
    points, _ = trimesh.sample.sample_surface(shifted, NUM_CLOUD_POINTS, seed=0)
    write_pcd(out_dir / "dense_map.pcd", points)

    meta = {
        "map_id": MAP_ID,
        "version": MAP_VERSION,
        "frame_id": "map",
        "units": "meter",
        "up_axis": "z",
        "gravity_aligned": True,
        "origin_description": "演示用：餐厅北侧通道空地，模拟充电桩；地面高度为 z=0",
        "source": "Habitat test scenes / apartment_1.glb（开源扫描 Mesh，代替 SLAM 输出）",
    }
    with open(out_dir / "meta.yaml", "w") as f:
        yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)

    print(f"地图包已写入 {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
