"""把单独一份重建网格（相机坐标系：y 朝下，米）转成地图包，供 map_to_mujoco.py 使用。

适用于只有网格、没有轨迹的重建结果，两种输入：
- Open3D 写出的带顶点颜色 PLY（如 robot_downloads/path2/mesh_colored.ply）：顶点颜色烘焙成贴图；
- nerfstudio 导出的带贴图 OBJ（如 robot_downloads/path1/mesh.obj + material_0.png）：沿用原 UV 和贴图。
碰撞高度场总是在网格表面按 1 cm 间距采样生成。可另给一份点云（如 path1/point_cloud.ply），只用来确定场景主体范围
（各轴 1%~99% 分位再外扩 CROP_PAD）：泊松重建的粗网格会在场景外围生成包裹的“气泡”面，范围外的三角形一律删掉。
点云本身不用于碰撞：NeRF 点云里满是悬空浮点，做成高度场会把整个房间填满假障碍。

处理：
1. 粗对齐：相机坐标系 y 朝下 → 绕 x 轴转 90°，使 -y 变成 +z；
2. 重力微调：RANSAC 拟合主地面，转平并把地面归零；
3. 原点：选离障碍和未重建区域最远的地面栅格（仿真器总把机器狗放在世界原点、朝 +x）；
4. dense_map.pcd：网格表面按 1 cm 间距均匀采样，供生成碰撞高度场；
5. 视觉网格（PLY 输入时）：MuJoCo 不支持顶点颜色，把每个三角形三个顶点颜色的平均值烘焙到贴图的一个像素上，
   该三角形三个顶点的 UV 都指向这个像素中心（逐面纯色）；
6. 点位只放原点一个（充电桩）。

用法: make_mesh_map_package.py <网格 .ply/.obj> <地图包目录> <map_id> [来源说明] [碰撞用点云 .ply]
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_hilti_map_package import ransac_floor, rotation_to_up, write_pcd  # noqa: E402
from map_to_mujoco import RESOLUTION, build_heightfield  # noqa: E402
from nav_grid import NavGrid  # noqa: E402

CAM_TO_ZUP = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])  # (x, y, z) → (x, z, -y)
CROP_PAD = 1.0  # 米


def robust_box(xyz):
    lo, hi = np.percentile(xyz, [1, 99], axis=0)
    return lo - CROP_PAD, hi + CROP_PAD


def in_box(xyz, box):
    return np.all((xyz >= box[0]) & (xyz <= box[1]), axis=1)


def transform_textured_obj(src, dst, T, box_raw):
    """变换带贴图 OBJ 的顶点（其余行原样保留），删掉任一顶点落在裁剪框外的面。"""
    lines = open(src).read().splitlines()
    v = np.array([l.split()[1:4] for l in lines if l.startswith("v ")], float)
    keep = in_box(v @ CAM_TO_ZUP.T, box_raw)
    vm = iter(v @ T[:3, :3].T + T[:3, 3])
    out, dropped = [], 0
    for l in lines:
        if l.startswith("v "):
            x, y, z = next(vm)
            out.append(f"v {x:.4f} {y:.4f} {z:.4f}")
        elif l.startswith("f "):
            if keep[[int(t.split("/")[0]) - 1 for t in l.split()[1:]]].all():
                out.append(l)
            else:
                dropped += 1
        elif not l.startswith(("mtllib", "usemtl")):
            out.append(l)
    Path(dst).write_text("\n".join(out) + "\n")
    return dropped


def bake_face_colors(tris, colors, out_dir):
    """逐面颜色烘焙成贴图，返回每个三角形的 UV（像素中心）。"""
    n = len(tris)
    side = int(np.ceil(np.sqrt(n)))
    rgb = (colors[tris].mean(1) * 255).round().astype(np.uint8)
    img = np.zeros((side * side, 3), np.uint8)
    img[:n] = rgb
    Image.fromarray(img.reshape(side, side, 3)).save(out_dir / "scene_texture.png")
    k = np.arange(n)
    u = (k % side + 0.5) / side
    v = 1.0 - (k // side + 0.5) / side  # OBJ 的 v 从图片底部算起
    return np.column_stack([u, v])


def write_obj(path, verts, tris, uv):
    with open(path, "w") as f:
        f.write("# make_mesh_map_package.py：逐面颜色烘焙到 scene_texture.png\n")
        f.write("".join(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in verts))
        f.write("".join(f"vt {a:.7f} {b:.7f}\n" for a, b in uv))
        f.write("".join(f"f {a + 1}/{i + 1} {b + 1}/{i + 1} {c + 1}/{i + 1}\n"
                        for i, (a, b, c) in enumerate(tris)))


def main(ply, map_dir, map_id, source="reconstructed mesh", points=None):
    ply, map_dir = Path(ply), Path(map_dir)
    (map_dir / "mesh").mkdir(parents=True, exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(str(ply))
    verts = np.asarray(mesh.vertices) @ CAM_TO_ZUP.T
    tris = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    if points:
        box_raw = robust_box(np.asarray(o3d.io.read_point_cloud(str(points)).points) @ CAM_TO_ZUP.T)
        mesh.remove_triangles_by_mask(~in_box(verts, box_raw)[tris].all(1))
        print(f"按点云主体范围裁剪网格：保留 {len(mesh.triangles)}/{len(tris)} 个三角形")
    else:
        box_raw = (np.full(3, -np.inf), np.full(3, np.inf))
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) @ CAM_TO_ZUP.T)
    area = mesh.get_surface_area()
    xyz = np.asarray(mesh.sample_points_uniformly(int(area / 0.01 ** 2)).points)
    print(f"网格 {len(verts)} 顶点 / {len(tris)} 三角形，碰撞采样表面积 {area:.1f} m²，采样 {len(xyz)} 点")

    nrm, d = ransac_floor(xyz)
    Rg = rotation_to_up(nrm)
    xyz = xyz @ Rg.T
    z0 = np.median(xyz[np.abs(xyz[:, 2] - d) < 0.05, 2])
    tilt = np.degrees(np.arccos(abs(nrm[2])))
    xyz[:, 2] -= z0
    print(f"重力微调 {tilt:.2f}°，地面高度 {z0:.3f} m 归零")

    heights, lo, observed = build_heightfield(xyz)
    grid = NavGrid(heights, lo, RESOLUTION, observed)
    r, c = np.unravel_index(np.argmax(grid.clearance), grid.clearance.shape)
    o = grid.to_xy((r, c))
    print(f"原点选在离障碍最远处 ({o[0]:+.2f}, {o[1]:+.2f})，离障碍/未重建区域 {grid.clearance[r, c]:.2f} m")
    xyz[:, :2] -= o

    T = np.eye(4)
    T[:3, :3] = Rg @ CAM_TO_ZUP
    T[:3, 3] = [-o[0], -o[1], -z0]
    np.savetxt(map_dir / "map_from_slam.txt", T, fmt="%.9f", header="4x4 齐次变换：地图坐标 = T @ 原始网格坐标")
    write_pcd(map_dir / "dense_map.pcd", [("x", "F"), ("y", "F"), ("z", "F")],
              [xyz[:, i].astype(np.float32) for i in range(3)])

    if ply.suffix == ".obj":
        n_drop = transform_textured_obj(ply, map_dir / "mesh" / "scene.obj", T, box_raw)
        tex = [l.split(None, 1)[1].strip() for l in open(ply.with_suffix(".mtl") if ply.with_suffix(".mtl").exists()
               else next(ply.parent.glob("*.mtl"))) if l.startswith("map_Kd")][0]
        shutil.copy(ply.parent / tex, map_dir / "mesh" / "scene_texture.png")
        print(f"视觉网格沿用原贴图 {tex}，删掉主体范围外的三角形 {n_drop} 个")
    else:
        v_map = np.asarray(o3d.io.read_triangle_mesh(str(ply)).vertices) @ T[:3, :3].T + T[:3, 3]
        uv = bake_face_colors(tris, colors, map_dir / "mesh")
        write_obj(map_dir / "mesh" / "scene.obj", v_map, tris, uv)

    wps = [{"id": "wp_01", "name": "原点（充电桩）",
            "stand_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "lookat": {"x": 1.0, "y": 0.0, "z": 0.5},
            "arrive_tolerance": {"xy": 0.15, "yaw": 0.2}, "safety_distance": 0.3}]
    (map_dir / "waypoints.yaml").write_text(yaml.safe_dump({"waypoints": wps}, sort_keys=False, allow_unicode=True))
    meta = {"map_id": map_id, "version": map_dir.name, "up_axis": "z", "units": "meter", "source": source,
            "gravity_correction_deg": float(round(tilt, 2)), "hfield_margin": 40}
    (map_dir / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
    print(f"地图范围 x[{xyz[:, 0].min():.1f},{xyz[:, 0].max():.1f}] y[{xyz[:, 1].min():.1f},{xyz[:, 1].max():.1f}] "
          f"z[{xyz[:, 2].min():.1f},{xyz[:, 2].max():.1f}]")


if __name__ == "__main__":
    main(*sys.argv[1:6])
