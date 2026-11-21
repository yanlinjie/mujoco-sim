"""把 SLAM 地图包转换成 unitree_mujoco 可加载的场景文件。

- 视觉：直接引用 mesh/scene.obj，不参与碰撞；若有 mesh/scene_texture.png 则贴纹理。
  （MuJoCo 会把碰撞 Mesh 当作凸包，整块场景 Mesh 做碰撞会把房间“填实”。）
- 碰撞：由 dense_map.pcd 生成高度场。每个栅格取低于 CUT_HEIGHT 的最高点，
  去掉天花板后墙体和家具会变成凸起的柱子。地面和地毯再整理成严格平整的台面：
  MuJoCo 把高度场每个格子当作独立棱柱计算碰撞，相邻格子哪怕只差几毫米，
  足端也会碰到棱柱的边棱，产生明显倾斜的接触力，使机器人朝某些方向行走时打滑、倒退甚至侧翻。
- 地面：即使高度全为 0，C++ 仿真器自带的 MuJoCo 3.3.6 在足端踩到栅格边缘时仍会产生
  倾斜且已压入地面的接触，机器狗站立时会以约 1 cm/s 向身后滑动（Python 环境的 3.17 无此问题）。
  因此另加一个 z=0 的平面承担落脚，高度场整体下移 HFIELD_DROP，只负责墙体、家具和地毯。
  （地毯台面仍由高度场承担，站在地毯上时这一问题依旧存在。）
- 点位库：每个点位生成一个可视标记（不参与碰撞），用于目视核对坐标。

场景文件写到机器人模型目录下（与 terrain_tool 生成 scene_terrain.xml 的做法一致），
因为 go2.xml 的 meshdir 是相对主场景文件解析的；地图相关资源一律使用绝对路径。

用法：
    python -I map_to_mujoco.py <地图包目录> <unitree_robots/go2 目录>
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage

RESOLUTION = 0.05   # 高度场栅格边长，米
CUT_HEIGHT = 1.5    # 高于此高度的点视为天花板/吊灯，忽略
BASE_THICKNESS = 0.1
FLOOR_MAX_HEIGHT = 0.12  # 低于此高度视为可行走的地面或地毯，高于此视为障碍
RUG_MIN_HEIGHT = 0.03    # 地面中高于此值、且成片的区域视为地毯（地图须已重力对齐）
RUG_MIN_AREA = 1.0       # 平方米；更小的凸起当作地面噪声压平
HFIELD_DROP = 0.005     # 高度场整体下移量，米；地面改由 z=0 的平面承担，见开头说明
HEIGHT_PERCENTILE = None  # 栅格高度取法：None 为最高点；地图包 meta.yaml 可用 hfield_height_percentile 指定分位数


def read_pcd_xyz(path):
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("#"):
                continue
            key, _, value = line.partition(" ")
            header[key] = value
            if key == "DATA":
                break
        if header["DATA"] != "binary" or header["FIELDS"] != "x y z":
            raise ValueError("仅支持 FIELDS 为 x y z 的 binary PCD")
        n = int(header["POINTS"])
        return np.frombuffer(f.read(n * 12), dtype="<f4").reshape(n, 3)


def build_heightfield(points):
    """返回 (高度矩阵[行=y, 列=x], 网格最小角点 xy, 是否有扫描数据的掩码)。"""
    points = points[points[:, 2] < CUT_HEIGHT]
    lo = np.floor(points[:, :2].min(0) / RESOLUTION) * RESOLUTION
    ij = np.round((points[:, :2] - lo) / RESOLUTION).astype(int)
    ncol, nrow = ij.max(0) + 1
    flat = np.full(nrow * ncol, 0.0)
    lin = ij[:, 1] * ncol + ij[:, 0]
    if HEIGHT_PERCENTILE is None:
        np.maximum.at(flat, lin, points[:, 2])
    else:
        # 按分位数取栅格高度：噪声较大的雷达（如 L1，地面层厚约 20 cm）取最高点会把地面噪声上沿当成障碍
        o = np.lexsort((points[:, 2], lin))
        lin_s, z_s = lin[o], points[o, 2]
        start = np.searchsorted(lin_s, np.arange(nrow * ncol))
        cnt = np.bincount(lin_s, minlength=nrow * ncol)
        has = cnt > 0
        k = start[has] + np.floor((cnt[has] - 1) * HEIGHT_PERCENTILE / 100).astype(int)
        flat[has] = np.maximum(z_s[k], 0.0)
    counts = np.bincount(lin, minlength=nrow * ncol).reshape(nrow, ncol)
    # 填掉采样稀疏造成的零星空格，只保留成片的扫描缺失区
    observed = ndimage.binary_closing(counts > 0, iterations=2) | (counts > 0)
    # 地面噪声会略低于 0；没有点的栅格保持为 0（仿真中当作地面，导航时当作未知）
    heights = flatten_floor(np.clip(flat.reshape(nrow, ncol), 0.0, None))
    return heights, lo, observed


def flatten_floor(heights):
    """把地面压成 z=0，把每片地毯压成各自的固定高度，障碍保持不变。"""
    floor = heights < FLOOR_MAX_HEIGHT
    rugs = ndimage.binary_opening(floor & (heights >= RUG_MIN_HEIGHT), iterations=2)
    labels, n = ndimage.label(rugs)
    result = np.where(floor, 0.0, heights)
    for i in range(1, n + 1):
        region = labels == i
        if region.sum() * RESOLUTION ** 2 >= RUG_MIN_AREA:
            result[region] = np.median(heights[region])
    return result


def write_hfield_bin(path, heights):
    nrow, ncol = heights.shape
    with open(path, "wb") as f:
        f.write(np.array([nrow, ncol], dtype="<i4").tobytes())
        f.write(heights.astype("<f4").tobytes())


def local_ground(heights, lo, x, y, radius=0.15):
    """点位附近高度场的最高点，用于把标记抬到地毯等表面之上。"""
    c0, r0 = np.round((np.array([x, y]) - lo) / RESOLUTION).astype(int)
    k = int(np.ceil(radius / RESOLUTION))
    patch = heights[max(r0 - k, 0):r0 + k + 1, max(c0 - k, 0):c0 + k + 1]
    return float(patch.max()) if patch.size else 0.0


def waypoint_markers(waypoints, heights, lo):
    lines = []
    for wp in waypoints:
        p = wp["stand_pose"]
        yaw = p["yaw"]
        z = max(p["z"], local_ground(heights, lo, p["x"], p["y"]))
        quat = f"{np.cos(yaw / 2):.4f} 0 0 {np.sin(yaw / 2):.4f}"
        lines.append(
            f'    <body name="{wp["id"]}" pos="{p["x"]} {p["y"]} {z:.3f}" quat="{quat}">\n'
            f'      <site name="{wp["id"]}_disk" type="cylinder" size="0.12 0.005" pos="0 0 0.005" rgba="0.1 0.9 0.2 0.8"/>\n'
            f'      <site name="{wp["id"]}_heading" type="box" size="0.15 0.02 0.006" pos="0.15 0 0.012" rgba="0.9 0.1 0.1 0.9"/>\n'
            f"    </body>"
        )
    return "\n".join(lines)


SCENE_TEMPLATE = """<mujoco model="go2 scene on map {map_id}">
  <!-- 自动生成：slam_map_demo/scripts/map_to_mujoco.py，请勿手改 -->
  <include file="go2.xml"/>

  <statistic center="0 0 0.5" extent="4"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.4 0.4 0.4" specular="0 0 0"/>
    <global azimuth="-130" elevation="-30" offwidth="1280" offheight="960"/>
    <quality shadowsize="0"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
{material}
    <mesh name="map_mesh" file="{mesh}"/>
    <hfield name="map_hfield" file="{hfield}" size="{rx:.4f} {ry:.4f} {zmax:.4f} {base}"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <!-- 视觉：地图 Mesh，不参与碰撞 -->
    <geom name="map_visual" type="mesh" mesh="map_mesh" material="map_material"
          contype="0" conaffinity="0" group="1"/>
    <!-- 碰撞：由点云生成的高度场，放在第 3 组，查看器中默认隐藏，按数字键 3 显示。
         高度场下移 {hfield_drop} m，地面由 z=0 的平面承担，见脚本开头说明 -->
    <geom name="map_floor" type="plane" size="0 0 0.05" group="3" rgba="0.6 0.6 0.6 0.6"/>
    <geom name="map_collision" type="hfield" hfield="map_hfield" pos="{cx:.4f} {cy:.4f} {cz}"
          group="3" rgba="0.8 0.6 0.4 0.6"/>
    <!-- 点位库标记：绿色圆盘为站位，红色短条指向朝向 -->
{markers}
  </worldbody>
</mujoco>
"""


def map_material(texture_path):
    """地图包带纹理（mesh/scene_texture.png，如 3DGS 重建烘焙的纹理）时用纹理材质，否则用统一灰蓝色（与白色 Go2 拉开对比）。
    纹理已包含真实光照，材质不加高光，并给一定自发光，避免在仿真灯光下显得过暗。"""
    if texture_path.exists():
        return (f'    <texture name="map_texture" type="2d" file="{texture_path}"/>\n'
                f'    <material name="map_material" texture="map_texture" rgba="1 1 1 1" '
                f'specular="0" shininess="0" emission="0.6"/>')
    return '    <material name="map_material" rgba="0.20 0.30 0.45 1.0" specular="0.2" shininess="0.1"/>'


def main(map_dir, robot_dir):
    map_dir = Path(map_dir).resolve()
    robot_dir = Path(robot_dir).resolve()
    out_dir = map_dir / "mujoco"
    out_dir.mkdir(exist_ok=True)

    meta = yaml.safe_load(open(map_dir / "meta.yaml"))
    if meta["up_axis"] != "z" or meta["units"] != "meter":
        raise ValueError("地图必须是 z 轴朝上、单位为米")
    waypoints = yaml.safe_load(open(map_dir / "waypoints.yaml"))["waypoints"]
    global HEIGHT_PERCENTILE
    HEIGHT_PERCENTILE = meta.get("hfield_height_percentile")
    if HEIGHT_PERCENTILE is not None:
        print(f"高度场栅格高度取 {HEIGHT_PERCENTILE} 分位数（meta.yaml 指定）")

    heights, lo, observed = build_heightfield(read_pcd_xyz(map_dir / "dense_map.pcd"))
    # 可选：在高度场四周补零高度边距（地面扫描块太小时，防止机器狗脚踩出场外坠落）
    margin = int(meta.get("hfield_margin", 0))
    if margin > 0:
        heights = np.pad(heights, margin)
        observed = np.pad(observed, margin)
        lo = lo - margin * RESOLUTION
        print(f"高度场已加 {margin * RESOLUTION:.1f} m 零高度边距")
    nrow, ncol = heights.shape
    hfield_path = out_dir / "collision_hfield.bin"
    write_hfield_bin(hfield_path, heights)
    # 供检查和导航脚本使用，与仿真中的碰撞体完全一致
    np.savez(out_dir / "collision_hfield.npz", heights=heights, lo=lo, resolution=RESOLUTION,
             observed=observed)

    # MuJoCo 高度场的采样点落在 [-rx, rx] 的两端之间，共 ncol 个
    rx = (ncol - 1) * RESOLUTION / 2
    ry = (nrow - 1) * RESOLUTION / 2
    scene = SCENE_TEMPLATE.format(
        map_id=meta["map_id"],
        mesh=map_dir / "mesh" / "scene.obj",
        material=map_material(map_dir / "mesh" / "scene_texture.png"),
        hfield=hfield_path,
        rx=rx, ry=ry, zmax=heights.max(), base=BASE_THICKNESS,
        cx=lo[0] + rx, cy=lo[1] + ry, cz=-HFIELD_DROP, hfield_drop=HFIELD_DROP,
        markers=waypoint_markers(waypoints, heights, lo),
    )
    scene_path = robot_dir / f"scene_slam_{meta['map_id']}.xml"
    scene_path.write_text(scene)
    print(f"高度场 {nrow} 行 x {ncol} 列，最高 {heights.max():.2f} m")
    print(f"场景文件已写入 {scene_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
