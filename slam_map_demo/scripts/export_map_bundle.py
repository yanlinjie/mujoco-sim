"""把地图包整理成可以发给别人的独立目录（不依赖本机绝对路径）。

内容：
- map/：地图包原样复制（点云、视觉网格、碰撞高度场、点位库、图像数据库、轨迹、坐标变换）
- mujoco_scene/：改成相对路径的 MuJoCo 场景 + Go2 模型，复制到任何位置都能直接打开
- nav2_map/：由碰撞高度场生成的 2D 栅格地图（Nav2 / ROS map_server 格式：.pgm + .yaml）
- scripts/：障碍判定、A* 规划、场景生成、自动检查、图像库查询等参考代码
- docs/、previews/：说明文档与预览图

用法: export_map_bundle.py <地图包目录> <场景 xml> <输出目录>
"""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROBOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from map_to_mujoco import FLOOR_MAX_HEIGHT  # noqa: E402

map_dir, scene_xml, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=False)

# 1) 地图包原样复制，保持脚本期望的目录结构
dst_map = out / "map"
shutil.copytree(map_dir, dst_map)
print(f"地图包 -> {dst_map}")

# 2) 可移植的 MuJoCo 场景：Go2 模型放在场景旁边，地图资源用相对路径
scene_dir = out / "mujoco_scene"
scene_dir.mkdir()
go2_dir = scene_xml.parent
shutil.copy2(go2_dir / "go2.xml", scene_dir / "go2.xml")
shutil.copytree(go2_dir / "assets", scene_dir / "assets")
text = scene_xml.read_text()
# go2.xml 设置了 meshdir="assets"，相对路径的网格和高度场文件都会被加上这个前缀，所以要多退一级
text = text.replace(str(map_dir / "mesh" / "scene.obj"), "../../map/mesh/scene.obj")
text = text.replace(str(map_dir / "mujoco" / "collision_hfield.bin"), "../../map/mujoco/collision_hfield.bin")
if re.search(r'file="/', text):
    raise RuntimeError("场景中仍有绝对路径：" + ", ".join(re.findall(r'file="(/[^"]+)"', text)))
(scene_dir / "scene_r2b_storage.xml").write_text(text)
print(f"MuJoCo 场景 -> {scene_dir}")

# 3) Nav2 栅格地图：障碍=占用，无扫描数据=未知，其余=空闲；与 nav_grid.py 的障碍判定一致
hf = np.load(map_dir / "mujoco" / "collision_hfield.npz")
H, lo, res, observed = hf["heights"], hf["lo"], float(hf["resolution"]), hf["observed"]
img = np.full(H.shape, 205, np.uint8)            # 未知
img[observed & (H <= FLOOR_MAX_HEIGHT)] = 254     # 空闲
img[observed & (H > FLOOR_MAX_HEIGHT)] = 0        # 占用
img = img[::-1]                                   # PGM 第一行在上方，地图 y 轴朝上
nav = out / "nav2_map"
nav.mkdir()
with open(nav / "map.pgm", "wb") as f:
    f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode())
    f.write(img.tobytes())
# 高度场第 i 列的中心在 lo + i*res，左下角像素的左下角为 lo - res/2
(nav / "map.yaml").write_text(yaml.safe_dump({
    "image": "map.pgm", "mode": "trinary", "resolution": res,
    "origin": [float(lo[0] - res / 2), float(lo[1] - res / 2), 0.0],
    "negate": 0, "occupied_thresh": 0.65, "free_thresh": 0.25}, sort_keys=False))
print(f"Nav2 地图 -> {nav}（{img.shape[1]}x{img.shape[0]}，空闲 {np.mean(img == 254):.0%}，"
      f"占用 {np.mean(img == 0):.0%}，未知 {np.mean(img == 205):.0%}）")

# 4) 参考脚本
scripts = out / "scripts"
scripts.mkdir()
for name in ("nav_grid.py", "map_to_mujoco.py", "waypoint_tour.py", "check_scene.py", "make_r2b_map_package.py"):
    shutil.copy2(HERE / name, scripts / name)
for name in ("query_place_db.py", "embed_clip.py", "extract_keyframes.py"):
    shutil.copy2(ROBOT / "slam_data" / "scripts" / "place_db" / name, scripts / name)

# 5) 文档与预览图
docs = out / "docs"
docs.mkdir()
for src, name in ((ROBOT / "点位数据库.md", "点位数据库.md"),
                  (ROBOT / "SLAM建图方法.md", "SLAM建图方法.md"),
                  (ROBOT / "slam_data" / "scripts" / "r2b" / "README.md", "r2b建图流程.md"),
                  (ROBOT / "slam_data" / "scripts" / "place_db" / "README.md", "图像数据库说明.md")):
    if src.exists():
        shutil.copy2(src, docs / name)
prev = out / "previews"
prev.mkdir()
for src in [*(ROBOT / "slam_map_demo" / "output" / "check_r2b_v2").glob("*.png"),
            ROBOT / "slam_data" / "out" / "r2b_static_check.png",
            ROBOT / "slam_data" / "out" / "place_db_demo" / "map_results.png",
            ROBOT / "slam_data" / "out" / "place_db_demo" / "report.md"]:
    if src.exists():
        shutil.copy2(src, prev / src.name)
print(f"脚本、文档、预览图 -> {out}")
