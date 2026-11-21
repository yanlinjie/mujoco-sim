"""把 CloudCompare 点选导出的点列表转成地图包的点位库 waypoints.yaml。

点选约定（在地图坐标系的点云上点选，例如 output/<地图>_below1.8m.ply）：
    每个点位点两个点，按顺序：先点站位（机器狗站在哪，点在地面上即可），再点要看向的位置（物体本身）。
    点选顺序就是巡航顺序。原点处的充电桩自动加在最前面，不用点。

点列表文件：CloudCompare「Point list picking」导出的 ASCII，两种格式都支持：
    x,y,z                  （导出 X,Y,Z）
    Point #0,x,y,z         （导出 label,X,Y,Z）

名称文件（可选）：每行对应一个点位（一对点），格式为「名称 | 标签1, 标签2」，标签可省略；
    空行和 # 开头的行忽略。不给名称文件时，名称为「点位 N」。

检查：站位离障碍不足 STAND_CLEARANCE（可原地转身）或从原点走不到时给出警告，
    并给出附近 SNAP_RADIUS 内最近的合格位置；加 --snap 时自动把站位挪过去（朝向仍对准看向的位置）。
导航栅格与巡航脚本相同（mujoco/collision_hfield.npz），因此需先运行 map_to_mujoco.py。

用法: waypoints_from_picks.py <地图包目录> <点列表.txt> [名称.txt] [--snap]
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I 模式下需显式加入本目录
from nav_grid import NavGrid, ROBOT_RADIUS  # noqa: E402

STAND_CLEARANCE = 0.45   # 与 annotate_waypoints.py 相同：Go2 半长约 0.35 m，留出原地转身空间
SNAP_RADIUS = 0.8        # 站位不合格时，在多大范围内找替代位置
NEIGHBOR_MAX_PATH = 4.5  # 规划路线不超过此长度的两个点位记为相邻
LOOKAT_FLOOR_Z = 0.1     # 看向的点低于此高度时视为点在了地面上（俯视点选常见），
LOOKAT_DEFAULT_Z = 0.5   # 看向高度改取此值，与其他地图的点位一致


def read_picks(path):
    pts = []
    for line in Path(path).read_text().splitlines():
        parts = [p for p in line.replace(";", ",").replace(" ", ",").split(",") if p.strip()]
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                pass  # 标签列，如 "Point #0" 拆开后的 "Point" / "#0"
        if len(nums) >= 3:
            pts.append(nums[-3:])
    return np.array(pts, dtype=float).reshape(-1, 3)


def read_names(path, n):
    rows = []
    if path is not None:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, _, tags = line.partition("|")
            rows.append((name.strip(), [t.strip() for t in tags.replace("，", ",").split(",") if t.strip()]))
    if path is not None and len(rows) != n:
        sys.exit(f"名称文件有 {len(rows)} 行，但点列表是 {n} 个点位（{2 * n} 个点），两者需一一对应")
    return rows or [(f"点位 {i + 1}", []) for i in range(n)]


def path_length(path):
    return float(sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:])))


def main(map_dir, picks_file, names_file=None, snap=False):
    map_dir = Path(map_dir)
    meta = yaml.safe_load(open(map_dir / "meta.yaml"))
    grid = NavGrid.load(map_dir / "mujoco" / "collision_hfield.npz")
    labels, _ = ndimage.label(grid.clearance > ROBOT_RADIUS)
    reachable = labels == labels[grid.to_cell((0.0, 0.0))]
    ok = (grid.clearance >= STAND_CLEARANCE) & reachable
    ok_rc = np.stack(np.nonzero(ok), axis=1)
    ok_xy = grid.lo + ok_rc[:, ::-1] * grid.res

    pts = read_picks(picks_file)
    if len(pts) == 0 or len(pts) % 2:
        sys.exit(f"点列表有 {len(pts)} 个点，应为偶数（每个点位一对：站位、看向的位置）")
    names = read_names(names_file, len(pts) // 2)

    waypoints = [{
        "id": "wp_00",
        "name": "充电桩（地图原点）",
        "semantic_tags": ["dock"],
        "stand_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
    }]
    n_warn = 0
    for k, ((name, tags), stand, look) in enumerate(zip(names, pts[0::2], pts[1::2]), start=1):
        xy = stand[:2].copy()
        cell = grid.to_cell(xy)
        good = grid.inside(cell) and ok[cell]
        note = ""
        if not good:
            n_warn += 1
            why = "从原点走不到" if grid.inside(cell) and not reachable[cell] else \
                f"离障碍 {grid.clearance_at(xy):.2f} m（需 ≥ {STAND_CLEARANCE} m）"
            d = np.linalg.norm(ok_xy - xy, axis=1)
            if d.size and d.min() <= SNAP_RADIUS:
                alt = ok_xy[d.argmin()]
                note = f"  ⚠ {why}；{d.min():.2f} m 外的 ({alt[0]:+.2f}, {alt[1]:+.2f}) 合格"
                if snap:
                    xy = alt
                    note += "，已挪过去"
            else:
                note = f"  ⚠ {why}；{SNAP_RADIUS} m 内没有合格位置，请重新点选"
        if look[2] < LOOKAT_FLOOR_Z:
            look = np.array([look[0], look[1], LOOKAT_DEFAULT_Z])
        yaw = float(np.arctan2(look[1] - xy[1], look[0] - xy[0]))
        waypoints.append({
            "id": f"wp_{k:02d}",
            "name": name,
            "semantic_tags": tags,
            "stand_pose": {"x": round(float(xy[0]), 3), "y": round(float(xy[1]), 3), "z": 0.0,
                           "yaw": round(yaw, 3)},
            "lookat": {"x": round(float(look[0]), 3), "y": round(float(look[1]), 3),
                       "z": round(float(look[2]), 3)},
        })
        print(f"wp_{k:02d} {name}: 站位 ({xy[0]:+.2f}, {xy[1]:+.2f}) 朝向 {np.degrees(yaw):+.0f}°，"
              f"看向 ({look[0]:+.2f}, {look[1]:+.2f}, {look[2]:.2f})，离障碍 {grid.clearance_at(xy):.2f} m{note}")

    for wp in waypoints:
        wp["neighbors"] = []
        wp["arrive_tolerance"] = {"xy": 0.15, "yaw": 0.2}
        wp["safety_distance"] = 0.3
    for i, wa in enumerate(waypoints):
        for wb in waypoints[i + 1:]:
            pa = (wa["stand_pose"]["x"], wa["stand_pose"]["y"])
            pb = (wb["stand_pose"]["x"], wb["stand_pose"]["y"])
            if np.hypot(pb[0] - pa[0], pb[1] - pa[1]) > NEIGHBOR_MAX_PATH:
                continue
            path = grid.plan(pa, pb)
            if path is not None and path_length(path) <= NEIGHBOR_MAX_PATH:
                wa["neighbors"].append(wb["id"])
                wb["neighbors"].append(wa["id"])

    out = map_dir / "waypoints.yaml"
    backup = map_dir / "waypoints_trajectory_sampled.yaml"
    if out.exists() and not backup.exists():
        shutil.copy(out, backup)  # 打包脚本沿轨迹自动取的点位，首次覆盖前留底
    with open(out, "w") as f:
        yaml.safe_dump({"map_id": meta["map_id"], "map_version": meta["version"], "waypoints": waypoints},
                       f, allow_unicode=True, sort_keys=False)
    lonely = [wp["id"] for wp in waypoints if not wp["neighbors"]]
    print(f"已写入 {out}：{len(waypoints)} 个点位（含充电桩），{n_warn} 个站位有警告"
          + (f"；没有相邻点位：{lonely}" if lonely else ""))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--snap"]
    if len(args) not in (2, 3):
        sys.exit(__doc__)
    main(*args, snap="--snap" in sys.argv)
