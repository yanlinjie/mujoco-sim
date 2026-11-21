"""目标是坐标而不是点位名称时的 A* 规划（规划服务只接受名称）。

与规划服务使用同一套导航栅格和 A*（nav_grid.NavGrid.plan，避障半径 0.3 m、尽量走通道中间），
在本进程内直接计算，不经过服务。可以给多个目标坐标，按顺序依次前往，输出整条折线。

用法：
    $P -I scripts/plan_to_xy.py <地图包> --start X Y --goals X1,Y1 [X2,Y2 ...] [--final-yaw YAW]
输出一行 JSON，可直接交给 follow_path.py --path-json：
    成功 {"ok": true, "points": [[x, y], ...], "final_yaw": 可选, "length": 米}
    失败 {"ok": false, "error": "..."}，退出码 1
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I 模式下需显式加入本目录
from nav_grid import NavGrid  # noqa: E402


def protect_negative_points(argv):
    """argparse 会把 -0.37,4.54 这类负坐标当成选项名；前面加一个空格让它按普通参数处理（float 会忽略空格）。"""
    return [" " + s if re.match(r"^-\d*\.?\d+[,，]", s) else s for s in argv]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map_dir")
    ap.add_argument("--start", nargs=2, type=float, required=True, metavar=("X", "Y"))
    ap.add_argument("--goals", nargs="+", required=True, metavar="X,Y")
    ap.add_argument("--final-yaw", type=float)
    a = ap.parse_args(protect_negative_points(sys.argv[1:]))

    grid = NavGrid.load(Path(a.map_dir) / "mujoco" / "collision_hfield.npz")
    pos = np.array(a.start, float)
    points = [pos]
    for g in a.goals:
        try:
            goal = np.array([float(v) for v in g.replace("，", ",").split(",")])
            assert goal.shape == (2,)
        except (ValueError, AssertionError):
            print(json.dumps({"ok": False, "error": f"目标坐标格式应为 X,Y，收到「{g}」"}, ensure_ascii=False))
            sys.exit(1)
        if grid.clearance_at(goal) <= 0:
            print(json.dumps({"ok": False, "error": f"目标 ({goal[0]:.2f}, {goal[1]:.2f}) 在障碍、未扫描区域或地图外"},
                             ensure_ascii=False))
            sys.exit(1)
        path = grid.plan(pos, goal)
        if path is None:
            print(json.dumps({"ok": False, "error": f"从 ({pos[0]:.2f}, {pos[1]:.2f}) 到 ({goal[0]:.2f}, {goal[1]:.2f}) "
                                                    "规划失败：目标不可达"}, ensure_ascii=False))
            sys.exit(1)
        points += [np.asarray(p, float) for p in path[1:]]
        pos = goal
    pts = np.array(points)
    out = {"ok": True, "points": np.round(pts, 3).tolist(),
           "length": round(float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()), 3)}
    if a.final_yaw is not None:
        out["final_yaw"] = a.final_yaw
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
