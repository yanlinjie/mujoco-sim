"""让 MuJoCo 中的 Go2 沿给定的折线（一串地图坐标点）行走。

与 nav_demo.py 的区别：nav_demo.py 只接受点位名称，自己去调规划服务；本脚本直接接受坐标，
不关心坐标来自哪里——可以是 A* 规划服务返回的 path，也可以是人工给出的一串点。
控制器与 waypoint_tour.py 完全相同（同一套步态、跟踪、到达判定），只是不在控制器里重新规划。

输入（三选一）：
    --points X,Y X,Y ...      直接在命令行给出折线拐点（地图坐标系，米），整条折线算一段
    --path-json 文件           JSON 文件，"-" 表示从标准输入读。支持三种内容（点可写成 [x, y]、
                               [x, y, ...] 或 {"x": .., "y": ..}，多出的列和键忽略，
                               方便直接接收 VLN 等其他导航算法的轨迹；坐标必须是地图坐标系）：
                                 1) [[x, y], ...]                              整条折线算一段
                                 2) {"points": [[x, y], ...], "final_yaw": 1.57} 同上，可带终点朝向
                                 3) A* MCP get_plan 的完整返回（含 legs）     每个 leg 一段，
                                    终点朝向取 goal.stand_pose.yaw，并核对地图版本
可选：
    --final-yaw YAW           走到最后一个点后原地转到此朝向（弧度）；不给则保持最后一小段的行进方向
    --result-json 文件         走完（或放弃）后把每段的到达结果写成 JSON，供调用方判断
    --force                   折线穿过障碍或未扫描区域时也照走（默认拒绝）

折线第一个点的处理：机器狗当前位置离第一个点不超过 0.3 m 时，把第一个点替换成当前位置
（A* 返回的 path 第一个点就是起点，属于这种情况）；否则先从当前位置直线走到第一个点。

用法（在 slam_map_demo 目录下，P=/home/oc/miniconda3/envs/unitree-mujoco/bin/python）：
    # 接已启动的 unitree_mujoco 仿真器（DDS），走完后保持站立，Ctrl+C 退出
    $P -I -u scripts/follow_path.py dds <地图包> --points 0,0 1.5,0 1.5,2 --final-yaw 1.57
    $P -I -u scripts/follow_path.py dds <地图包> --path-json <get_plan 输出文件>
    # 不开窗口，在 MuJoCo 里快速跑一遍并输出轨迹图（用于先验证折线能不能走通）
    $P -I -u scripts/follow_path.py headless <地图包> --scene <场景 xml> --points ... [--image 轨迹.png] [--spawn X Y YAW]

停止：Ctrl+C 或 kill <pid>（SIGTERM）都会先让机器狗站稳 3 秒再退出；不要用 kill -9，
      否则仿真器会一直执行最后一条行走指令，机器狗会原地乱动。

退出码：全部到达为 0；有段没到达、摔倒、被中断或输入有误为 1。
"""
import argparse
import json
import re
import signal
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I 模式下需显式加入本目录
import waypoint_tour  # noqa: E402
from nav_grid import NavGrid, ROBOT_RADIUS  # noqa: E402

REPLACE_FIRST_POINT = 0.3  # 当前位置离折线第一个点不超过此距离时，直接以当前位置为起点


def parse_point(text):
    try:
        x, y = (float(v) for v in text.replace("，", ",").split(","))
    except ValueError:
        raise SystemExit(f"点坐标格式应为 X,Y（如 1.5,-0.3），收到「{text}」")
    return [x, y]


def to_xy_list(points, where):
    """把各种写法的轨迹点统一成 [[x, y], ...]。

    接受 [x, y]、[x, y, 第三个数, ...]（多出的列忽略，可能是 z 也可能是朝向，一律不用）
    和 {"x": .., "y": .., ...}（其他键忽略）。A* 以外的导航算法（如 VLN）给出的轨迹常带多余的列。"""
    if not isinstance(points, list) or not points:
        raise SystemExit(f"{where}：轨迹应为非空的点列表")
    out = []
    for i, p in enumerate(points):
        try:
            if isinstance(p, dict):
                xy = [float(p["x"]), float(p["y"])]
            else:
                xy = [float(p[0]), float(p[1])]
        except (KeyError, IndexError, TypeError, ValueError):
            raise SystemExit(f"{where} 第 {i + 1} 个点格式错误：{p!r}；应为 [x, y]、[x, y, ...] 或 {{\"x\": .., \"y\": ..}}")
        out.append(xy)
    return out


def load_legs(a, meta):
    """返回 [{"name": ..., "path": [[x, y], ...], "yaw": 终点朝向或 None}, ...]"""
    if a.points:
        return [{"id": "end", "name": "终点", "path": [parse_point(p) for p in a.points], "yaw": a.final_yaw}]
    text = sys.stdin.read() if a.path_json == "-" else Path(a.path_json).read_text()
    data = json.loads(text)
    if isinstance(data, list):
        return [{"id": "end", "name": "终点", "path": to_xy_list(data, "轨迹"), "yaw": a.final_yaw}]
    if "legs" in data:  # A* 规划服务的返回
        if not data.get("ok"):
            raise SystemExit(f"规划服务返回失败：{data.get('error')}")
        if data.get("map_version") != str(meta["version"]) or data.get("map_id") != meta["map_id"]:
            raise SystemExit(f"规划结果来自地图 {data.get('map_id')}/{data.get('map_version')}，"
                             f"与本地地图包 {meta['map_id']}/{meta['version']} 不一致，坐标对不上")
        for w in data.get("warnings", []):
            print("规划服务提示：" + w)
        legs = [{"id": l["goal"]["id"], "name": l["goal"]["name"], "path": to_xy_list(l["path"], l["goal"]["name"]),
                 "yaw": l["goal"]["stand_pose"]["yaw"]} for l in data["legs"]]
        if a.final_yaw is not None:
            legs[-1]["yaw"] = a.final_yaw
        return legs
    if "points" in data:
        return [{"id": "end", "name": "终点", "path": to_xy_list(data["points"], "points"),
                 "yaw": a.final_yaw if a.final_yaw is not None else data.get("final_yaw")}]
    raise SystemExit("--path-json 内容无法识别：应为点列表、{\"points\": ...} 或规划服务 /plan 的返回")


def check_legs(grid, legs, force):
    """逐段检查折线：穿过障碍/未扫描区域时拒绝（除非 --force），离障碍太近时提示。"""
    bad = False
    for leg in legs:
        pts = np.asarray(leg["path"], float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 1 or not np.isfinite(pts).all():
            raise SystemExit(f"{leg['name']}：折线格式错误，应为 [[x, y], ...]")
        leg["path"] = pts
        for i, (p, q) in enumerate(zip(pts[:-1], pts[1:])):
            n = max(int(np.linalg.norm(q - p) / (grid.res / 2)), 1)
            c = min(grid.clearance_at(p + (q - p) * t) for t in np.linspace(0, 1, n + 1))
            where = f"{leg['name']} 第 {i + 1} 小段 ({p[0]:.2f}, {p[1]:.2f}) → ({q[0]:.2f}, {q[1]:.2f})"
            if c <= 0:
                print(f"[错误] {where} 穿过障碍、未扫描区域或地图外")
                bad = True
            elif c < ROBOT_RADIUS:
                print(f"[提示] {where} 离障碍最近仅 {c:.2f} m（规划标准 {ROBOT_RADIUS} m），可能蹭到")
    if bad and not force:
        raise SystemExit("折线不可通行；请让规划算法（A* 或 VLN 等）重新给出路线，或确认无误后加 --force")


def protect_negative_points(argv):
    """argparse 会把 -0.37,4.54 这类负坐标当成选项名；前面加一个空格让它按普通参数处理（float 会忽略空格）。"""
    return [" " + s if re.match(r"^-\d*\.?\d+[,，]", s) else s for s in argv]


class PathController(waypoint_tour.TourController):
    """每段沿给定折线行走；走完所有段后（或任何一段放弃后）写结果文件。"""

    leg_paths = []
    result_json = None
    instance = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        PathController.instance = self
        self.result_written = False

    def _begin_leg(self, xy):
        wp_id, name, _, _ = self.tour[self.leg]
        path = [np.asarray(p, float) for p in self.leg_paths[self.leg]]
        xy = np.asarray(xy, float)
        if np.linalg.norm(path[0] - xy) <= REPLACE_FIRST_POINT:
            path[0] = xy
        elif self.grid.segment_free(xy, path[0], radius=0.15):
            self.log(f"[提示] 当前位置 ({xy[0]:.2f}, {xy[1]:.2f}) 离折线起点 {np.linalg.norm(path[0] - xy):.2f} m，"
                     "先直线走过去")
            path.insert(0, xy)
        else:  # 直线会穿过障碍（例如上一段没走到就放弃了）：用同一套 A* 先规划到折线起点
            bridge = self.grid.plan(xy, path[0])
            if bridge is None:
                self.log(f"[跳过] {wp_id} {name}：当前位置 ({xy[0]:.2f}, {xy[1]:.2f}) 到折线起点不连通")
                self.results.append((wp_id, name, float(np.linalg.norm(self.tour[self.leg][2] - xy)), None, None))
                self._next_leg()
                return
            self.log(f"[提示] 当前位置 ({xy[0]:.2f}, {xy[1]:.2f}) 到折线起点的直线有障碍，先按 A* 绕过去")
            path = [np.asarray(p, float) for p in bridge] + path[1:]
        self.path, self.path_index = path, 0
        self.leg_start = self.time
        self.best_dist, self.last_progress = np.inf, self.time
        length = sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:]))
        self.log(f"[出发] 前往 {wp_id} {name}：{length:.1f} m，{len(path) - 1} 段折线")
        self._goto("follow")

    def _next_leg(self):
        super()._next_leg()
        if self.state == "done":
            self.write_result()

    def write_result(self, note=None):
        if self.result_json and not self.result_written:
            legs = [{"id": r[0], "name": r[1], "reached": r[3] is not None,
                     "position_error": None if r[2] is None else round(float(r[2]), 3),
                     "yaw_error": None if r[3] is None else round(float(r[3]), 3),
                     "time": None if r[4] is None else round(float(r[4]), 1)} for r in self.results]
            out = {"ok": len(legs) == len(self.tour) and all(l["reached"] for l in legs), "legs": legs}
            if note:
                out["note"] = note
            self.result_written = True
            Path(self.result_json).write_text(json.dumps(out, ensure_ascii=False, indent=2))
            self.log(f"结果已写入 {self.result_json}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["dds", "headless"])
    ap.add_argument("map_dir")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--points", nargs="+", metavar="X,Y")
    src.add_argument("--path-json", metavar="文件")
    ap.add_argument("--final-yaw", type=float)
    ap.add_argument("--result-json")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-hold", action="store_true", help="dds 模式走完后站 3 秒就退出（默认保持站立直到 Ctrl+C）")
    ap.add_argument("--dds-domain", type=int, default=1)
    ap.add_argument("--scene", help="headless 模式的场景 xml")
    ap.add_argument("--image", help="headless 模式的轨迹图输出路径")
    ap.add_argument("--spawn", nargs=3, type=float, metavar=("X", "Y", "YAW"),
                    help="headless 模式的出生位姿，默认地图原点朝 +x（与 C++ 仿真器相同）")
    a = ap.parse_args(protect_negative_points(sys.argv[1:]))

    map_dir = Path(a.map_dir)
    meta = yaml.safe_load(open(map_dir / "meta.yaml"))
    grid = NavGrid.load(map_dir / "mujoco" / "collision_hfield.npz")
    legs = load_legs(a, meta)
    check_legs(grid, legs, a.force)

    tour = []
    for i, leg in enumerate(legs):
        pts = leg["path"]
        if leg["yaw"] is None:  # 没有指定终点朝向：保持最后一小段的行进方向，相当于不转身
            d = pts[-1] - (pts[-2] if len(pts) > 1 else (legs[i - 1]["path"][-1] if i else pts[-1]))
            leg["yaw"] = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 1e-6 else 0.0
        tour.append((leg["id"], leg["name"], pts[-1], float(leg["yaw"])))
        print(f"第 {i + 1} 段 {leg['name']}：{len(pts)} 个点 {np.round(pts, 2).tolist()}，"
              f"到达后朝向 {np.degrees(leg['yaw']):+.0f}°")

    PathController.leg_paths = [l["path"] for l in legs]
    PathController.result_json = a.result_json
    waypoint_tour.TourController = PathController  # run_dds / run_headless 内部按此名字创建控制器

    # 后台运行（nohup … &）时 shell 会让进程忽略 SIGINT；这里显式接管 SIGINT 和 SIGTERM，
    # 使 kill <pid> 与 Ctrl+C 一样：先站稳 3 秒、写出结果再退出，而不是停在半步姿态
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    if a.mode == "headless" and not a.scene:
        sys.exit("headless 模式需要 --scene <场景 xml>")
    ok = False
    try:
        if a.mode == "dds":
            waypoint_tour.run_dds(str(map_dir), tour=tour, hold=not a.no_hold, domain=a.dds_domain)
            c = PathController.instance
            ok = c is not None and c.state == "done" and all(r[3] is not None for r in c.results)
        else:
            if a.spawn:
                _patch_spawn(*a.spawn)
            ok = waypoint_tour.run_headless(str(map_dir), a.scene, a.image, tour=tour)
    finally:
        # 摔倒、Ctrl+C 中断等没有正常走完的情况，也把已有结果写出去
        if PathController.instance is not None:
            PathController.instance.write_result(note="未正常走完（摔倒或被中断），见日志")
    sys.exit(0 if ok else 1)


def _patch_spawn(x, y, yaw):
    """让 run_headless 加载模型后把机器狗挪到指定位姿（替换自由关节的初始位置和朝向）。"""
    import mujoco

    original = mujoco.MjData

    def make_data(model):
        data = original(model)
        data.qpos[0:2] = x, y
        data.qpos[3:7] = np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)
        return data

    mujoco.MjData = make_data


if __name__ == "__main__":
    main()
