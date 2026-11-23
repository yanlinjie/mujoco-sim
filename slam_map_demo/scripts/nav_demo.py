"""导航演示：在本地生成路线，再让 MuJoCo 中的 Go2 沿这条路线依次走到各目标。

流程：
    1. 根据出生点或 --start 指定的位置离线规划到指定点位；
    2. 打印服务返回的每段路线；
    3. 用 waypoint_tour.py 的控制器行走，但每段的路线换成服务返回的折线（不在控制器里重新规划），
       到达每个目标后转到该点位的朝向，最后一个目标到达后保持站立。

用法（在 slam_map_demo 目录下）：
    # 先启动仿真器：cd ../unitree_mujoco/simulate/build && ./unitree_mujoco -r go2 -s scene_slam_go2_room_selfbuilt.xml
    python -I nav_demo.py dds <地图包> 广告牌 厕所通道 [--start X Y]
    # 不开窗口，在 MuJoCo 中跑一遍并输出轨迹图
    python -I nav_demo.py headless <地图包> 广告牌 厕所通道 --scene <场景 xml> [--image 轨迹图.png]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I 模式下需显式加入本目录
import waypoint_tour  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slam_map_demo.scripts.nav_service import PlanError, Planner  # noqa: E402


class ServicePathController(waypoint_tour.TourController):
    """每段使用规划服务给出的折线；只把折线第一个点换成机器狗当前的实际位置。"""

    service_paths = []

    def _begin_leg(self, xy):
        if self.leg >= len(self.service_paths):
            return super()._begin_leg(xy)
        wp_id, name, _, _ = self.tour[self.leg]
        path = [np.asarray(p, float) for p in self.service_paths[self.leg]]
        path[0] = np.asarray(xy, float)
        self.path, self.path_index = path, 0
        self.leg_start = self.time
        self.best_dist, self.last_progress = np.inf, self.time
        length = sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:]))
        self.log(f"[出发] 前往 {wp_id} {name}，按服务给出的路线行走：{length:.1f} m，{len(path) - 1} 段折线")
        self._goto("follow")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dds", "headless"])
    ap.add_argument("map_dir")
    ap.add_argument("goals", nargs="+")
    ap.add_argument("--start", nargs=2, type=float, default=(0.0, 0.0), metavar=("X", "Y"),
                    help="起点，默认出生点（地图原点）")
    ap.add_argument("--scene", help="headless 模式的场景 xml")
    ap.add_argument("--image", help="headless 模式的轨迹图输出路径")
    a = ap.parse_args()

    try:
        res = Planner(a.map_dir).plan({"x": a.start[0], "y": a.start[1]}, a.goals)
    except PlanError as exc:
        sys.exit(f"规划失败：{exc}")
    print(f"路线：出生点 ({a.start[0]:.2f}, {a.start[1]:.2f}) → "
          + " → ".join(f"{l['goal']['name']}" for l in res["legs"]) + f"，全长 {res['total_length']:.2f} m")
    for l in res["legs"]:
        sp = l["goal"]["stand_pose"]
        print(f"  {l['goal']['id']} {l['goal']['name']}：{l['length']:.2f} m，拐点 {l['path']}，"
              f"到达后朝向 {np.degrees(sp['yaw']):+.0f}°")
    for w in res.get("warnings", []):
        print("  提示：" + w)

    tour = [(l["goal"]["id"], l["goal"]["name"],
             np.array([l["goal"]["stand_pose"]["x"], l["goal"]["stand_pose"]["y"]]),
             l["goal"]["stand_pose"]["yaw"]) for l in res["legs"]]
    ServicePathController.service_paths = [l["path"] for l in res["legs"]]
    waypoint_tour.TourController = ServicePathController  # run_dds / run_headless 内部按此名字创建控制器

    if a.mode == "dds":
        waypoint_tour.run_dds(a.map_dir, tour=tour, hold=True)
    else:
        if not a.scene:
            sys.exit("headless 模式需要 --scene <场景 xml>")
        ok = waypoint_tour.run_headless(a.map_dir, a.scene, a.image, tour=tour)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
