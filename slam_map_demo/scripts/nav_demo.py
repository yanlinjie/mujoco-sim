"""导航演示：调用路径规划服务（nav_service.py）得到路线，再让 MuJoCo 中的 Go2 沿这条路线依次走到各目标。

流程：
    1. 向规划服务发 POST /plan，起点为出生点（地图原点）或 --start 指定的位置，目标为命令行给出的点位名称；
       服务没有启动时，直接在本进程内调用同一套规划代码（nav_service.Planner），结果相同；
    2. 打印服务返回的每段路线；
    3. 用 waypoint_tour.py 的控制器行走，但每段的路线换成服务返回的折线（不在控制器里重新规划），
       到达每个目标后转到该点位的朝向，最后一个目标到达后保持站立。

用法（在 slam_map_demo 目录下）：
    # 先启动仿真器：cd ../unitree_mujoco/simulate/build && ./unitree_mujoco -r go2 -s scene_slam_go2_room_selfbuilt.xml
    python -I nav_demo.py dds <地图包> 广告牌 厕所通道 [--service http://127.0.0.1:8765] [--start X Y]
    # 不开窗口，在 MuJoCo 中跑一遍并输出轨迹图
    python -I nav_demo.py headless <地图包> 广告牌 厕所通道 --scene <场景 xml> [--image 轨迹图.png]
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I 模式下需显式加入本目录
import waypoint_tour  # noqa: E402
from nav_service import Planner  # noqa: E402


def request_plan(service, map_dir, start, goals):
    body = {"start": {"x": start[0], "y": start[1]}, "goals": goals}
    if service:
        req = urllib.request.Request(service.rstrip("/") + "/plan", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        # 本机服务不走代理
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            res = json.load(opener.open(req, timeout=10))
            print(f"已调用规划服务 {service}/plan")
            return res
        except urllib.error.HTTPError as e:
            return json.load(e)
        except (urllib.error.URLError, OSError) as e:
            print(f"规划服务 {service} 连不上（{e}），改为在本进程内规划（与服务同一套代码）")
    return Planner(map_dir).handle({"type": "plan", "start": body["start"], "goals": goals})


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
    ap.add_argument("--service", default="http://127.0.0.1:8765", help="规划服务地址；设为空字符串则直接本地规划")
    ap.add_argument("--start", nargs=2, type=float, default=(0.0, 0.0), metavar=("X", "Y"),
                    help="起点，默认出生点（地图原点）")
    ap.add_argument("--scene", help="headless 模式的场景 xml")
    ap.add_argument("--image", help="headless 模式的轨迹图输出路径")
    a = ap.parse_args()

    res = request_plan(a.service, a.map_dir, a.start, a.goals)
    if not res.get("ok"):
        sys.exit(f"规划失败：{res.get('error')}")
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
