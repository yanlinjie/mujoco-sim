"""用文字指定目的地，让 Go2 在仿真中走过去：文字 → 场景图找目标站位 → A* 规划 → 行走控制。

    # 接 unitree_mujoco 仿真器（DDS）；仿真器没在运行时自动在后台启动（弹出 MuJoCo 窗口，
    # 日志写到 /tmp/umj_slam_scene.log）。可给多个目标，依次前往，到达最后一个后站立，Ctrl+C 退出
    python -I navigate_to.py dds <地图包> "木箱" "印着DGX STATION的纸箱" "充电桩"
    # 不开窗口，在 MuJoCo 中从原点出发跑一遍，输出轨迹图
    python -I navigate_to.py headless <地图包> <场景 xml> "北侧通道里的纸箱" [--image 轨迹图.png]

目标可以是节点编号（obj_03、wp_05、region_north、place_007）或任意中文描述。
行走控制沿用 waypoint_tour.py（A* 在碰撞高度场上规划，开环小跑步态 + 位置闭环）。
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from scene_graph import SceneGraph  # noqa: E402


def build_tour(sg, targets, start=(0.0, 0.0)):
    """依次解析每个目标，返回 waypoint_tour 的路线格式 [(编号, 名称, xy, 朝向)]，并打印解析结果。"""
    tour, pos = [], np.asarray(start, float)
    for t in targets:
        g = sg.goal_pose(t, near=pos)
        route = sg.route_description(pos, g["id"])
        print(f"「{t}」→ {g['id']}「{g['name']}」，站位 ({g['x']:.2f}, {g['y']:.2f})，朝向 {np.degrees(g['yaw']):.0f}°，"
              f"经过 {' → '.join(route or ['（通行网络上不连通）'])}"
              + ("" if g["reachable"] else "；注意：该物体走不到跟前，只能走到最近处"))
        tour.append((g["id"], g["name"], np.array([g["x"], g["y"]]), g["yaw"]))
        pos = np.array([g["x"], g["y"]])
    return tour


SIM_DIR = HERE.parents[2] / "unitree_mujoco" / "simulate" / "build"
SIM_LOG = Path("/tmp/umj_slam_scene.log")


def pgrep(pattern, program):
    """命令行匹配 pattern、且可执行文件名包含 program 的其他进程：[(pid, 命令行)]。
    只按命令行匹配会把“命令文本里含有同样字符串”的 shell 也算进来。"""
    out = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True).stdout
    procs = [line.split(" ", 1) for line in out.splitlines() if " " in line]
    return [(int(pid), cmd) for pid, cmd in procs
            if int(pid) != os.getpid() and program in Path(cmd.split()[0]).name]


def ensure_simulator(map_dir):
    """确认没有别的控制程序在发指令；仿真器没运行就在后台启动本地图的场景。返回 False 表示不能继续。"""
    others = pgrep(r"waypoint_tour\.py dds|navigate_to\.py dds", "python")
    if others:
        print("有另一个控制程序在给机器狗发指令，两者会互相干扰，请先停掉它（Ctrl+C 或 kill -INT <进程号>）：")
        for pid, cmd in others:
            print(f"  {pid}  {cmd}")
        return False
    map_id = next(line.split(":", 1)[1].strip() for line in open(Path(map_dir) / "meta.yaml")
                  if line.startswith("map_id:"))
    scene = f"scene_slam_{map_id}.xml"
    if pgrep(f"unitree_mujoco -r go2 -s {scene}", "unitree_mujoco"):
        print("仿真器已在运行")
        return True
    if pgrep(r"unitree_mujoco", "unitree_mujoco") or pgrep(r"unitree_mujoco\.py", "python"):
        print("有另一个 unitree_mujoco 仿真器在运行（加载的是别的场景），两者会在同一 DDS 域冲突，请先关掉它")
        return False
    print(f"启动仿真器（{scene}），日志 {SIM_LOG} ...")
    with open(SIM_LOG, "w") as log:
        subprocess.Popen(["./unitree_mujoco", "-r", "go2", "-s", scene], cwd=SIM_DIR, stdout=log,
                         stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(60):
        if "Link_index" in SIM_LOG.read_text(errors="ignore"):
            break
        time.sleep(0.5)
    else:
        print(f"仿真器 30 秒内没有就绪，请查看 {SIM_LOG}")
        return False
    time.sleep(2)  # 等机器狗落地
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["dds", "headless"])
    ap.add_argument("map_dir")
    ap.add_argument("rest", nargs="+", help="headless 模式第一个为场景 xml，其余为目标")
    ap.add_argument("--image", help="headless 模式输出轨迹图")
    ap.add_argument("--from", dest="start", nargs=2, type=float, default=[0.0, 0.0], metavar=("X", "Y"),
                    help="解析“最近的某物体”时参考的起点，dds 模式默认原点")
    args = ap.parse_args()
    import waypoint_tour

    sg = SceneGraph.load(args.map_dir)
    if args.mode == "headless":
        scene, targets = args.rest[0], args.rest[1:]
        ok = waypoint_tour.run_headless(args.map_dir, scene, args.image, tour=build_tour(sg, targets, args.start))
        sys.exit(0 if ok else 1)
    tour = build_tour(sg, args.rest, args.start)
    if ensure_simulator(args.map_dir):
        waypoint_tour.run_dds(args.map_dir, tour=tour, hold=True)


if __name__ == "__main__":
    main()
