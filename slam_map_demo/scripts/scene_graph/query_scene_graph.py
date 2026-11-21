"""场景图命令行查询工具。

    python -I query_scene_graph.py <地图包> find "北侧通道里的纸箱"      # 候选节点
    python -I query_scene_graph.py <地图包> goal "木箱" [--from X Y]      # 导航目标站位、路线经过的区域、A* 折线
    python -I query_scene_graph.py <地图包> show obj_03                   # 某个节点的全部信息和相连的边
    python -I query_scene_graph.py <地图包> describe [--from X Y]         # 整个场景的文字描述（给大语言模型）
    python -I query_scene_graph.py <地图包> list [object|waypoint|region|place]
加 --json 输出 JSON，方便其他程序调用。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scene_graph import SceneGraph  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_dir")
    ap.add_argument("command", choices=["find", "goal", "show", "describe", "list"])
    ap.add_argument("text", nargs="?")
    ap.add_argument("--from", dest="start", nargs=2, type=float, default=[0.0, 0.0], metavar=("X", "Y"))
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    sg = SceneGraph.load(args.map_dir)
    start = tuple(args.start)

    if args.command == "find":
        res = sg.find(args.text, top_k=args.top, near=start)
        if args.json:
            print(json.dumps([{"id": r["node"]["id"], "score": r["score"], "match": r["match"]} for r in res],
                             ensure_ascii=False))
            return
        if not res:
            print("没有匹配的节点")
        for r in res:
            n = r["node"]
            print(f"{r['score']:.3f} [{r['match']}] {n['id']} {n['type']} 「{n.get('name', '')}」 "
                  f"位置 ({n['position'][0]:.2f}, {n['position'][1]:.2f})"
                  + (f" 区域 {sg.node(n['region'])['name']}" if "region" in n else ""))
    elif args.command == "goal":
        g = sg.goal_pose(args.text, near=start)
        g["route_regions"] = sg.route_description(start, g["id"])
        g["topo_path"] = sg.topo_path(start, g["id"])
        g["metric_path"] = sg.metric_path(start, g["id"])
        if args.json:
            print(json.dumps(g, ensure_ascii=False))
            return
        print(f"目标 {g['id']}「{g['name']}」（{g['type']}）")
        print(f"站位 x={g['x']:.2f} y={g['y']:.2f} 朝向 {g['yaw']:.2f} rad，所在地点 {g['place']}，"
              f"区域 {sg.node(g['region'])['name']}" + ("" if g["reachable"] else "（注意：走不到跟前）"))
        print(f"经过区域：{' → '.join(g['route_regions'] or ['无法到达'])}")
        print(f"通行网络路径：{' → '.join(g['topo_path'] or [])}")
        print(f"A* 折线（{len(g['metric_path'] or []) - 1} 段）：{g['metric_path']}")
    elif args.command == "show":
        n = sg.node(args.text)
        print(json.dumps(n, ensure_ascii=False, indent=1))
        for other, e in sg.neighbors(n["id"]):
            extra = {k: v for k, v in e.items() if k not in ("source", "target", "type")}
            print(f"  --{e['type']}--> {other['id']} 「{other.get('name', '')}」 {extra if extra else ''}")
    elif args.command == "describe":
        print(sg.describe(position=start))
    elif args.command == "list":
        for n in sg.of_type(args.text or "object"):
            print(f"{n['id']} 「{n.get('name', '')}」 ({n['position'][0]:.2f}, {n['position'][1]:.2f})"
                  + (f" 区域 {sg.node(n['region'])['name']}" if "region" in n else ""))


if __name__ == "__main__":
    main()
