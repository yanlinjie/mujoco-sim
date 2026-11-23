"""Plan routes by waypoint name, coordinate or scene-graph target."""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml

if __package__:
    from .nav_grid import NavGrid, ROBOT_RADIUS
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from nav_grid import NavGrid, ROBOT_RADIUS


class PlanError(Exception):
    pass


class Planner:
    def __init__(self, map_dir):
        self.map_dir = Path(map_dir).resolve()
        meta = yaml.safe_load(open(self.map_dir / "meta.yaml"))
        self.map_id, self.map_version = meta["map_id"], meta["version"]
        self.grid = NavGrid.load(self.map_dir / "mujoco" / "collision_hfield.npz")
        self.waypoints = yaml.safe_load(open(self.map_dir / "waypoints.yaml"))["waypoints"]
        self.sg = None
        if (self.map_dir / "scene_graph" / "scene_graph.json").exists():
            if __package__:
                from .scene_graph.scene_graph import SceneGraph
            else:
                sys.path.insert(0, str(Path(__file__).resolve().parent / "scene_graph"))
                from scene_graph import SceneGraph
            self.sg = SceneGraph.load(self.map_dir)
            self.sg._grid = self.grid  # 与规划共用同一份导航栅格

    # ---------- 名称解析 ----------
    def waypoint_list(self):
        return [{"id": wp["id"], "name": wp["name"], "semantic_tags": wp.get("semantic_tags", []),
                 "stand_pose": wp["stand_pose"], "lookat": wp.get("lookat")} for wp in self.waypoints]

    def resolve(self, query):
        if isinstance(query, dict):
            return self._resolve_xy(query)
        q = str(query).strip()
        if not q:
            raise PlanError("目标名称为空")
        ql = q.lower()
        rules = [
            lambda wp: wp["id"].lower() == ql,
            lambda wp: wp["name"] == q,
            lambda wp: ql in [t.lower() for t in wp.get("semantic_tags", [])],
            lambda wp: q in wp["name"],
        ]
        for rule in rules:
            hits = [wp for wp in self.waypoints if rule(wp)]
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise PlanError(f"「{q}」匹配到多个点位："
                                + "、".join(f"{wp['id']}「{wp['name']}」" for wp in hits) + "，请写得更具体")
        if self.sg is not None:
            hit = self._resolve_scene_graph(q)
            if hit is not None:
                return hit
            raise PlanError(f"找不到名为「{q}」的点位，场景图中也没有名称匹配的物体或区域")
        raise PlanError(f"找不到名为「{q}」的点位")

    def _resolve_xy(self, q):
        """坐标目标 {"x": 2.0, "y": 3.0, "yaw": 可选}：不查点位库，直接以该坐标为站位。"""
        try:
            x, y = float(q["x"]), float(q["y"])
            yaw = None if q.get("yaw") is None else float(q["yaw"])
        except (KeyError, TypeError, ValueError):
            raise PlanError('坐标目标格式应为 {"x": 2.0, "y": 3.0}，可选 "yaw"（弧度）')
        if not all(math.isfinite(v) for v in (x, y) + (() if yaw is None else (yaw,))):
            raise PlanError("坐标目标不是有效数字")
        if self.grid.clearance_at((x, y)) <= 0:
            raise PlanError(f"坐标目标 ({x:.2f}, {y:.2f}) 在障碍、未扫描区域或地图外")
        return {"id": "xy", "name": f"坐标 ({x:.2f}, {y:.2f})", "type": "xy", "semantic_tags": [],
                "stand_pose": {"x": round(x, 3), "y": round(y, 3), "z": 0.0,
                               "yaw": None if yaw is None else round(yaw, 4)},
                "lookat": None}

    def _resolve_scene_graph(self, q):
        """在场景图里按编号或关键词找物体、区域，返回与点位相同格式的目标；找不到返回 None。"""
        res = self.sg.find(q, top_k=100, types=("object", "region"), semantic=False)
        if not res:
            return None
        if res[0]["match"] == "id":
            cands = [res[0]["node"]]
        else:
            # 同类物体的先后只取决于检测次数，没有意义，因此命中多个物体一律视为有歧义
            cands = [r["node"] for r in res if r["node"]["type"] == "object"] or \
                    [r["node"] for r in res if r["score"] == res[0]["score"]]
        if len(cands) > 1:
            raise PlanError(f"「{q}」在场景图中匹配到多个：" + "、".join(self._sg_label(n) for n in cands)
                            + "，请写编号或加上所在区域、更具体的名称")
        n = cands[0]
        if n["type"] not in ("object", "region"):
            return None
        g = self.sg.goal_pose(n["id"])
        lookat = None
        if n["type"] == "object":
            lookat = {"x": n["position"][0], "y": n["position"][1], "z": n["position"][2]}
        return {"id": n["id"], "name": n["name"], "type": n["type"],
                "semantic_tags": [n["label"]] if n.get("label") else [],
                "region": self.sg.node(g["region"])["name"], "reachable": g["reachable"],
                "stand_pose": {"x": round(g["x"], 3), "y": round(g["y"], 3), "z": 0.0, "yaw": round(g["yaw"], 3)},
                "lookat": lookat}

    def _sg_label(self, n):
        region = self.sg.node(n["region"])["name"] if n.get("region") else ""
        return f"{n['id']}「{n['name']}」" + (f"（{region}）" if region else "")

    def scene_graph_list(self):
        if self.sg is None:
            raise PlanError("当前地图没有场景图")
        out = []
        for n in self.sg.regions + self.sg.objects:
            g = self.sg.goal_pose(n["id"])
            item = {"id": n["id"], "type": n["type"], "name": n["name"], "aliases": n.get("aliases", []),
                    "position": n["position"],
                    "stand_pose": {"x": round(g["x"], 3), "y": round(g["y"], 3), "z": 0.0, "yaw": round(g["yaw"], 3)}}
            if n["type"] == "object":
                item.update(label=n["label"], region=self.sg.node(n["region"])["name"],
                            size=n["size"], reachable=g["reachable"], dynamic=n.get("dynamic", False))
            out.append(item)
        return out

    # ---------- 规划 ----------
    def plan(self, start, goals, step=None):
        t0 = time.perf_counter()
        x, y = float(start["x"]), float(start["y"])
        if not (math.isfinite(x) and math.isfinite(y)):
            raise PlanError("起点坐标不是有效数字")
        if isinstance(goals, str):
            goals = [g for g in goals.replace("，", ",").split(",") if g.strip()]
        if not goals:
            raise PlanError("至少需要一个目标名称")
        targets = [self.resolve(g) for g in goals]  # 先全部解析，名称有误时不做任何规划

        pos = np.array([x, y])
        warnings = []
        if self.grid.clearance_at(pos) <= 0:
            warnings.append("起点在障碍、未扫描区域或地图外，路径第一段会先直线回到最近的可通行处")
        legs, total = [], 0.0
        for query, wp in zip(goals, targets):
            sp = wp["stand_pose"]
            goal_xy = np.array([sp["x"], sp["y"]])
            path = self.grid.plan(pos, goal_xy)
            if path is None:
                path = self._plan_from_tight_start(pos, goal_xy)
                if path is not None:
                    warnings.append(f"起点离障碍仅 {self.grid.clearance_at(pos):.2f} m，"
                                    f"前往 {wp['id']} 时先直线移到最近的可通行处")
            if path is None:
                raise PlanError(f"从 ({pos[0]:.2f}, {pos[1]:.2f}) 到 {wp['id']}「{wp['name']}」规划失败：目标不可达")
            if not wp.get("reachable", True):
                warnings.append(f"{wp['id']}「{wp['name']}」离可通行区域较远，站位只能到达附近，到不了跟前")
            path = drop_collinear(np.asarray(path, float))
            seg = np.diff(path, axis=0)
            length = float(np.linalg.norm(seg, axis=1).sum())
            leg = {
                "query": query,
                "goal": {"id": wp["id"], "name": wp["name"], "semantic_tags": wp.get("semantic_tags", []),
                         "stand_pose": sp, "lookat": wp.get("lookat"),
                         **{k: wp[k] for k in ("type", "region") if k in wp}},
                "from": {"x": round(float(pos[0]), 3), "y": round(float(pos[1]), 3)},
                "length": round(length, 3),
                "path": np.round(path, 3).tolist(),
                "headings": np.round(np.arctan2(seg[:, 1], seg[:, 0]), 4).tolist(),
            }
            if step:
                leg["dense_path"] = np.round(densify(path, float(step)), 3).tolist()
            legs.append(leg)
            total += length
            pos = goal_xy
        return {
            "ok": True,
            "map_id": self.map_id, "map_version": self.map_version, "frame": "map",
            "units": {"length": "meter", "angle": "radian"},
            "start": {k: start[k] for k in ("x", "y", "yaw") if k in start},
            "goals": [wp["id"] for wp in targets],
            "total_length": round(total, 3),
            "legs": legs,
            "warnings": warnings,
            "planning_time_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def _plan_from_tight_start(self, start, goal):
        """起点离障碍比避障半径还近（例如停在家具旁）时，NavGrid.plan 会因起点被围住而失败：
        先直线移到与目标连通的最近可通行栅格（直线不穿过实际障碍），再从那里规划。"""
        free = self.grid.clearance > ROBOT_RADIUS
        esc = self.grid._escape_cell(start, free, self.grid.to_cell(goal))
        if esc is None:
            return None
        rest = self.grid.plan(self.grid.to_xy(esc), goal)
        return None if rest is None else [np.asarray(start, float)] + list(rest)

    def handle(self, req):
        """处理一个请求字典，总是返回可序列化的字典（错误也以 JSON 形式返回）。"""
        try:
            kind = req.get("type", "plan")
            if kind == "health":
                return {"ok": True, "map_id": self.map_id, "map_version": self.map_version,
                        "waypoints": len(self.waypoints), "scene_graph": self.sg is not None}
            if kind == "scene_graph":
                return {"ok": True, "map_id": self.map_id, "map_version": self.map_version,
                        "nodes": self.scene_graph_list()}
            if kind == "waypoints":
                return {"ok": True, "map_id": self.map_id, "map_version": self.map_version,
                        "waypoints": self.waypoint_list()}
            if kind != "plan":
                raise PlanError(f"未知的请求类型 {kind}，可用 plan / waypoints / health / scene_graph")
            start = req.get("start")
            if not isinstance(start, dict) or "x" not in start or "y" not in start:
                raise PlanError('缺少起点，格式为 "start": {"x": 0.0, "y": 0.0}')
            step = req.get("step")
            if step is not None and float(step) <= 0:
                step = None
            return self.plan(start, req.get("goals"), step)
        except PlanError as e:
            return {"ok": False, "error": str(e),
                    "available": [f"{wp['id']} {wp['name']}" for wp in self.waypoints]}
        except (TypeError, ValueError, KeyError) as e:
            return {"ok": False, "error": f"请求格式错误：{e}"}


def drop_collinear(path, tol=1e-3):
    """去掉重合点和三点共线时的中间点（贴障碍起步时栅格路径里常有一串同向小段）。"""
    out = [path[0]]
    for p in path[1:]:
        if np.linalg.norm(p - out[-1]) < tol:
            continue
        if len(out) >= 2:
            a, b = out[-2], out[-1]
            u, v = b - a, p - b
            if abs(u[0] * v[1] - u[1] * v[0]) < tol * np.linalg.norm(u) * np.linalg.norm(v) and np.dot(u, v) > 0:
                out[-1] = p
                continue
        out.append(p)
    if len(out) == 1:
        out.append(path[-1])
    return np.array(out)


def densify(path, step):
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(int(math.ceil(np.linalg.norm(b - a) / step)), 1)
        out.extend(a + (b - a) * t for t in np.arange(1, n + 1) / n)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description="按点位名称做 A* 路径规划，输出 JSON")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="规划一次并打印 JSON")
    p.add_argument("map_dir")
    p.add_argument("--start", nargs=2, type=float, required=True, metavar=("X", "Y"))
    p.add_argument("--yaw", type=float)
    p.add_argument("--goals", nargs="+", required=True)
    p.add_argument("--step", type=float)
    a = ap.parse_args()

    planner = Planner(a.map_dir)
    start = {"x": a.start[0], "y": a.start[1]}
    if a.yaw is not None:
        start["yaw"] = a.yaw
    res = planner.handle({"type": "plan", "start": start, "goals": a.goals, "step": a.step})
    print(json.dumps(res, ensure_ascii=False, indent=2))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
