"""按点位名称规划路径：输入当前位置和一个或多个目标点位名称，按先后顺序依次前往，
用 A*（nav_grid.NavGrid.plan，与巡航脚本同一套导航栅格和避障标准）规划每一段，输出 JSON。

目标名称的匹配顺序：点位编号（wp_02）→ 名称完全相同 → 语义标签相同 → 名称包含输入的文字。
同一级匹配到多个点位时报错并列出候选，不猜测。
地图包带场景图（<地图包>/scene_graph/scene_graph.json）时，点位都匹配不到的名称再到场景图里按关键词
找物体或区域（如 obj_17、"木箱"、"北侧通道里的纸箱"、"北侧通道"），目标为物体的到达站位；
只用关键词，不用 CLIP 语义检索；同类物体有多个时同样报错并列出候选编号。

两种用法：

1) 命令行，直接打印 JSON：
    python -I nav_service.py plan <地图包> --start X Y [--yaw YAW] --goals 广告牌 厕所通道 [--step 0.1]

2) 局域网服务（同时提供 HTTP 和 WebSocket，监听所有网卡）：
    python -I nav_service.py serve <地图包> [--host 0.0.0.0] [--http-port 8765] [--ws-port 8766]

   HTTP：
     GET  /health                  服务状态
     GET  /waypoints               点位列表
     POST /plan                    请求体 {"start": {"x": 0, "y": 0, "yaw": 0}, "goals": ["广告牌", "厕所通道"], "step": 0.1}
     GET  /plan?x=0&y=0&goals=广告牌,厕所通道[&yaw=0][&step=0.1]
   WebSocket（ws://<主机>:8766），每条消息一个 JSON：
     {"type": "plan", "start": {...}, "goals": [...], "step": 0.1, "request_id": 可选，原样返回}
     {"type": "waypoints"}   {"type": "health"}   {"type": "scene_graph"}（场景图中的物体和区域）

请求字段：
    start  当前位置，地图坐标系（米）；yaw 可省略（弧度）
    goals  目标列表，按顺序前往。每一项是点位名称（字符串），或坐标 {"x": 2.0, "y": 3.0, "yaw": 可选}；
           也接受逗号分隔的名称字符串
    step   可选，>0 时把折线按此间距（米）加密成稠密路径点，放在每段的 dense_path 中

返回（成功时 ok 为 true）：
    legs[i].path        A* 规划并拉直后的折线 [[x, y], ...]，第一个点是该段起点，最后一个点是目标站位
    legs[i].headings    折线每一段的行进方向（弧度），比 path 少一个
    legs[i].goal        目标点位：编号、名称、站位、到达后应朝向的 yaw、看向的位置
    total_length        各段长度之和（米）
失败时 ok 为 false，error 说明原因（例如名称找不到、目标不可达），并给出可用的点位名称。
"""
import argparse
import asyncio
import json
import math
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I 模式下需显式加入本目录
from nav_grid import NavGrid, ROBOT_RADIUS  # noqa: E402


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


# ---------- HTTP ----------
def make_http_handler(planner):
    class Handler(BaseHTTPRequestHandler):
        def parse_request(self):
            # 网址里直接写中文时（curl 不会自动编码），http.server 把请求行按 Latin-1 解码后用 str.split() 切分，
            # 而部分汉字的 UTF-8 字节（0x85、0xA0，如「充电桩」「食堂」）会被当成空白，整行被判为格式错误。
            # 先把非 ASCII 字节转成网址编码，parse_qs 会再还原成中文
            self.raw_requestline = re.sub(rb"[\x80-\xff]", lambda m: b"%%%02X" % m.group()[0], self.raw_requestline)
            return super().parse_request()

        def _send(self, obj, status=200):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")  # 允许局域网网页直接调用
            self.end_headers()
            self.wfile.write(body)

        def _reply(self, req):
            res = planner.handle(req)
            self._send(res, 200 if res.get("ok") else 400)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            # 网址里直接写中文时（curl 不会自动编码），原始 UTF-8 字节被 http.server 按 Latin-1 解码，
            # 这里还原回来；已做网址编码（%E5...）的请求不受影响
            try:
                raw = self.path.encode("latin-1").decode("utf-8")
            except UnicodeError:
                raw = self.path
            u = urlparse(raw)
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            if u.path in ("/health", "/waypoints", "/scene_graph"):
                return self._reply({"type": u.path[1:]})
            if u.path == "/plan":
                try:
                    start = {"x": float(q["x"]), "y": float(q["y"])}
                    if "yaw" in q:
                        start["yaw"] = float(q["yaw"])
                except (KeyError, ValueError):
                    return self._send({"ok": False, "error": "需要数字参数 x、y，例如 /plan?x=0&y=0&goals=广告牌"}, 400)
                return self._reply({"type": "plan", "start": start, "goals": q.get("goals", ""),
                                    "step": q.get("step")})
            self._send({"ok": False, "error": "可用接口：GET /health、GET /waypoints、GET /scene_graph、GET 或 POST /plan"}, 404)

        def do_POST(self):
            if urlparse(self.path).path != "/plan":
                return self._send({"ok": False, "error": "POST 只支持 /plan"}, 404)
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if not isinstance(req, dict):
                    raise ValueError("请求体应为 JSON 对象")
            except ValueError as e:
                return self._send({"ok": False, "error": f"请求体不是有效的 JSON：{e}"}, 400)
            req.setdefault("type", "plan")
            self._reply(req)

        def log_message(self, fmt, *args):
            print(f"[HTTP] {self.client_address[0]} {fmt % args}", flush=True)

    return Handler


# ---------- WebSocket ----------
async def ws_main(planner, host, port):
    from websockets.asyncio.server import serve

    async def handler(conn):
        peer = conn.remote_address[0] if conn.remote_address else "?"
        async for msg in conn:
            try:
                req = json.loads(msg)
                if not isinstance(req, dict):
                    raise ValueError("消息应为 JSON 对象")
            except ValueError as e:
                await conn.send(json.dumps({"ok": False, "error": f"消息不是有效的 JSON：{e}"}, ensure_ascii=False))
                continue
            res = await asyncio.to_thread(planner.handle, req)  # A* 是计算密集的，放到线程里不阻塞其他连接
            if "request_id" in req:
                res["request_id"] = req["request_id"]
            print(f"[WS] {peer} {req.get('type', 'plan')} {req.get('goals', '')} -> ok={res.get('ok')}", flush=True)
            await conn.send(json.dumps(res, ensure_ascii=False))

    async with serve(handler, host, port):
        await asyncio.Future()


def serve(planner, host, http_port, ws_port):
    httpd = ThreadingHTTPServer((host, http_port), make_http_handler(planner))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"地图 {planner.map_id}/{planner.map_version}，{len(planner.waypoints)} 个点位："
          + "、".join(wp["name"] for wp in planner.waypoints))
    if planner.sg is not None:
        print(f"场景图：{len(planner.sg.regions)} 个区域、{len(planner.sg.objects)} 个物体，可作为目标名称")
    print(f"HTTP      http://{host}:{http_port}  （/health、/waypoints、/scene_graph、/plan）")
    print(f"WebSocket ws://{host}:{ws_port}", flush=True)
    try:
        asyncio.run(ws_main(planner, host, ws_port))
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()


def main():
    ap = argparse.ArgumentParser(description="按点位名称做 A* 路径规划，输出 JSON；可作为局域网 HTTP/WebSocket 服务运行")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="规划一次并打印 JSON")
    p.add_argument("map_dir")
    p.add_argument("--start", nargs=2, type=float, required=True, metavar=("X", "Y"))
    p.add_argument("--yaw", type=float)
    p.add_argument("--goals", nargs="+", required=True)
    p.add_argument("--step", type=float)
    s = sub.add_parser("serve", help="启动 HTTP + WebSocket 服务")
    s.add_argument("map_dir")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--http-port", type=int, default=8765)
    s.add_argument("--ws-port", type=int, default=8766)
    a = ap.parse_args()

    planner = Planner(a.map_dir)
    if a.cmd == "plan":
        start = {"x": a.start[0], "y": a.start[1]}
        if a.yaw is not None:
            start["yaw"] = a.yaw
        res = planner.handle({"type": "plan", "start": start, "goals": a.goals, "step": a.step})
        print(json.dumps(res, ensure_ascii=False, indent=2))
        sys.exit(0 if res["ok"] else 1)
    serve(planner, a.host, a.http_port, a.ws_port)


if __name__ == "__main__":
    main()
