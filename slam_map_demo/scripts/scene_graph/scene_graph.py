"""场景图运行时接口：给导航（A*）和视觉语言导航（VLN）用的查询库。

只依赖 numpy；读取 <地图包>/scene_graph/ 下由 build_scene_graph.py 生成的文件。

    import sys; sys.path.insert(0, ".../slam_map_demo/scripts/scene_graph")
    from scene_graph import SceneGraph

    sg = SceneGraph.load("maps/r2b_storage_ns/2026-11-01.3")
    sg.find("北侧通道里的纸箱")            # 文字 → 候选节点（物体/点位/区域/地点），按匹配度排序
    goal = sg.goal_pose("木箱")            # 文字或节点编号 → 导航目标站位 {x, y, yaw, ...}
    sg.topo_path((0, 0), "木箱")           # 沿通行网络经过的地点序列（VLN 的分段子目标）
    sg.metric_path((0, 0), "木箱")         # 在碰撞高度场上 A* 规划的折线（米）
    print(sg.describe(position=(0, 0)))    # 场景的文字描述，可直接放进大语言模型的提示词

文字匹配分两种：
- 关键词：名称或别名出现在查询文字里即命中（“去纸箱旁边”命中“纸箱”）；同时出现区域名时只在该区域内找；
- 语义：没有关键词命中、或同类物体有多个需要区分时，用中文 CLIP 比较查询文字与物体图像向量
  （“印着 DGX STATION 的箱子”）。当前 Python 环境没有 torch 时，自动调用 SCENE_GRAPH_TORCH_PYTHON
  指向的解释器在子进程里计算文字向量（首次约 10 秒）。
"""
import heapq
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TORCH_PYTHON = os.environ.get("SCENE_GRAPH_TORCH_PYTHON", "/home/oc/miniconda3/bin/python")
CLIP_MODEL = os.environ.get("SCENE_GRAPH_CLIP_MODEL",
                            "/home/oc/projects/robot/slam_data/models/chinese-clip-vit-large-patch14")
NODE_TYPES = ("object", "waypoint", "region", "place")
KEYWORD_BONUS = 0.03  # 有描述词时，名称命中的物体在 CLIP 相似度上额外加的分
# 判断查询里是否有描述词时去掉的虚词
FILLER_WORDS = ["请", "帮我", "去", "到", "走到", "走去", "前往", "导航到", "找", "找到", "看看", "一下", "那个", "这个",
                "旁边", "附近", "边上", "跟前", "前面", "里面", "里的", "里", "中的", "在", "上的", "的", "一个", "个",
                "goto", "goto the", "the", "near", "to"]


class SceneGraph:
    def __init__(self, map_dir, graph, features):
        self.map_dir = Path(map_dir).resolve()
        self.info = {k: v for k, v in graph.items() if k not in ("nodes", "edges")}
        self.nodes = {n["id"]: n for n in graph["nodes"]}
        self.edges = graph["edges"]
        self.features = features
        self.adj = {}  # 节点编号 -> [(相邻节点, 边)]
        for e in self.edges:
            self.adj.setdefault(e["source"], []).append((e["target"], e))
            self.adj.setdefault(e["target"], []).append((e["source"], e))
        self._place_ids = [i for i, n in self.nodes.items() if n["type"] == "place"]
        self._place_xy = np.array([self.nodes[i]["position"][:2] for i in self._place_ids])
        self._grid = None
        self._text_cache = {}

    @classmethod
    def load(cls, map_dir):
        d = Path(map_dir) / "scene_graph"
        graph = json.loads((d / "scene_graph.json").read_text())
        feats = np.load(d / "object_features.npy") if (d / "object_features.npy").exists() else np.zeros((0, 768))
        return cls(map_dir, graph, feats)

    # ------------------------------------------------------------ 基本访问

    def of_type(self, node_type):
        return [n for n in self.nodes.values() if n["type"] == node_type]

    @property
    def regions(self):
        return self.of_type("region")

    @property
    def objects(self):
        return self.of_type("object")

    @property
    def waypoints(self):
        return self.of_type("waypoint")

    @property
    def places(self):
        return self.of_type("place")

    def node(self, node_id):
        return self.nodes[node_id]

    def neighbors(self, node_id, edge_type=None):
        """与节点相连的 (节点, 边)；edge_type 可选 traversable / in_region / near / at / next_to / on。"""
        return [(self.nodes[o], e) for o, e in self.adj.get(node_id, []) if edge_type in (None, e["type"])]

    @property
    def grid(self):
        """碰撞高度场上的导航栅格（nav_grid.NavGrid），与仿真中的碰撞体完全一致。"""
        if self._grid is None:
            sys.path.insert(0, str(HERE.parent))
            from nav_grid import NavGrid
            self._grid = NavGrid.load(self.map_dir / "mujoco" / "collision_hfield.npz")
        return self._grid

    def nearest_place(self, xy):
        return self._place_ids[int(np.linalg.norm(self._place_xy - np.asarray(xy)[:2], axis=1).argmin())]

    def region_at(self, xy):
        return self.nodes[self.nodes[self.nearest_place(xy)]["region"]]

    def objects_in(self, region_id, label=None):
        return [o for o in self.objects if o["region"] == region_id and label in (None, o["label"], o["name"])]

    def region_neighbors(self, region_id):
        """通行网络上与该区域直接相连的其他区域。"""
        out = set()
        for e in self.edges:
            if e["type"] == "traversable":
                a, b = self.nodes[e["source"]]["region"], self.nodes[e["target"]]["region"]
                if a != b and region_id in (a, b):
                    out.add(b if a == region_id else a)
        return sorted(out)

    # ------------------------------------------------------------ 文字查询

    def find(self, text, top_k=5, types=NODE_TYPES, near=None, semantic="auto"):
        """文字 → 候选节点列表 [{node, score, match}]，score 越大越匹配。

        near：当前位置 (x, y)，同分时优先近的；semantic：True 总用 CLIP，False 不用，auto 仅在需要时用。
        """
        text = text.strip()
        if text in self.nodes:
            return [{"node": self.nodes[text], "score": 1.0, "match": "id"}]
        q = normalize(text)

        def hits(n):
            names = [a for a in map(normalize, [n.get("name", "")] + list(n.get("aliases", []))) if a]
            h = max((len(a) for a in names if a in q), default=0)
            # 点位、区域的名称较长（“走廊 1（充电桩）”），查询是名称的一部分也算命中
            if n["type"] in ("waypoint", "region") and len(q) >= 2 and any(q in a for a in names):
                h = max(h, len(q))
            return h

        # 区域名互相包含时（“通道”与“北侧通道”）只认最长的那个
        region_len = {r["id"]: hits(r) for r in self.regions}
        top = max(region_len.values(), default=0)
        regions_hit = [r for r in self.regions if top and region_len[r["id"]] == top]
        region_ids = {r["id"] for r in regions_hit}
        results = []
        objs = self.objects if "object" in types else []
        obj_hits = {o["id"]: hits(o) for o in objs if hits(o)}
        if objs and regions_hit and (obj_hits or semantic is True):
            # 查询里同时有区域名：只在这些区域里找（区域里没有候选时不限定）
            inside = [o for o in objs if o["region"] in region_ids]
            if any(o["id"] in obj_hits for o in inside) or not obj_hits:
                objs = inside or objs
        descriptive = self._has_descriptors(q, objs, regions_hit)
        use_sem = semantic is True or (semantic == "auto" and (descriptive or len(obj_hits) > 1 or
                                                               (not obj_hits and not self._hit_named(q, types))))
        if obj_hits or (use_sem and objs and not self._hit_named(q, types)):
            cands = objs if (descriptive and use_sem) or not obj_hits else [o for o in objs if o["id"] in obj_hits]
            sims = self._semantic_scores(text, cands) if use_sem else np.zeros(len(cands))
            for o, sim in zip(cands, sims):
                kw = o["id"] in obj_hits
                # 有描述词时以 CLIP 相似度为主，关键词命中只加一点；没有描述词时只在命中的物体里用 CLIP 排序
                sc = (0.5 + sim + (KEYWORD_BONUS if kw else 0.0)) if use_sem else 0.8 + 0.001 * min(o["num_detections"], 50)
                if near is not None:
                    sc -= 0.001 * np.linalg.norm(np.array(o["position"][:2]) - near)
                how = ("关键词+" if kw else "") + ("语义（CLIP）" if use_sem else "") if (kw or use_sem) else ""
                results.append({"node": o, "score": round(float(sc), 4), "match": how.rstrip("+") or "关键词"})
        else:
            for t, base in (("waypoint", 0.95), ("region", 0.9)):
                if t in types:
                    for n in self.of_type(t):
                        h = hits(n)
                        if h:
                            results.append({"node": n, "score": base + 0.001 * h, "match": "关键词"})
        results.sort(key=lambda r: -r["score"])
        return results[:top_k]

    def _hit_named(self, q, types):
        """查询是否点名了某个点位或区域（且没有点名物体）。"""
        for n in self.nodes.values():
            if n["type"] in ("waypoint", "region") and n["type"] in types:
                names = [normalize(a) for a in [n.get("name", "")] + list(n.get("aliases", [])) if a]
                if any(a in q or (len(q) >= 2 and q in a) for a in names):
                    return True
        return False

    def _has_descriptors(self, q, objs, regions_hit):
        """去掉物体名、区域名和「去」「旁边」之类的虚词后，查询里是否还剩描述性的词（颜色、文字等）。"""
        rest = q
        names = sorted({normalize(a) for o in objs for a in [o["name"]] + list(o.get("aliases", []))} |
                       {normalize(a) for r in regions_hit for a in [r["name"]] + list(r.get("aliases", []))},
                       key=len, reverse=True)
        for a in names + sorted(FILLER_WORDS, key=len, reverse=True):
            if a:
                rest = rest.replace(a, "")
        return len(rest) >= 2

    def resolve(self, target, near=None):
        """节点编号、节点字典或文字 → 最匹配的一个节点；找不到时抛 KeyError。"""
        if isinstance(target, dict):
            return target
        res = self.find(target, top_k=1, near=near)
        if not res:
            raise KeyError(f"场景图中找不到与「{target}」匹配的节点")
        return res[0]["node"]

    def _semantic_scores(self, text, objs):
        if not objs or len(self.features) == 0:
            return np.zeros(len(objs))
        if text not in self._text_cache:
            self._text_cache[text] = encode_texts([text])[0]
        v = self._text_cache[text]
        return np.array([float(self.features[o["feature_row"]] @ v) for o in objs])

    # ------------------------------------------------------------ 导航目标与路线

    def goal_pose(self, target, near=None):
        """导航目标站位 {id, name, type, x, y, yaw, place, region}。

        物体：离物体表面最近、四周留有转身空间的站位，朝向物体；点位：点位库中的站位；
        区域：区域中心地点，朝向取区域内点位的朝向（没有则为 0）；地点：地点位置。
        """
        n = self.resolve(target, near=near)
        if n["type"] == "object":
            a = n["approach"]
            x, y, yaw, place = a["x"], a["y"], a["yaw"], a["place"]
        elif n["type"] == "waypoint":
            (x, y), yaw, place = n["position"][:2], n["yaw"], n["place"]
        elif n["type"] == "region":
            place = n["center_place"]
            x, y = self.nodes[place]["position"][:2]
            wps = [w for w in self.waypoints if w["region"] == n["id"]]
            yaw = wps[0]["yaw"] if wps else 0.0
        else:
            (x, y), yaw, place = n["position"][:2], 0.0, n["id"]
        region = self.nodes[place]["region"]
        return {"id": n["id"], "name": n.get("name", n["id"]), "type": n["type"], "x": float(x), "y": float(y),
                "yaw": float(yaw), "place": place, "region": region,
                "reachable": n.get("approach", {}).get("reachable", True)}

    def topo_path(self, start_xy, target):
        """沿通行网络从起点附近地点到目标地点的地点编号序列（Dijkstra 最短路）。
        供 VLN 逐段给出子目标，或用来生成「先到主通道，再进北侧通道」这样的路线描述。"""
        src = self.nearest_place(start_xy)
        dst = self.goal_pose(target)["place"]
        dist, prev = {src: 0.0}, {}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == dst:
                break
            if d > dist[u]:
                continue
            for v, e in self.adj.get(u, []):
                if e["type"] != "traversable":
                    continue
                nd = d + e["length"]
                if nd < dist.get(v, np.inf):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(heap, (nd, v))
        if dst not in dist:
            return None
        path = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]])
        return path[::-1]

    def route_description(self, start_xy, target):
        """把 topo_path 压缩成依次经过的区域名称。"""
        path = self.topo_path(start_xy, target)
        if path is None:
            return None
        names = []
        for p in path:
            r = self.nodes[self.nodes[p]["region"]]["name"]
            if not names or names[-1] != r:
                names.append(r)
        return names

    def metric_path(self, start_xy, target):
        """在碰撞高度场上用 A* 规划到目标站位的折线 [[x, y], ...]（与 waypoint_tour.py 用的是同一个规划器）。"""
        g = self.goal_pose(target)
        path = self.grid.plan(np.asarray(start_xy, float)[:2], np.array([g["x"], g["y"]]))
        return None if path is None else [[round(float(x), 3), round(float(y), 3)] for x, y in path]

    # ------------------------------------------------------------ 文字描述（给大语言模型）

    def describe(self, position=None, max_objects_per_region=20):
        lines = [f"地图 {self.info['map_id']}（版本 {self.info['map_version']}）。坐标单位米，"
                 f"x 轴朝东（建图时的前进方向），y 轴朝北，原点为充电桩。"]
        if position is not None:
            r = self.region_at(position)
            lines.append(f"机器人当前位于 ({position[0]:.2f}, {position[1]:.2f})，在「{r['name']}」。")
        for r in self.regions:
            nb = "、".join(self.nodes[i]["name"] for i in self.region_neighbors(r["id"])) or "无"
            lines.append(f"\n区域「{r['name']}」（{r['id']}）：可行走面积约 {r['area_m2']} m²，"
                         f"中心 ({r['position'][0]:.2f}, {r['position'][1]:.2f})，与 {nb} 相连。")
            for w in self.waypoints:
                if w["region"] == r["id"]:
                    lines.append(f"  - 点位 {w['id']}「{w['name']}」位于 ({w['position'][0]:.2f}, {w['position'][1]:.2f})")
            objs = sorted(self.objects_in(r["id"]), key=lambda o: -o["num_detections"])[:max_objects_per_region]
            for o in objs:
                p, s, a = o["position"], o["size"], o["approach"]
                rel = [f"{e['type']} {self.nodes[x]['name']}({x})" for x, e in self.adj.get(o["id"], [])
                       if e["type"] in ("next_to", "on")]
                extra = ("；可能移动" if o["dynamic"] else "") + ("" if a["reachable"] else "；走不到跟前")
                lines.append(f"  - 物体 {o['id']}「{o['name']}」中心 ({p[0]:.2f}, {p[1]:.2f})，离地 {o['bbox_min'][2]:.2f}~"
                             f"{o['bbox_max'][2]:.2f} m，尺寸约 {s[0]:.1f}×{s[1]:.1f} m；"
                             f"站位 ({a['x']:.2f}, {a['y']:.2f}) 离它 {a['distance_to_surface']:.1f} m{extra}"
                             + (f"；{'，'.join(rel)}" if rel else ""))
        return "\n".join(lines)

    def to_networkx(self):
        """转成 networkx.Graph（需要安装 networkx），节点属性即 scene_graph.json 中的字段。"""
        import networkx as nx
        g = nx.Graph()
        for i, n in self.nodes.items():
            g.add_node(i, **n)
        for e in self.edges:
            g.add_edge(e["source"], e["target"], **{k: v for k, v in e.items() if k not in ("source", "target")})
        return g


def normalize(s):
    return "".join(str(s).lower().split())


# ---------------------------------------------------------------- 文字向量（中文 CLIP）

_clip = None


def encode_texts(texts):
    """文字 → L2 归一化的中文 CLIP 文字向量 (N, 768)。没有 torch 时在子进程中计算。"""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        out = subprocess.run([TORCH_PYTHON, "-E", "-P", str(Path(__file__).resolve()), "encode-text", *texts],
                             capture_output=True, text=True, check=True)
        return np.array(json.loads(out.stdout.strip().splitlines()[-1]), dtype=np.float32)
    global _clip
    import torch
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor
    if _clip is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip = (ChineseCLIPModel.from_pretrained(CLIP_MODEL).to(device).eval(),
                 ChineseCLIPProcessor.from_pretrained(CLIP_MODEL), device)
    model, proc, device = _clip
    with torch.no_grad():
        f = model.get_text_features(**proc(text=list(texts), return_tensors="pt", padding=True).to(device))
        f = f if torch.is_tensor(f) else f.pooler_output
        return torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy()


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "encode-text":
    import logging
    logging.disable(logging.WARNING)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    print(json.dumps(encode_texts(sys.argv[2:]).round(6).tolist()))
