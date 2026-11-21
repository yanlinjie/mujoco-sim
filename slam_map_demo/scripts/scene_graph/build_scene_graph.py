"""为地图包构建三维场景图，写入 <地图包>/scene_graph/。

场景图分四层，全部使用地图坐标系（与点位库、碰撞高度场一致）：
- 区域 region：主通道、丁字路口等，由配置文件给名称和种子点；
- 地点 place：机器狗可站立的位置，从碰撞高度场的可通行区域自动提取（沿通道中线每约 0.5 m 一个），
  相邻且直线可通行的地点之间连「可通行」边，构成通行网络；
- 物体 object：开放词汇检测器（Grounding DINO）在建图照片里找出的货架、纸箱等，借助地图点云
  定出三维位置和包围盒，多张照片的检测合并为同一物体；每个物体存一个中文 CLIP 图像向量供文字检索；
- 点位 waypoint：waypoints.yaml 中的点位，挂到最近的地点上。

边的类型：traversable（地点-地点，可通行，带长度）、in_region（地点/物体/点位 → 区域）、
near（物体 → 地点，到达该物体时的站位所在地点）、at（点位 → 地点）、
next_to（物体-物体，水平距离小于 0.3 m）、on（物体 → 物体，放在其上）。

用法（需要 torch + transformers，本机基础 conda 环境；numpy 装在用户目录，故用 -E -P 而非 -I）：
    python -E -P build_scene_graph.py <地图包目录> <配置.yaml> <nerfstudio 数据目录> <检测模型目录> <CLIP 模型目录>
加 --places-only 则只构建区域/地点/点位三层（不需要 torch，几秒钟），物体层留空。
"""
import argparse
import heapq
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from map_to_mujoco import read_pcd_xyz  # noqa: E402
from nav_grid import NavGrid, ROBOT_RADIUS  # noqa: E402

PLACE_SPACING = 0.5    # 地点间距
EDGE_MAX = 0.85        # 两个地点相距不超过此值、且直线可通行时连边
STAND_CLEARANCE = 0.45  # 物体站位离障碍的最小距离（与点位库一致，留出原地转身空间）
STAND_MAX_DIST = 2.0   # 站位离物体表面超过此距离则认为该物体走不到跟前
NEXT_TO_DIST = 0.3
VOXEL = 0.05           # 物体体素边长
FLOOR_Z = 0.04         # 低于此高度的点视为地面，不归入物体
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


# ---------------------------------------------------------------- 地点与区域

def build_places(grid):
    """在可通行且与原点连通的区域内取地点：先沿离障碍距离的“山脊”（通道中线）取，再补空隙。"""
    free = grid.clearance > ROBOT_RADIUS
    labels, _ = ndimage.label(free)
    reach = labels == labels[grid.to_cell((0.0, 0.0))]
    c = np.where(reach, grid.clearance, 0.0)
    ridge = reach & (c >= ndimage.maximum_filter(c, size=3) - 1e-6)

    chosen = [np.zeros(2)]  # 原点（出生点）固定为一个地点

    def pick(mask):
        rows, cols = np.nonzero(mask)
        order = np.argsort(-c[rows, cols])
        for r, col in zip(rows[order], cols[order]):
            xy = grid.to_xy((r, col))
            if min(np.linalg.norm(np.array(chosen) - xy, axis=1)) >= PLACE_SPACING:
                chosen.append(xy)

    pick(ridge)
    # 补空隙：可通行区域中离已有地点超过 1.2 倍间距的栅格
    rows, cols = np.nonzero(reach)
    cells_xy = grid.lo + np.stack([cols, rows], 1) * grid.res
    d, _ = cKDTree(np.array(chosen)).query(cells_xy)
    gap = np.zeros_like(reach)
    gap[rows[d > 1.2 * PLACE_SPACING], cols[d > 1.2 * PLACE_SPACING]] = True
    pick(gap)
    return np.array(chosen), reach


def build_place_edges(grid, xy):
    edges = {}
    for i, j in cKDTree(xy).query_pairs(EDGE_MAX):
        if grid.segment_free(xy[i], xy[j], ROBOT_RADIUS):
            edges[(i, j)] = float(np.linalg.norm(xy[i] - xy[j]))
    # 保证连通：把每个孤立分量用 A* 路线接到主分量上
    while True:
        comp = components(len(xy), edges)
        if comp.max() == 0:
            return edges
        main, other = np.flatnonzero(comp == 0), np.flatnonzero(comp != 0)
        d = np.linalg.norm(xy[main][:, None] - xy[other][None], axis=2)
        a, b = np.unravel_index(d.argmin(), d.shape)
        i, j = main[a], other[b]
        path = grid.plan(xy[i], xy[j])
        length = sum(np.linalg.norm(q - p) for p, q in zip(path[:-1], path[1:])) if path else d.min()
        edges[(min(i, j), max(i, j))] = float(length)


def components(n, edges):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j in edges:
        parent[find(i)] = find(j)
    roots = [find(i) for i in range(n)]
    root0 = roots[0]
    uniq = {root0: 0}
    for r in roots:
        uniq.setdefault(r, len(uniq))
    return np.array([uniq[r] for r in roots])


def dijkstra_multi(n, edges, sources):
    """多源最短路：返回每个节点最近的源编号。"""
    adj = [[] for _ in range(n)]
    for (i, j), w in edges.items():
        adj[i].append((j, w))
        adj[j].append((i, w))
    dist = np.full(n, np.inf)
    owner = np.full(n, -1)
    heap = []
    for k, s in enumerate(sources):
        dist[s], owner[s] = 0.0, k
        heap.append((0.0, s))
    heapq.heapify(heap)
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v], owner[v] = d + w, owner[u]
                heapq.heappush(heap, (d + w, v))
    return owner


# ---------------------------------------------------------------- 物体

def quadratic(c, x, y):
    return c[0] + c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y


class Lifter:
    """把图像中的检测框借助地图点云抬升为三维点集。"""

    def __init__(self, map_dir, intr, downsample=4):
        self.p_map = read_pcd_xyz(map_dir / "dense_map.pcd").astype(np.float64)
        # 地图坐标 → 重建坐标（相机位姿所在的坐标系）：先撤销地面展平，再乘 map_from_slam 的逆
        T = np.loadtxt(map_dir / "map_from_slam.txt")
        coef = np.loadtxt(map_dir / "floor_flatten.txt")
        p = self.p_map.copy()
        p[:, 2] += quadratic(coef, p[:, 0], p[:, 1])
        self.p_rec = np.c_[p, np.ones(len(p))] @ np.linalg.inv(T).T
        self.fx, self.fy, self.cx, self.cy = intr["fl_x"], intr["fl_y"], intr["cx"], intr["cy"]
        self.w, self.h, self.ds = intr["w"], intr["h"], downsample

    def project(self, c2w_gl):
        """返回所有地图点的像素坐标、深度，以及是否可见（未被更近的表面遮挡）。"""
        w2c = GL_TO_CV @ np.linalg.inv(c2w_gl)
        pc = self.p_rec @ w2c.T
        z = pc[:, 2]
        front = z > 0.1
        u = np.full(len(z), -1.0)
        v = np.full(len(z), -1.0)
        u[front] = self.fx * pc[front, 0] / z[front] + self.cx
        v[front] = self.fy * pc[front, 1] / z[front] + self.cy
        inside = front & (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        W, H = self.w // self.ds, self.h // self.ds
        pix = np.full(len(z), -1)
        pix[inside] = (v[inside] // self.ds).astype(int) * W + (u[inside] // self.ds).astype(int)
        zbuf = np.full(W * H, np.inf)
        np.minimum.at(zbuf, pix[inside], z[inside])
        # 点云在远处较稀疏，最小值滤波把前景表面向周围扩一个像素，避免背景点从缝隙“透”出来
        zbuf = ndimage.minimum_filter(zbuf.reshape(H, W), size=3).ravel()
        visible = inside.copy()
        visible[inside] = z[inside] <= zbuf[pix[inside]] + 0.05 + 0.02 * z[inside]
        return u, v, z, visible

    def lift_box(self, proj, box):
        """框内可见、离地的点中，取包含框中心区域点最多的那个空间连通块。返回地图点序号。"""
        u, v, z, visible = proj
        x0, y0, x1, y1 = box
        sel = visible & (u >= x0) & (u < x1) & (v >= y0) & (v < y1) & (self.p_map[:, 2] > FLOOR_Z)
        idx = np.flatnonzero(sel)
        if len(idx) < 30:
            return None
        mx, my = (x1 - x0) * 0.25, (y1 - y0) * 0.25
        central = (u[idx] >= x0 + mx) & (u[idx] < x1 - mx) & (v[idx] >= y0 + my) & (v[idx] < y1 - my)
        if central.sum() < 10:
            return None
        vox = np.floor(self.p_map[idx] / 0.04).astype(int)
        vox -= vox.min(0)
        occ = np.zeros(vox.max(0) + 1, bool)
        occ[tuple(vox.T)] = True
        lab, _ = ndimage.label(occ, structure=np.ones((3, 3, 3)))
        pl = lab[tuple(vox.T)]
        best = np.bincount(pl[central]).argmax()
        keep = idx[pl == best]
        return keep if len(keep) >= 30 else None


def voxel_keys(points):
    k = np.floor(points / VOXEL).astype(np.int64) + 10000
    return set((k[:, 0] * 100000 + k[:, 1]) * 100000 + k[:, 2])


def keys_to_points(keys):
    k = np.array(sorted(keys), dtype=np.int64)
    z = k % 100000
    y = (k // 100000) % 100000
    x = k // 10000000000
    return (np.stack([x, y, z], 1) - 10000 + 0.5) * VOXEL


def match_label(text, vocab):
    """检测器输出的类别文字（可能只是 prompt 的一部分）对应到词表条目。"""
    text = text.strip().lower()
    best, score = None, 0
    for v in vocab:
        words = set(v["prompt"].split())
        s = len(words & set(text.split()))
        if text == v["prompt"]:
            s += 10
        if s > score:
            best, score = v, s
    return best


def detect_objects(map_dir, data_dir, cfg, det_model_dir, clip_model_dir, out_dir):
    """返回 (物体列表, CLIP 图像向量)。每个物体含 label、keys（体素集合）、count、views、label_scores、crop。"""
    import torch
    from PIL import Image
    from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, ChineseCLIPModel,
                              ChineseCLIPProcessor)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = [v for v in cfg["vocabulary"]]
    dcfg = cfg["detection"]
    tf = json.load(open(data_dir / "transforms.json"))
    frames = [f for f in tf["frames"] if f.get("original_name", "").endswith("_L.jpg")]
    frames.sort(key=lambda f: f["original_name"])
    frames = frames[::dcfg["frame_step"]]
    lifter = Lifter(map_dir, tf)
    print(f"物体检测：{len(frames)} 张左目图像，地图点 {len(lifter.p_map)} 个，设备 {device}")

    proc = AutoProcessor.from_pretrained(det_model_dir)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(det_model_dir).to(device).eval()
    prompt = " . ".join(v["prompt"] for v in vocab) + " ."

    detections, n_person = [], 0
    t0 = time.time()
    for fi, fr in enumerate(frames):
        img = Image.open(data_dir / fr["file_path"]).convert("RGB")
        # 行人掩码（make_person_masks.py 生成，0 为行人）：与行人重叠的检测多半是人手里拿的东西，不是固定物体
        person = None
        if fr.get("mask_path") and (data_dir / fr["mask_path"]).exists():
            person = np.array(Image.open(data_dir / fr["mask_path"])) == 0
        with torch.no_grad():
            inp = proc(images=img, text=prompt, return_tensors="pt").to(device)
            out = model(**inp)
        res = proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=dcfg["box_threshold"], text_threshold=dcfg["text_threshold"],
            target_sizes=[img.size[::-1]])[0]
        labels = res.get("text_labels", res.get("labels"))
        proj = None
        for box, score, text in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), labels):
            v = match_label(text, vocab)
            if v is None or v.get("exclude"):
                continue
            # 占满大半个画面的框通常是把整面货架/墙当成一个物体，三维抬升不可靠
            if box_area(box) > 0.6 * img.size[0] * img.size[1]:
                continue
            if person is not None:
                x0, y0, x1, y1 = np.round(box).astype(int)
                if person[max(y0, 0):y1, max(x0, 0):x1].mean() > dcfg.get("max_person_overlap", 0.15):
                    n_person += 1
                    continue
            if proj is None:
                proj = lifter.project(np.array(fr["transform_matrix"]))
            keep = lifter.lift_box(proj, box)
            if keep is None:
                continue
            detections.append({"label": v["prompt"], "score": float(score), "frame": fr["file_path"],
                               "box": [float(b) for b in box], "keys": voxel_keys(lifter.p_map[keep])})
        if fi % 20 == 0:
            print(f"  {fi + 1}/{len(frames)} 张，累计检测 {len(detections)} 个，{time.time() - t0:.0f} s")
    del model
    torch.cuda.empty_cache()
    print(f"检测 {len(detections)} 次（另有 {n_person} 次与行人重叠，已丢弃）")

    # 合并：体素重叠足够多的检测视为同一物体。同类别阈值较低；不同类别（检测器对同一物体给出不同名字）
    # 要求重叠更多。类别按各次检测的置信度投票。做三遍，避免合并顺序造成的拆分。
    objs = []
    for det in sorted(detections, key=lambda d: -d["score"]):
        objs = merge_into(objs, new_object(det), dcfg)
    for _ in range(2):
        merged = []
        for o in sorted(objs, key=lambda o: -o["count"]):
            merged = merge_into(merged, o, dcfg)
        objs = merged
    objs = [o for o in objs if o["count"] >= dcfg["min_detections"] and len(o["views"]) >= dcfg["min_detections"]]
    # 只保留被多次检测命中的体素，去掉个别检测框带进来的背景
    for o in objs:
        need = max(1, int(np.ceil(dcfg.get("min_voxel_share", 0.2) * o["count"])))
        core = {k for k, n in o["hits"].items() if n >= need}
        o["keys"] = core if len(core) >= 8 else set(o["hits"])
    print(f"合并为 {len(objs)} 个物体（至少在 {dcfg['min_detections']} 张图中被检测到）")

    # 中文 CLIP：每个物体取最好的 3 次检测裁剪图，图像向量取平均；再与各类别名称做零样本分类，
    # 与检测器投票各占一半决定最终类别（检测器常把白色小机器人叫成 cart 之类）
    clip = ChineseCLIPModel.from_pretrained(clip_model_dir).to(device).eval()
    cproc = ChineseCLIPProcessor.from_pretrained(clip_model_dir)

    def feats(out):
        return out if torch.is_tensor(out) else out.pooler_output

    @torch.no_grad()
    def embed_images(images):
        f = feats(clip.get_image_features(**cproc(images=images, return_tensors="pt").to(device)))
        return torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy()

    with torch.no_grad():
        t = cproc(text=[f"一张{v['name']}的照片" for v in vocab], return_tensors="pt", padding=True).to(device)
        text_feat = torch.nn.functional.normalize(feats(clip.get_text_features(**t)), dim=-1).float().cpu().numpy()

    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    for old in (out_dir / "crops").glob("*.jpg"):
        old.unlink()
    img_feat = np.zeros((len(objs), text_feat.shape[1]), np.float32)
    prompts = [v["prompt"] for v in vocab]
    for i, o in enumerate(objs):
        crops = []
        for d in o["dets"]:
            img = Image.open(data_dir / d["frame"]).convert("RGB")
            x0, y0, x1, y1 = d["box"]
            px, py = 0.1 * (x1 - x0), 0.1 * (y1 - y0)
            crop = img.crop((max(0, x0 - px), max(0, y0 - py), min(img.width, x1 + px), min(img.height, y1 + py)))
            crop.thumbnail((448, 448))
            crops.append(crop)
        f = embed_images(crops)
        img_feat[i] = f.mean(0) / np.linalg.norm(f.mean(0))
        logits = 100.0 * f @ text_feat.T
        clip_prob = np.exp(logits - logits.max(1, keepdims=True))
        clip_prob = (clip_prob / clip_prob.sum(1, keepdims=True)).mean(0)
        votes = np.array([o["votes"].get(pr, 0.0) for pr in prompts])
        votes = votes / votes.sum()
        excluded = np.array([bool(v.get("exclude")) for v in vocab])
        combined = np.where(excluded, 0.0, 0.5 * votes + 0.5 * clip_prob)
        o["label"] = prompts[int(combined.argmax())]
        o["label_scores"] = {prompts[k]: round(float(combined[k]), 3) for k in np.argsort(-combined)[:3]}
        o["crop"] = f"crops/obj_{i + 1:02d}.jpg"
        crops[0].save(out_dir / o["crop"], quality=90)
    return objs, img_feat


def new_object(det):
    return {"votes": {det["label"]: det["score"]}, "keys": set(det["keys"]),
            "hits": dict.fromkeys(det["keys"], 1), "count": 1, "views": {det["frame"]}, "dets": [det]}


def top_label(o):
    return max(o["votes"], key=o["votes"].get)


def merge_into(objs, obj, dcfg):
    for o in objs:
        inter = len(o["keys"] & obj["keys"])
        if top_label(o) == top_label(obj):
            # 同类别：相对较小者的重叠比例（同一物体从不同角度只看到一部分）
            merge = inter / max(1, min(len(o["keys"]), len(obj["keys"]))) > dcfg["merge_overlap"]
        else:
            # 不同类别：要求交并比够大，即大小相近且基本重合，避免小物体（如垃圾桶）被旁边的大纸箱吞掉
            merge = inter / max(1, len(o["keys"] | obj["keys"])) > dcfg.get("merge_iou_cross_label", 0.4)
        if merge:
            o["keys"] |= obj["keys"]
            for k, n in obj["hits"].items():
                o["hits"][k] = o["hits"].get(k, 0) + n
            for lab, sc in obj["votes"].items():
                o["votes"][lab] = o["votes"].get(lab, 0.0) + sc
            o["count"] += obj["count"]
            o["views"] |= obj["views"]
            o["dets"] = sorted(o["dets"] + obj["dets"], key=lambda d: -d["score"] * np.sqrt(box_area(d["box"])))[:3]
            return objs
    objs.append(obj)
    return objs


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def stand_pose_for(grid, reach, pts):
    """离物体表面最近、且离障碍不少于 STAND_CLEARANCE 的可达栅格作为站位，朝向对准物体最近点。"""
    ok = reach & (grid.clearance >= STAND_CLEARANCE)
    if not ok.any():
        ok = reach
    rows, cols = np.nonzero(ok)
    cells = grid.lo + np.stack([cols, rows], 1) * grid.res
    d, nn = cKDTree(pts[:, :2]).query(cells)
    k = d.argmin()
    xy, target = cells[k], pts[nn[k], :2]
    yaw = float(np.arctan2(*(target - xy)[::-1]))
    return xy, yaw, float(d[k])


# ---------------------------------------------------------------- 组装

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_dir", type=Path)
    ap.add_argument("config", type=Path)
    ap.add_argument("data_dir", type=Path, nargs="?")
    ap.add_argument("det_model", type=Path, nargs="?")
    ap.add_argument("clip_model", type=Path, nargs="?")
    ap.add_argument("--places-only", action="store_true")
    args = ap.parse_args()
    map_dir = args.map_dir.resolve()
    cfg = yaml.safe_load(open(args.config))
    meta = yaml.safe_load(open(map_dir / "meta.yaml"))
    out_dir = map_dir / "scene_graph"
    out_dir.mkdir(exist_ok=True)
    grid = NavGrid.load(map_dir / "mujoco" / "collision_hfield.npz")

    # 地点与通行网络
    pxy, reach = build_places(grid)
    pedges = build_place_edges(grid, pxy)
    print(f"地点 {len(pxy)} 个，可通行边 {len(pedges)} 条，可达面积 {reach.sum() * grid.res ** 2:.1f} m²")

    # 区域：每个地点归到沿通行网络最近的种子
    seeds, seed_region = [], []
    for k, r in enumerate(cfg["regions"]):
        for sxy in r["seeds"]:
            seeds.append(int(np.linalg.norm(pxy - sxy, axis=1).argmin()))
            seed_region.append(k)
    owner = np.array(seed_region)[dijkstra_multi(len(pxy), pedges, seeds)]
    cell_rows, cell_cols = np.nonzero(reach)
    cell_xy = grid.lo + np.stack([cell_cols, cell_rows], 1) * grid.res
    cell_owner = owner[cKDTree(pxy).query(cell_xy)[1]]

    nodes, edges = [], []
    for k, r in enumerate(cfg["regions"]):
        members = np.flatnonzero(owner == k)
        center = pxy[members].mean(0)
        center_place = members[np.linalg.norm(pxy[members] - center, axis=1).argmin()]
        nodes.append({"id": r["id"], "type": "region", "name": r["name"], "aliases": r.get("aliases", []),
                      "center_place": f"place_{center_place:03d}",
                      "position": [round(float(x), 3) for x in pxy[center_place]] + [0.0],
                      "area_m2": round(float((cell_owner == k).sum() * grid.res ** 2), 2)})
    for i, xy in enumerate(pxy):
        nodes.append({"id": f"place_{i:03d}", "type": "place", "region": cfg["regions"][owner[i]]["id"],
                      "position": [round(float(xy[0]), 3), round(float(xy[1]), 3), 0.0],
                      "clearance": round(float(grid.clearance_at(xy)), 3)})
        edges.append({"source": f"place_{i:03d}", "target": cfg["regions"][owner[i]]["id"], "type": "in_region"})
    for (i, j), w in sorted(pedges.items()):
        edges.append({"source": f"place_{i:03d}", "target": f"place_{j:03d}", "type": "traversable",
                      "length": round(w, 3)})

    def nearest_place(xy):
        return int(np.linalg.norm(pxy - xy, axis=1).argmin())

    # 点位库
    wps = yaml.safe_load(open(map_dir / "waypoints.yaml"))["waypoints"]
    for wp in wps:
        p = wp["stand_pose"]
        k = nearest_place(np.array([p["x"], p["y"]]))
        nodes.append({"id": wp["id"], "type": "waypoint", "name": wp["name"],
                      "aliases": [wp["name"], wp["id"]] + cfg.get("waypoint_aliases", {}).get(wp["id"], []),
                      "region": cfg["regions"][owner[k]]["id"],
                      "position": [p["x"], p["y"], 0.0], "yaw": p["yaw"], "place": f"place_{k:03d}"})
        edges.append({"source": wp["id"], "target": f"place_{k:03d}", "type": "at"})
        edges.append({"source": wp["id"], "target": cfg["regions"][owner[k]]["id"], "type": "in_region"})

    # 物体
    feats = np.zeros((0, 768), np.float32)
    n_obj = 0
    if not args.places_only:
        objs, feats = detect_objects(map_dir, args.data_dir, cfg, args.det_model, args.clip_model, out_dir)
        vocab = {v["prompt"]: v for v in cfg["vocabulary"]}
        boxes = []
        for i, o in enumerate(objs):
            v = vocab[o["label"]]
            pts = keys_to_points(o["keys"])
            lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
            lo[2] = min(lo[2], pts[:, 2].min())
            boxes.append((lo, hi, pts))
            sxy, syaw, sdist = stand_pose_for(grid, reach, pts)
            k = nearest_place(sxy)
            region = cfg["regions"][owner[k]]["id"]
            oid = f"obj_{i + 1:02d}"
            nodes.append({
                "id": oid, "type": "object", "label": v["prompt"], "name": v["name"],
                "aliases": v.get("aliases", []), "dynamic": bool(v.get("dynamic", False)),
                "position": [round(float(x), 3) for x in (lo + hi) / 2],
                "bbox_min": [round(float(x), 3) for x in lo], "bbox_max": [round(float(x), 3) for x in hi],
                "size": [round(float(x), 3) for x in hi - lo],
                "num_detections": int(o["count"]), "num_views": len(o["views"]),
                "label_scores": o["label_scores"], "image": o["crop"], "feature_row": i,
                "region": region,
                "approach": {"x": round(float(sxy[0]), 3), "y": round(float(sxy[1]), 3), "yaw": round(syaw, 3),
                             "place": f"place_{k:03d}", "distance_to_surface": round(sdist, 3),
                             "reachable": bool(sdist <= STAND_MAX_DIST)},
            })
            edges.append({"source": oid, "target": region, "type": "in_region"})
            edges.append({"source": oid, "target": f"place_{k:03d}", "type": "near",
                          "distance": round(sdist, 3)})
        # 物体之间的关系
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                (lo_a, hi_a, pa), (lo_b, hi_b, pb) = boxes[i], boxes[j]
                if np.any(lo_a[:2] > hi_b[:2] + 1) or np.any(lo_b[:2] > hi_a[:2] + 1):
                    continue
                d = cKDTree(pa[:, :2]).query(pb[:, :2])[0].min()
                if d < NEXT_TO_DIST:
                    edges.append({"source": f"obj_{i + 1:02d}", "target": f"obj_{j + 1:02d}", "type": "next_to",
                                  "distance": round(float(d), 3)})
                for (a, la, ha), (b, lb, hb) in (((i, lo_a, hi_a), (j, lo_b, hi_b)), ((j, lo_b, hi_b), (i, lo_a, hi_a))):
                    ov = np.clip(np.minimum(ha[:2], hb[:2]) - np.maximum(la[:2], lb[:2]), 0, None).prod()
                    area = max(1e-6, np.prod(ha[:2] - la[:2]))
                    if ov / area > 0.5 and abs(la[2] - hb[2]) < 0.15 and la[2] > 0.1:
                        edges.append({"source": f"obj_{a + 1:02d}", "target": f"obj_{b + 1:02d}", "type": "on"})
        n_obj = len(objs)

    graph = {
        "map_id": meta["map_id"], "map_version": meta["version"], "frame": "map",
        "units": "meter, radian", "built": time.strftime("%Y-%m-%d %H:%M"),
        "config": str(args.config.resolve()),
        "robot_radius": ROBOT_RADIUS, "place_spacing": PLACE_SPACING,
        "object_feature_model": "OFA-Sys/chinese-clip-vit-large-patch14" if n_obj else None,
        "nodes": nodes, "edges": edges,
    }
    (out_dir / "scene_graph.json").write_text(json.dumps(graph, ensure_ascii=False, indent=1))
    np.save(out_dir / "object_features.npy", feats)
    counts = {t: sum(n["type"] == t for n in nodes) for t in ("region", "place", "object", "waypoint")}
    print(f"场景图已写入 {out_dir}：{counts}，边 {len(edges)} 条")
    draw_overview(grid, reach, graph, out_dir / "overview.png")


def draw_overview(grid, reach, graph, path):
    from PIL import Image, ImageDraw, ImageFont
    s = 8
    H, W = grid.heights.shape
    img = np.full((H, W, 3), 70, np.uint8)
    img[grid.clearance > 0] = (225, 225, 225)
    nodes = {n["id"]: n for n in graph["nodes"]}
    colors = [(170, 210, 250), (250, 210, 160), (190, 240, 190), (240, 190, 230), (230, 230, 150)]
    regions = [n for n in graph["nodes"] if n["type"] == "region"]
    places = [n for n in graph["nodes"] if n["type"] == "place"]
    pxy = np.array([n["position"][:2] for n in places])
    rows, cols = np.nonzero(reach)
    owner = cKDTree(pxy).query(grid.lo + np.stack([cols, rows], 1) * grid.res)[1]
    ridx = {r["id"]: k for k, r in enumerate(regions)}
    for k, (r, c) in enumerate(zip(rows, cols)):
        img[r, c] = colors[ridx[places[owner[k]]["region"]] % len(colors)]
    im = Image.fromarray(np.flipud(img)).resize((W * s, H * s), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 14)
    except OSError:
        font = ImageFont.load_default()

    def px(xy):
        c, r = (np.asarray(xy[:2]) - grid.lo) / grid.res
        return float(c * s), float((H - r) * s)

    for e in graph["edges"]:
        if e["type"] == "traversable":
            dr.line([px(nodes[e["source"]]["position"]), px(nodes[e["target"]]["position"])], fill=(90, 90, 200), width=2)
    for n in places:
        x, y = px(n["position"])
        dr.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(40, 40, 160))
    for n in graph["nodes"]:
        if n["type"] == "object":
            (x0, y1), (x1, y0) = px(n["bbox_min"]), px(n["bbox_max"])
            col = (200, 120, 0) if n["dynamic"] else (200, 30, 30)
            dr.rectangle([x0, y0, x1, y1], outline=col, width=2)
            dr.text((x0 + 2, y0 + 1), f"{n['id'][-2:]} {n['name']}", fill=col, font=font)
            a = n["approach"]
            if a["reachable"]:
                dr.line([px((a["x"], a["y"])), px(n["position"])], fill=(200, 30, 30), width=1)
        elif n["type"] == "waypoint":
            x, y = px(n["position"])
            dr.rectangle([x - 5, y - 5, x + 5, y + 5], fill=(0, 150, 0))
            dr.text((x + 6, y + 2), n["id"][-2:], fill=(0, 110, 0), font=font)
    for r in regions:
        x, y = px(r["position"])
        dr.text((x - 20, y - 26), r["name"], fill=(0, 0, 0), font=font)
    im.save(path)
    print(f"示意图已保存到 {path}")


if __name__ == "__main__":
    main()
