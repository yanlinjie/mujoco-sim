"""在已有地图中做激光实时定位：每来一帧激光，配准到地图点云上，得到机器狗在地图坐标系中的位姿。

只依赖 numpy + scipy（不需要 Open3D / ROS），可在 unitree-mujoco 环境中与 DDS 订阅一起运行。

输入（与 Go2 话题一一对应）：
- 激光帧 /utlidar/cloud：雷达坐标系的点、每点相对帧头的时间（秒）、帧头时间戳；
- 腿式里程计 /utlidar/robot_odom：Go2 运控输出的机身位姿（机身 IMU + 腿部运动学，横滚/俯仰以重力为参考）。
  只用它的“相对运动”做预测和帧内去畸变，它自身的累计漂移由激光配准消除。

每帧流程：
1. 预处理：去掉无效点、0.25 m 以内的点、机身自遮挡点（雷达系中固定的体素，见 config/go2_l1/self_voxels.npz）；
2. 去畸变：按腿式里程计插值出每个点时刻的雷达位姿，把点统一到帧结束时刻；
3. 预测：上一帧地图位姿 × 腿式里程计在两帧之间的相对运动；
4. 配准：点到面 ICP（Tukey 鲁棒核，先 0.5 m 再 0.15 m 匹配距离），目标是预先算好法向的地图点云（KD 树只建一次）；
5. 校验：结构内点比例（离地 0.15 m 以上的点中离地图 0.1 m 以内的比例；L1 倒装、大部分点打在地面上，
   地面点在哪儿都对得上，不能用来判断对错）过低、修正量过大时不采纳，沿用预测；连续 lost_frames 帧不可信判为丢失；
6. 融合：只采纳 alpha 比例的修正量，抑制单帧稀疏点云配准的抖动；横滚/俯仰校正为腿式里程计给出的重力方向。

初始化与重定位：积累最近 1.5 s 的帧（用腿式里程计拼成一片），按重力摆正后，
在三维栅格上穷举位置（0.2 m 步长）和朝向（5° 步长）打分（命中结构的似然 − 射线穿过地图实心体素的比例），
取 20 个互相分开的候选做 ICP，综合得分 = 结构内点比例 − 看穿比例（射线终点能对上地图、途中却穿过地图实心处的比例）。
只有最佳得分够高、且明显高于收敛到别处的次佳候选时才采纳；否则继续积累数据（窗口最长 8 s）每秒重试，
宁可多等，也尽量不给出错误位姿（但不能完全避免，见 README 中的测试结果）。
- 有初始位姿提示（set_initial_pose）：只在提示附近（默认 1 m、±30°）搜索，ICP 滑出该范围的结果丢弃；
  看穿比例只占 0.3 权重（地图与现场有差异时，如门开关，看穿判据会误伤正确位姿）。推荐的启动方式；
- 无提示（全图搜索）：要求连续两次搜索结果一致（按里程计推算对齐）才采纳。机器狗静止、只看到眼前一面墙时
  无法唯一确定位置，会一直等待，需要走动到开阔处；
- 丢失后：先在最后可信位置 3 m 内（按提示方式）搜索，失败再全图搜索。

坐标：地图坐标系即地图包 dense_map.pcd 的坐标系（重力对齐，地面 z=0）。输出雷达位姿与机身位姿，
机身位姿 = 雷达位姿 × 雷达到机身外参的逆（T_base_lidar.txt：机身坐标 = T @ 雷达坐标）。
"""
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

CACHE_VERSION = 1


@dataclass
class Params:
    map_vox: float = 0.05            # 地图下采样体素（米）
    normal_k: int = 20               # 估计法向的近邻数
    scan_vox: float = 0.05           # 单帧下采样体素
    min_range: float = 0.25
    max_range: float = 20.0
    icp_stages: tuple = ((0.5, 0.2, 8), (0.15, 0.08, 10))   # (匹配距离, Tukey 参数, 迭代次数)
    inlier_dist: float = 0.10        # 内点判据
    min_fitness: float = 0.45        # 内点比例低于此值不采纳配准
    max_corr_trans: float = 0.30     # 单帧修正量上限（米）
    max_corr_rot: float = 5.0        # 单帧修正量上限（度）
    alpha: float = 0.5               # 修正量采纳比例
    lost_frames: int = 15            # 连续多少帧不可信判为丢失（约 1 s）
    reloc_window: float = 1.5        # 重定位首次积累的时长（秒）
    reloc_max_window: float = 8.0    # 判定有歧义时继续积累，最长到此时长（之后滑动）
    reloc_retry: float = 1.0         # 有歧义时每隔多久重试（秒）
    reloc_min_frames: int = 10
    reloc_min_score: float = 0.5     # 综合得分（结构内点比例 − 看穿比例）下限
    reloc_min_margin: float = 0.25   # 最佳与“收敛到别处”的次佳候选的综合得分之差下限
    hint_see_weight: float = 0.3     # 有提示（限定范围）搜索时，看穿比例在综合得分中的权重（全图搜索为 1）
    hint_min_score: float = 0.6
    hint_min_margin: float = 0.1
    global_confirm: int = 2          # 全图搜索（无提示）时，需连续几次搜索结果一致才采纳
    lidar_height: float = 0.29       # 雷达离地高度（米），重定位时的初值，ICP 会修正
    band: tuple = (0.25, 1.5)        # 二维搜索使用的高度带（离地，米）
    grid_res: float = 0.1            # 二维距离场分辨率
    search_step: float = 0.2         # 位置搜索步长
    yaw_step_deg: float = 5.0
    n_candidates: int = 20
    keep_odom: float = 10.0          # 腿式里程计缓存时长（秒）


@dataclass
class Result:
    stamp: float                     # 帧结束时刻（与腿式里程计同一时钟）
    status: str                      # TRACKING / RELOCALIZED / PREDICT（本帧配准未采纳，仅预测）/ LOST / INIT
    T_map_lidar: np.ndarray = None
    T_map_base: np.ndarray = None
    fitness: float = 0.0
    n_points: int = 0
    ms: float = 0.0                  # 本帧处理耗时（毫秒）
    info: dict = field(default_factory=dict)

    def xy_yaw(self):
        """机身中心在地图中的 (x, y, yaw)；yaw 为机身 x 轴在水平面上的朝向。"""
        T = self.T_map_base
        return float(T[0, 3]), float(T[1, 3]), float(np.arctan2(T[1, 0], T[0, 0]))

    def to_dict(self):
        d = {"stamp": self.stamp, "status": self.status, "fitness": round(self.fitness, 3),
             "n_points": self.n_points, "ms": round(self.ms, 1)}
        if self.T_map_base is not None:
            x, y, yaw = self.xy_yaw()
            d.update(x=round(x, 4), y=round(y, 4), z=round(float(self.T_map_base[2, 3]), 4), yaw=round(yaw, 4))
            q = R.from_matrix(self.T_map_base[:3, :3]).as_quat()
            d["quat_xyzw"] = [round(float(v), 6) for v in q]
        d.update(self.info)
        return d


# ---------------------------------------------------------------- 工具函数
def read_pcd_xyz(path):
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if not line or line.startswith("#"):
                continue
            k, _, v = line.partition(" ")
            header[k] = v
            if k == "DATA":
                break
        buf = f.read()
    if header["DATA"] != "binary":
        raise ValueError(f"只支持 binary PCD：{path}")
    fields, sizes, n = header["FIELDS"].split(), [int(s) for s in header["SIZE"].split()], int(header["POINTS"])
    if any(s != 4 for s in sizes):
        raise ValueError("只支持 4 字节字段")
    arr = np.frombuffer(buf[: n * 4 * len(fields)], np.float32).reshape(n, len(fields))
    return arr[:, [fields.index("x"), fields.index("y"), fields.index("z")]].astype(np.float64)


def voxel_down(x, vox):
    """每个体素取点的均值。"""
    k = np.floor(x / vox).astype(np.int64)
    _, inv, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out = np.zeros((len(cnt), 3))
    np.add.at(out, inv, x)
    return out / cnt[:, None]


def estimate_normals(x, k):
    tree = cKDTree(x)
    _, idx = tree.query(x, k=k, workers=-1)
    nb = x[idx]                                         # (N, k, 3)
    c = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / k
    w, v = np.linalg.eigh(cov)
    n = v[:, :, 0]
    planarity = (w[:, 1] - w[:, 0]) / np.maximum(w[:, 2], 1e-12)
    return n, planarity


def make_T(Rm, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rm, t
    return T


def rot_between(a, b):
    """把单位向量 a 转到 b 的最小旋转。"""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    ax = np.cross(a, b)
    s = np.linalg.norm(ax)
    if s < 1e-12:
        return R.identity()
    return R.from_rotvec(ax / s * np.arctan2(s, a @ b))


def yaw_of(Rm):
    return np.arctan2(Rm[1, 0], Rm[0, 0])


# ---------------------------------------------------------------- 腿式里程计缓存
class OdomBuffer:
    """缓存腿式里程计（机身位姿），按时间插值出雷达在里程计坐标系中的位姿。"""

    def __init__(self, T_BL, keep=10.0):
        self.T_BL = T_BL
        self.R_BL = R.from_matrix(T_BL[:3, :3])
        self.keep = keep
        self.t, self.p, self.q = [], [], []

    def add(self, t, p, q_xyzw):
        if self.t and t <= self.t[-1]:
            return
        self.t.append(t); self.p.append(p); self.q.append(q_xyzw)
        if self.t[-1] - self.t[0] > self.keep * 1.5:
            n = int(np.searchsorted(self.t, self.t[-1] - self.keep))
            del self.t[:n], self.p[:n], self.q[:n]

    def covers(self, t0, t1):
        return len(self.t) >= 2 and self.t[0] <= t0 and self.t[-1] >= t1 - 0.03

    def lidar_pose(self, ts):
        """ts 时刻（数组）雷达在里程计系中的 (Rotation, 位置 (N,3))；超出缓存范围时取端点。"""
        ts = np.atleast_1d(np.asarray(ts, float))
        t = np.asarray(self.t)
        ts_c = np.clip(ts, t[0], t[-1])
        i = np.clip(np.searchsorted(t, ts_c) - 1, 0, len(t) - 2)
        a = ((ts_c - t[i]) / np.maximum(t[i + 1] - t[i], 1e-9))[:, None]
        P, Q = np.asarray(self.p), np.asarray(self.q)
        pb = P[i] * (1 - a) + P[i + 1] * a
        R0 = R.from_quat(Q[i])
        dr = (R0.inv() * R.from_quat(Q[i + 1])).as_rotvec()
        Rb = R0 * R.from_rotvec(dr * a)
        return Rb * self.R_BL, pb + Rb.apply(self.T_BL[:3, 3])

    def lidar_T(self, t):
        Rl, pl = self.lidar_pose([t])
        return make_T(Rl.as_matrix()[0], pl[0])


# ---------------------------------------------------------------- 地图
class LocMap:
    """定位用的地图：三维点 + 法向 + KD 树；二维障碍物距离场（重定位用）。结果缓存在地图包 localization/ 下。"""

    def __init__(self, map_dir, prm: Params):
        map_dir = Path(map_dir)
        cache = map_dir / "localization" / "loc_map.npz"
        src = map_dir / "dense_map.pcd"
        key = np.array([CACHE_VERSION, prm.map_vox, prm.normal_k, src.stat().st_mtime, *prm.band, prm.grid_res])
        if cache.exists():
            c = np.load(cache)
            if c["key"].shape == key.shape and np.allclose(c["key"], key):
                self._load(c)
                return
        t0 = time.time()
        xyz = read_pcd_xyz(src)
        pts = voxel_down(xyz, prm.map_vox)
        nrm, planar = estimate_normals(pts, prm.normal_k)
        # 二维：高度带内的点视为障碍；地面点（z<0.1）所在格子视为“有地面”（可能的站位）
        lo = pts[:, :2].min(0) - 1.0
        shape = np.ceil((pts[:, :2].max(0) + 1.0 - lo) / prm.grid_res).astype(int)
        occ = np.zeros(shape, bool)
        floor = np.zeros(shape, bool)
        ij = np.floor((xyz[:, :2] - lo) / prm.grid_res).astype(int)
        band = (xyz[:, 2] > prm.band[0]) & (xyz[:, 2] < prm.band[1])
        occ[tuple(ij[band].T)] = True
        floor[tuple(ij[np.abs(xyz[:, 2]) < 0.1].T)] = True
        cache.parent.mkdir(exist_ok=True)
        np.savez(cache, key=key, pts=pts.astype(np.float32), nrm=nrm.astype(np.float32),
                 planar=planar.astype(np.float32), lo=lo, occ=occ, floor=floor, res=prm.grid_res)
        print(f"[定位] 地图预处理 {len(xyz)} → {len(pts)} 点，{time.time() - t0:.1f} s，缓存到 {cache}", flush=True)
        self._load(np.load(cache))

    def _load(self, c):
        self.pts = c["pts"].astype(np.float64)
        self.nrm = c["nrm"].astype(np.float64)
        self.tree = cKDTree(self.pts)
        self.lo, self.occ, self.floor = c["lo"], c["occ"], c["floor"]
        self.res = float(c["res"])
        # 二维似然场（离障碍越近得分越高）与候选站位（附近有地面、离障碍 ≥ 0.2 m）
        dist = ndimage.distance_transform_edt(~self.occ) * self.res
        self.cand = ndimage.binary_dilation(self.floor, iterations=3) & (dist >= 0.2)
        # 三维栅格（0.1 m）：结构似然场（离地 > 0.15 m 的地图点，离得越近得分越高）与“实心”体素（看穿判据用）
        v = self.v3 = 0.1
        self.lo3 = self.pts.min(0) - 0.5
        self.sh3 = np.ceil((self.pts.max(0) + 0.5 - self.lo3) / v).astype(int)
        cnt = np.zeros(self.sh3, np.int32)
        np.add.at(cnt, tuple(np.floor((self.pts - self.lo3) / v).astype(int).T), 1)
        zc = self.lo3[2] + (np.arange(self.sh3[2]) + 0.5) * v
        struct = (cnt > 0) & (zc > 0.15)[None, None, :]
        d3 = ndimage.distance_transform_edt(~struct) * v
        self.like3 = np.exp(-0.5 * (np.minimum(d3, 0.5) / 0.1) ** 2).astype(np.float32)
        # 至少 3 个地图点（5 cm 下采样后）才算实心，去掉噪点；不含地面层
        self.occ3 = (cnt >= 3) & (zc > 0.2)[None, None, :]

    def lookup3(self, grid, x, outside=0):
        ij = np.floor((x - self.lo3) / self.v3).astype(np.int64)
        ins = ((ij >= 0) & (ij < self.sh3)).all(-1)
        ij = np.where(ins[..., None], ij, 0)
        v = grid[ij[..., 0], ij[..., 1], ij[..., 2]]
        return np.where(ins, v, outside)

    def see_through(self, x, origin, margin=0.3, step=0.1, struct_z=0.25, inlier_dist=0.1):
        """看穿比例：终点能对上地图（离地图 inlier_dist 内）的射线中，途中穿过地图实心体素的比例。
        只看能对上地图的终点，避免把窗外、门外（地图裁掉的范围）的点误算成看穿。"""
        x = x[x[:, 2] > struct_z]
        d, _ = self.tree.query(x, distance_upper_bound=inlier_dist, workers=-1)
        x = x[np.isfinite(d)]
        if len(x) < 20:
            return 1.0
        dv = x - origin
        r = np.linalg.norm(dv, axis=1)
        u = dv / r[:, None]
        viol = np.zeros(len(x), bool)
        for s in np.arange(0.3, r.max() - margin, step):
            m = r - margin > s
            hit = self.lookup3(self.occ3, origin + u[m] * s, False)
            viol[np.flatnonzero(m)[hit]] = True
        return float(viol.mean())


# ---------------------------------------------------------------- ICP
def icp_point_to_plane(src, T0, mp: LocMap, stages, inlier_dist, struct_z=0.15):
    """src: (N,3) 雷达系点。返回 (T, 结构内点比例, 结构点数, 全部点内点比例)。

    结构内点比例只统计变换后离地高于 struct_z 的点（墙、家具）：L1 倒装、大部分点打在地面上，
    地面点在任何位置都能对上，用全部点的内点比例无法区分对错位姿。"""
    Rm, t = T0[:3, :3].copy(), T0[:3, 3].copy()
    for max_d, k, iters in stages:
        for _ in range(iters):
            x = src @ Rm.T + t
            d, j = mp.tree.query(x, distance_upper_bound=max_d, workers=-1)
            ok = np.isfinite(d)
            if ok.sum() < 30:
                break
            x, j = x[ok], j[ok]
            n = mp.nrm[j]
            r = np.einsum("ij,ij->i", x - mp.pts[j], n)
            w = np.where(np.abs(r) < k, (1 - (r / k) ** 2) ** 2, 0.0)
            J = np.hstack([np.cross(x, n), n])
            H = (J * w[:, None]).T @ J
            g = (J * w[:, None]).T @ r
            try:
                delta = -np.linalg.solve(H + 1e-6 * np.eye(6), g)
            except np.linalg.LinAlgError:
                break
            dR = R.from_rotvec(delta[:3]).as_matrix()
            Rm, t = dR @ Rm, dR @ t + delta[3:]
            if np.linalg.norm(delta[:3]) < 1e-4 and np.linalg.norm(delta[3:]) < 1e-4:
                break
    x = src @ Rm.T + t
    d, _ = mp.tree.query(x, distance_upper_bound=inlier_dist, workers=-1)
    inl = np.isfinite(d)
    st = x[:, 2] > struct_z
    n_st = int(st.sum())
    fit_st = float(inl[st].mean()) if n_st >= 20 else 0.0
    return make_T(Rm, t), fit_st, n_st, float(inl.mean())


# ---------------------------------------------------------------- 定位器
class Localizer:
    def __init__(self, map_dir, robot_cfg_dir, prm: Params = None, log=print):
        self.prm = prm or Params()
        self.log = log
        robot_cfg_dir = Path(robot_cfg_dir)
        self.T_BL = np.loadtxt(robot_cfg_dir / "T_base_lidar.txt")
        sv = np.load(robot_cfg_dir / "self_voxels.npz")
        self._init_self_filter(sv["voxels"], float(sv["size"]))
        self.map = LocMap(map_dir, self.prm)
        self.odom = OdomBuffer(self.T_BL, self.prm.keep_odom)
        self.recent = deque()             # (t_end, 雷达系去畸变点, 里程计系雷达位姿 4x4)
        self.state = "INIT"
        self.T = None                     # 当前地图系雷达位姿
        self.T_odom_last = None           # 上一帧对应的里程计系雷达位姿
        self.n_bad = 0
        self.guess = None                 # (x, y, yaw, 半径, 角度范围) 初始位姿提示（机身）
        self.last_good = None
        self.pending = deque()            # 等待里程计的激光帧
        self.n_dropped = 0
        self.window = self.prm.reloc_window
        self.next_try = -np.inf
        self.verbose = False
        self.pending_hyp = None           # 全图搜索的待确认结果：(雷达位姿, 对应的里程计位姿, 已一致次数)

    # ---------- 外部接口
    def set_initial_pose(self, x, y, yaw, radius=1.0, yaw_range_deg=30.0):
        """给定机身在地图中的大致位姿，下一次初始化只在其附近搜索。"""
        self.guess = (x, y, yaw, radius, np.radians(yaw_range_deg))
        self.state = "INIT"
        self.window, self.next_try = self.prm.reloc_window, -np.inf

    def add_odom(self, t, p, q_xyzw):
        """加入一条腿式里程计；返回因此得以处理的激光帧结果列表。"""
        self.odom.add(t, np.asarray(p, float), np.asarray(q_xyzw, float))
        return self._flush()

    def add_scan(self, stamp, xyz, trel):
        """stamp：帧头时间；xyz：(N,3) 雷达系点；trel：每点相对帧头的时间（秒）。
        里程计还没覆盖到帧结束时刻时先挂起，等后续里程计到达再处理。返回本次处理完的结果列表。"""
        self.pending.append((stamp, xyz, trel))
        return self._flush()

    def _flush(self):
        out = []
        while self.pending:
            stamp, xyz, trel = self.pending[0]
            t_end = stamp + float(np.max(trel)) if len(trel) else stamp
            if self.odom.t and self.odom.t[-1] < t_end - 0.03:
                # 里程计没跟上；挂起太久（里程计中断）则丢弃
                if len(self.pending) > 5:
                    self.pending.popleft()
                    self.n_dropped += 1
                    continue
                break
            self.pending.popleft()
            r = self._process(stamp, xyz, trel)
            if r is not None:
                out.append(r)
        return out

    def _process(self, stamp, xyz, trel):
        t_start = time.perf_counter()
        xyz = np.asarray(xyz, np.float64)
        trel = np.asarray(trel, np.float64)
        ok = np.isfinite(xyz).all(1)
        rng = np.linalg.norm(xyz, axis=1)
        ok &= (rng > self.prm.min_range) & (rng < self.prm.max_range)
        ok[ok] &= ~self._is_self(xyz[ok])
        xyz, trel = xyz[ok], trel[ok]
        if len(xyz) < 50:
            return None
        t_end = stamp + trel.max()
        if not self.odom.covers(stamp + trel.min(), t_end):
            return None
        pts = self._deskew(xyz, stamp + trel, t_end)
        pts = voxel_down(pts, self.prm.scan_vox)
        T_odom = self.odom.lidar_T(t_end)
        self.recent.append((t_end, pts, T_odom))
        while self.recent and self.recent[0][0] < t_end - self.prm.reloc_max_window:
            self.recent.popleft()

        if self.state in ("INIT", "LOST"):
            res = self._relocalize(t_end)
        else:
            res = self._track(t_end, pts, T_odom)
        if res.T_map_lidar is not None:
            self.T, self.T_odom_last = res.T_map_lidar, T_odom
            res.T_map_base = res.T_map_lidar @ np.linalg.inv(self.T_BL)
        res.n_points = len(pts)
        res.ms = (time.perf_counter() - t_start) * 1000
        return res

    # ---------- 内部
    def _init_self_filter(self, vox, size):
        lo = vox.min(0) - 2
        g = np.zeros(vox.max(0) - lo + 3, bool)
        g[tuple((vox - lo).T)] = True
        self.self_grid = ndimage.binary_dilation(g, iterations=1)
        self.self_lo, self.self_size = lo, size

    def _is_self(self, x):
        k = np.floor(x / self.self_size).astype(np.int64) - self.self_lo
        inside = ((k >= 0) & (k < self.self_grid.shape)).all(1)
        res = np.zeros(len(x), bool)
        res[inside] = self.self_grid[tuple(k[inside].T)]
        return res

    def _deskew(self, xyz, tp, t_end):
        Rp, pp = self.odom.lidar_pose(tp)
        Re, pe = self.odom.lidar_pose([t_end])
        Re = Re[0]
        return Re.inv().apply(Rp.apply(xyz) + pp - pe[0])

    def _gravity_fix(self, T, T_odom):
        """把雷达姿态的“上”方向校正为腿式里程计给出的方向（偏航不动）。"""
        up_L = T_odom[:3, :3].T @ np.array([0, 0, 1.0])
        cur = T[:3, :3] @ up_L
        T = T.copy()
        T[:3, :3] = rot_between(cur, np.array([0, 0, 1.0])).as_matrix() @ T[:3, :3]
        return T

    def _track(self, t_end, pts, T_odom):
        prm = self.prm
        T_pred = self.T @ np.linalg.inv(self.T_odom_last) @ T_odom
        T_pred = self._gravity_fix(T_pred, T_odom)
        T_icp, fit, n_st, fit_all = icp_point_to_plane(pts, T_pred, self.map, prm.icp_stages, prm.inlier_dist)
        dRv = R.from_matrix(T_icp[:3, :3] @ T_pred[:3, :3].T).as_rotvec()
        dp = T_icp[:3, 3] - T_pred[:3, 3]
        good = fit >= prm.min_fitness and np.linalg.norm(dp) <= prm.max_corr_trans \
            and np.degrees(np.linalg.norm(dRv)) <= prm.max_corr_rot
        if good:
            T = make_T(R.from_rotvec(prm.alpha * dRv).as_matrix() @ T_pred[:3, :3], T_pred[:3, 3] + prm.alpha * dp)
            T = self._gravity_fix(T, T_odom)
            self.n_bad = 0
            self.last_good = T
            status = "TRACKING"
        else:
            T = T_pred
            self.n_bad += 1
            status = "PREDICT"
            if self.n_bad >= prm.lost_frames:
                self.state = "LOST"
                status = "LOST"
                self.window, self.next_try = prm.reloc_window, -np.inf
                self.log(f"[定位] 连续 {self.n_bad} 帧配准不可信（内点比例 {fit:.2f}），判为丢失，开始重定位")
        return Result(t_end, status, T, fitness=fit,
                      info={"fit_all": round(fit_all, 3), "n_struct": n_st,
                            "corr_m": round(float(np.linalg.norm(dp)), 3),
                            "corr_deg": round(float(np.degrees(np.linalg.norm(dRv))), 2)})

    def _accumulate(self, window):
        """把最近 window 秒的帧用里程计拼到最新一帧的雷达系。"""
        T_last_inv = np.linalg.inv(self.recent[-1][2])
        t_last = self.recent[-1][0]
        parts = []
        for t, p, To in self.recent:
            if t < t_last - window:
                continue
            Tr = T_last_inv @ To
            parts.append(p @ Tr[:3, :3].T + Tr[:3, 3])
        return voxel_down(np.concatenate(parts), self.prm.scan_vox)

    def _relocalize(self, t_end):
        prm = self.prm
        span = self.recent[-1][0] - self.recent[0][0] if self.recent else 0.0
        if len(self.recent) < prm.reloc_min_frames or span < self.window - 0.1 or t_end < self.next_try:
            return Result(t_end, self.state, None, info={"frames": len(self.recent), "window": self.window})
        t0 = time.perf_counter()
        cloud = self._accumulate(self.window)
        T_odom = self.recent[-1][2]
        if getattr(self, "debug_dump", None):
            np.savez(self.debug_dump, cloud=cloud, T_odom=T_odom, t_end=t_end)
        if self.state == "LOST" and self.last_good is not None:
            Tb = self.last_good @ np.linalg.inv(self.T_BL)
            regions = [(Tb[0, 3], Tb[1, 3], yaw_of(Tb[:3, :3]), 3.0, np.pi), None]
        elif self.guess is not None:
            regions = [self.guess]
        else:
            regions = [None]
        for reg in regions:
            T, score, info = self._search(cloud, T_odom, reg)
            min_s, min_m = (prm.reloc_min_score, prm.reloc_min_margin) if reg is None else (prm.hint_min_score, prm.hint_min_margin)
            ok = T is not None and score >= min_s and score - info["runner_up"] >= min_m
            if ok and reg is None and prm.global_confirm > 1:
                # 全图搜索：与上一次的结果（按里程计推算到现在）一致才累计，否则重新计数
                n = 1
                if self.pending_hyp is not None:
                    Tp, To_p, n_p = self.pending_hyp
                    Tprop = Tp @ np.linalg.inv(To_p) @ T_odom
                    dyaw = abs((yaw_of(Tprop[:3, :3]) - yaw_of(T[:3, :3]) + np.pi) % (2 * np.pi) - np.pi)
                    if np.linalg.norm(Tprop[:2, 3] - T[:2, 3]) < 0.3 and dyaw < np.radians(10):
                        n = n_p + 1
                self.pending_hyp = (T, T_odom, n)
                info["confirm"] = f"{n}/{prm.global_confirm}"
                if n < prm.global_confirm:
                    ok = False
                    info["reason"] = f"全图搜索结果待确认（{n}/{prm.global_confirm}），"
            elif not ok and reg is None:
                self.pending_hyp = None
            if ok:
                break
        info["search_s"] = round(time.perf_counter() - t0, 2)
        info["window"] = round(self.window, 1)
        if not ok:
            self.log(f"[定位] {'初始化' if self.state == 'INIT' else '重定位'}暂不确定（窗口 {self.window:.1f} s）："
                     f"最佳得分 {score:.2f}，次佳 {info.get('runner_up', 0):.2f}，{info.get('reason', '')}继续积累数据"
                     + (f"；候选（得分, x, y）{sorted(info['cands'], reverse=True)[:4]}" if self.verbose and "cands" in info else ""))
            if "confirm" not in info:
                self.window = min(self.window + prm.reloc_retry * 1.5, prm.reloc_max_window)
            self.next_try = t_end + prm.reloc_retry
            return Result(t_end, self.state, None, fitness=info.get("fit_struct", 0.0), info=info)
        prev = self.state
        self.state, self.n_bad, self.last_good, self.guess = "TRACKING", 0, T, None
        self.pending_hyp = None
        self.window, self.next_try = prm.reloc_window, -np.inf
        Tb = T @ np.linalg.inv(self.T_BL)
        short = {k: v for k, v in info.items() if k != "cands"}
        self.log(f"[定位] {'初始化' if prev == 'INIT' else '重定位'}成功：机身 x={Tb[0, 3]:.2f} y={Tb[1, 3]:.2f} "
                 f"yaw={np.degrees(yaw_of(Tb[:3, :3])):.1f}°，得分 {score:.2f}，{short}")
        return Result(t_end, "RELOCALIZED", T, fitness=info["fit_struct"], info=short)

    def _search(self, cloud, T_odom, region):
        """三维穷举打分 + ICP + 看穿校验。region=None 为全图；否则 (机身 x, y, yaw, 半径, 角度范围)。
        返回 (雷达位姿, 综合得分, 信息)；综合得分 = 结构内点比例 − 看穿比例。"""
        prm, mp = self.prm, self.map
        R_up = rot_between(T_odom[:3, :3].T @ np.array([0, 0, 1.0]), np.array([0, 0, 1.0])).as_matrix()
        g = cloud @ R_up.T                                      # 重力摆正、雷达在原点、朝向未知
        h = prm.lidar_height
        rr = np.linalg.norm(g, axis=1)
        sel = (g[:, 2] + h > 0.15) & (rr < 15)
        if sel.sum() < 50:
            return None, 0.0, {"reason": "离地结构点太少，", "runner_up": 0.0}
        rng = np.random.default_rng(0)
        S = g[sel]
        if len(S) > 400:
            S = S[rng.choice(len(S), 400, replace=False)]
        # 自由空间采样：每条射线在终点前 0.3 m 起往回每 0.4 m 取一点，最多 3 个
        rs = np.linalg.norm(S, axis=1)
        F = np.concatenate([S[m] / rs[m, None] * (rs[m, None] - 0.3 - 0.4 * k)
                            for k in range(3) for m in [rs - 0.3 - 0.4 * k > 0.3]])
        ci = np.argwhere(mp.cand)
        step = max(1, int(round(prm.search_step / mp.res)))
        ci = ci[(ci[:, 0] % step == 0) & (ci[:, 1] % step == 0)]
        pos = mp.lo + (ci + 0.5) * mp.res
        yaws = np.arange(0, 2 * np.pi, np.radians(prm.yaw_step_deg))
        lidar_xy0 = lidar_yaw = None
        if region is not None:
            bx, by, byaw, rad, yr = region
            # 机身 x 轴在重力摆正后的雷达系中的偏航：机身朝向 = 雷达偏航 + 该值
            fwd_g = R_up @ (self.T_BL[:3, :3].T @ np.array([1.0, 0, 0]))
            lidar_yaw = byaw - np.arctan2(fwd_g[1], fwd_g[0])
            off = R.from_euler("z", byaw).apply(self.T_BL[:3, 3])          # 雷达在机身前方约 0.29 m
            lidar_xy0 = np.array([bx + off[0], by + off[1]])
            pos = pos[np.hypot(pos[:, 0] - lidar_xy0[0], pos[:, 1] - lidar_xy0[1]) <= rad]
            yaws = yaws[np.abs((yaws - lidar_yaw + np.pi) % (2 * np.pi) - np.pi) <= yr + 1e-9]
        if len(pos) == 0 or len(yaws) == 0:
            return None, 0.0, {"reason": "搜索范围内没有候选，", "runner_up": 0.0}
        P3 = np.c_[pos, np.full(len(pos), h)]
        scores = np.empty((len(yaws), len(pos)), np.float32)
        for a, yw in enumerate(yaws):
            Rz = R.from_euler("z", yw).as_matrix()
            qs, qf = S @ Rz.T, F @ Rz.T
            for b0 in range(0, len(pos), 256):
                pp = P3[b0:b0 + 256, None, :]
                hit = mp.lookup3(mp.like3, pp + qs[None]).mean(1)
                viol = mp.lookup3(mp.occ3, pp + qf[None], False).mean(1) if len(qf) else 0.0
                scores[a, b0:b0 + 256] = hit - viol
        # 取若干个互相分开的候选
        order = np.argsort(scores, axis=None)[::-1]
        picks = []
        for o in order[:20000]:
            a, b = divmod(int(o), len(pos))
            if all(np.hypot(*(pos[b] - pos[b2])) > 0.5 or abs((yaws[a] - yaws[a2] + np.pi) % (2 * np.pi) - np.pi) > np.radians(20)
                   for a2, b2 in picks):
                picks.append((a, b))
            if len(picks) >= prm.n_candidates:
                break
        sub = cloud if len(cloud) <= 3000 else cloud[rng.choice(len(cloud), 3000, replace=False)]
        stages = ((1.0, 0.4, 10), (0.5, 0.2, 10), (0.15, 0.08, 15))
        results = []
        w_st = 1.0 if region is None else prm.hint_see_weight
        for a, b in picks:
            T0 = make_T(R.from_euler("z", yaws[a]).as_matrix() @ R_up, P3[b])
            T, fit, n_st, _ = icp_point_to_plane(sub, T0, mp, stages, prm.inlier_dist)
            T = self._gravity_fix(T, T_odom)
            if region is not None:
                # ICP 可能滑到提示范围以外（相似的另一处），这类结果不采纳
                dyaw = abs((yaw_of(T[:3, :3] @ R_up.T) - lidar_yaw + np.pi) % (2 * np.pi) - np.pi)
                if np.linalg.norm(T[:2, 3] - lidar_xy0) > region[3] + 0.3 or dyaw > region[4] + np.radians(10):
                    continue
            st = mp.see_through(cloud @ T[:3, :3].T + T[:3, 3], T[:3, 3])
            results.append((fit - w_st * st, T, fit, st))
        if not results:
            return None, 0.0, {"reason": "候选都滑出了提示范围，", "runner_up": 0.0}
        best = max(range(len(results)), key=lambda i: results[i][0])
        sc, Tb, fit, st = results[best]
        # 歧义度：收敛到另一个位姿（相距 > 0.5 m 或 > 20°）的候选中最高的综合得分
        others = [r[0] for r in results
                  if np.linalg.norm(r[1][:2, 3] - Tb[:2, 3]) > 0.5
                  or abs((yaw_of(r[1][:3, :3]) - yaw_of(Tb[:3, :3]) + np.pi) % (2 * np.pi) - np.pi) > np.radians(20)]
        info = {"candidates": len(pos) * len(yaws), "best_rank": best, "best_xy": [round(float(Tb[0, 3]), 2), round(float(Tb[1, 3]), 2)], "fit_struct": round(fit, 3),
                "see_through": round(st, 3), "runner_up": round(max(others), 3) if others else 0.0,
                "cloud_pts": len(cloud),
                "cands": [(round(r[0], 3), round(float(r[1][0, 3]), 2), round(float(r[1][1, 3]), 2)) for r in results]}
        return Tb, sc, info
