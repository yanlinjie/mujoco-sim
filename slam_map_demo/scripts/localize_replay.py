"""用录制包回放验证激光实时定位（localizer.py）：按录制顺序把 /utlidar/cloud 与 /utlidar/robot_odom 逐条喂给定位器，
与建图轨迹（参考位姿）对比误差，并统计每帧耗时。

可以从包的任意时刻开始（检验在地图中任意位置的初始化），可以按真实速度回放（--realtime）。
需要 rosbags（slam_data/.venv-r2b 环境中有）。

用法:
  localize_replay.py <地图包> <bag 目录> [--start 秒] [--duration 秒] [--init X Y YAW] [--realtime]
                     [--ref TUM] [--out 结果前缀] [--kidnap 秒]
  --ref：参考雷达轨迹（TUM，自建 SLAM 坐标系），配合地图包 map_from_slam.txt 换到地图坐标系做对比；
  --kidnap：回放到该时刻（相对 --start）时跳过 10 s 数据，并让腿式里程计在这 10 s 内“冻结”（跳过后的里程计
            整体平移旋转到跳过前的位姿上接续），相当于机器狗被抱走、腿没动，检验丢失检测与重定位。
输出：<前缀>_poses.txt（每帧机身 x y z yaw 与状态）、<前缀>.png（轨迹与误差图）、终端打印误差统计。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from localizer import Localizer, Params  # noqa: E402

HERE = Path(__file__).resolve().parent


def decode_cloud(m):
    off = {f.name: f.offset for f in m.fields}
    a = np.frombuffer(m.data, np.uint8).reshape(-1, m.point_step)
    f32 = lambda k: a[:, off[k]:off[k] + 4].copy().view("<f4").ravel()  # noqa: E731
    return np.stack([f32("x"), f32("y"), f32("z")], -1), f32("time")


def stamp(m):
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map_dir")
    ap.add_argument("bag")
    ap.add_argument("--robot-cfg", default=str(HERE.parent / "config" / "go2_l1"))
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=1e9)
    ap.add_argument("--init", type=float, nargs=3, metavar=("X", "Y", "YAW"))
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--ref")
    ap.add_argument("--out")
    ap.add_argument("--kidnap", type=float)
    ap.add_argument("--hint-from-ref", type=float, nargs=2, metavar=("偏移米", "偏航偏差度"),
                    help="测试用：以参考轨迹第一帧的机身位姿加上给定偏差作为初始位姿提示")
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--verbose", action="store_true", help="打印重定位候选")
    ap.add_argument("--dump", help="调试：保存初始化时的积累点云")
    ap.add_argument("--reloc-window", type=float, help="重定位积累时长（秒）")
    args = ap.parse_args()

    prm = Params()
    if args.alpha is not None:
        prm.alpha = args.alpha
    if args.reloc_window:
        prm.reloc_window = args.reloc_window
        prm.reloc_min_frames = int(args.reloc_window * 15 * 0.9)
    loc = Localizer(args.map_dir, args.robot_cfg, prm)
    loc.debug_dump = args.dump
    loc.verbose = args.verbose
    if args.init:
        loc.set_initial_pose(*args.init)

    ref_t = T_ref = None
    if args.ref:
        ref = np.loadtxt(args.ref, ndmin=2)
        A = np.loadtxt(Path(args.map_dir) / "map_from_slam.txt")
        T_ref = np.array([A @ np.vstack([np.c_[R.from_quat(p[4:8]).as_matrix(), p[1:4]], [0, 0, 0, 1]]) for p in ref])
        ref_t = ref[:, 0]

    def ref_err(r):
        i = int(np.argmin(np.abs(ref_t - r.stamp)))
        if abs(ref_t[i] - r.stamp) > 0.02:
            return None
        e = np.linalg.norm(r.T_map_lidar[:2, 3] - T_ref[i][:2, 3])
        ye = np.arctan2(r.T_map_lidar[1, 0], r.T_map_lidar[0, 0]) - np.arctan2(T_ref[i][1, 0], T_ref[i][0, 0])
        return e, abs(np.degrees((ye + np.pi) % (2 * np.pi) - np.pi))

    hinted = False
    odom_fix = None          # 模拟被抱走：跳过后的里程计接到跳过前的位姿上
    last_odom_T = None
    reloc_log = []
    ts = get_typestore(Stores.ROS2_HUMBLE)
    results = []
    t_wall0 = time.time()
    with AnyReader([Path(args.bag)], default_typestore=ts) as rd:
        conns = [c for c in rd.connections if c.topic in ("/utlidar/cloud", "/utlidar/robot_odom")]
        t_bag0 = rd.start_time * 1e-9
        t_first = None
        skip_until = None
        for c, t_log, raw in rd.messages(connections=conns, start=int((t_bag0 + args.start) * 1e9)):
            tl = t_log * 1e-9 - t_bag0 - args.start
            if tl > args.duration:
                break
            if args.kidnap is not None and skip_until is None and tl >= args.kidnap:
                skip_until = tl + 10.0
                print(f"[回放] {tl:.1f} s：跳过 10 s 数据（模拟被搬走）", flush=True)
            if skip_until is not None and tl < skip_until:
                continue
            if args.realtime:
                lag = tl - (time.time() - t_wall0)
                if lag > 0:
                    time.sleep(lag)
            m = rd.deserialize(raw, c.msgtype)
            if c.topic == "/utlidar/robot_odom":
                p, q = m.pose.pose.position, m.pose.pose.orientation
                To = np.eye(4)
                To[:3, :3], To[:3, 3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), [p.x, p.y, p.z]
                if skip_until is not None and odom_fix is None:
                    odom_fix = last_odom_T @ np.linalg.inv(To)
                    # 只保留偏航与水平位移的修正（重力方向不变）
                    yaw = np.arctan2(odom_fix[1, 0], odom_fix[0, 0])
                    odom_fix[:3, :3] = R.from_euler("z", yaw).as_matrix()
                    odom_fix[2, 3] = 0.0
                if odom_fix is not None:
                    To = odom_fix @ To
                last_odom_T = To
                out = loc.add_odom(stamp(m), To[:3, 3], R.from_matrix(To[:3, :3]).as_quat())
            else:
                xyz, trel = decode_cloud(m)
                if args.hint_from_ref and not hinted:
                    hinted = True
                    i = int(np.argmin(np.abs(ref_t - stamp(m))))
                    Tb = T_ref[i] @ np.linalg.inv(loc.T_BL)
                    yaw = np.arctan2(Tb[1, 0], Tb[0, 0])
                    dx, dy = args.hint_from_ref[0] * np.cos(yaw + 1.0), args.hint_from_ref[0] * np.sin(yaw + 1.0)
                    hint = (Tb[0, 3] + dx, Tb[1, 3] + dy, yaw + np.radians(args.hint_from_ref[1]))
                    loc.set_initial_pose(*hint)
                    print(f"[回放] 初始位姿提示 x={hint[0]:.2f} y={hint[1]:.2f} yaw={np.degrees(hint[2]):.0f}°"
                          f"（参考 x={Tb[0, 3]:.2f} y={Tb[1, 3]:.2f} yaw={np.degrees(yaw):.0f}°）", flush=True)
                out = loc.add_scan(stamp(m), xyz, trel)
            for r in out:
                if t_first is None:
                    t_first = r.stamp
                results.append(r)
                if r.status == "RELOCALIZED" and ref_t is not None:
                    e = ref_err(r)
                    reloc_log.append((round(tl, 1), None if e is None else (round(float(e[0]), 3), round(float(e[1]), 1))))
                    if e is not None:
                        print(f"[回放] 初始化/重定位位置与参考相差 {e[0] * 100:.1f} cm，偏航差 {e[1]:.1f}°", flush=True)
                if len(results) % 150 == 0 or r.status in ("RELOCALIZED", "LOST"):
                    d = r.to_dict()
                    print(f"[回放 {tl:6.1f} s] {d['status']:<11} "
                          + (f"x={d['x']:.2f} y={d['y']:.2f} yaw={np.degrees(d['yaw']):.0f}° " if "x" in d else "")
                          + f"内点 {d['fitness']:.2f} 耗时 {d['ms']:.0f} ms", flush=True)
    wall = time.time() - t_wall0

    t_first_fix = next((r.stamp for r in results if r.status == "RELOCALIZED"), None)
    if t_first_fix is not None:
        print(f"首次定位成功用时 {t_first_fix - results[0].stamp:.1f} s（从第一帧算起）")
    st = np.array([r.status for r in results])
    ms = np.array([r.ms for r in results if r.status in ("TRACKING", "PREDICT")])
    print(f"\n共 {len(results)} 帧（丢弃 {loc.n_dropped} 帧），状态计数：{dict(zip(*np.unique(st, return_counts=True)))}")
    if len(ms):
        print(f"跟踪帧耗时：中位数 {np.median(ms):.1f} ms，95% {np.percentile(ms, 95):.1f} ms，最大 {ms.max():.0f} ms"
              f"（激光 15 Hz，每帧预算 66 ms）；回放总墙钟 {wall:.0f} s")
    fit = np.array([r.fitness for r in results if r.status == "TRACKING"])
    if len(fit):
        print(f"跟踪帧内点比例：中位数 {np.median(fit):.2f}，5% 分位 {np.percentile(fit, 5):.2f}")

    ok = [r for r in results if r.T_map_base is not None]
    out = Path(args.out) if args.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(f"{out}_poses.txt", "w") as f:
            f.write("# 帧结束时刻 机身x y z yaw 状态 内点比例 耗时ms\n")
            for r in ok:
                x, y, yaw = r.xy_yaw()
                f.write(f"{r.stamp:.6f} {x:.4f} {y:.4f} {r.T_map_base[2, 3]:.4f} {yaw:.4f} {r.status} {r.fitness:.3f} {r.ms:.1f}\n")

    if args.ref and ok:
        rt = ref_t
        errs, yerrs, tt, est_xy, ref_xy, sts = [], [], [], [], [], []
        for r in ok:
            i = int(np.argmin(np.abs(rt - r.stamp)))
            if abs(rt[i] - r.stamp) > 0.02:
                continue
            Tr = T_ref[i]
            Te = r.T_map_lidar
            errs.append(np.linalg.norm(Te[:2, 3] - Tr[:2, 3]))
            ye = np.arctan2(Te[1, 0], Te[0, 0]) - np.arctan2(Tr[1, 0], Tr[0, 0])
            yerrs.append(abs((ye + np.pi) % (2 * np.pi) - np.pi))
            tt.append(r.stamp - t_first)
            est_xy.append(Te[:2, 3]); ref_xy.append(Tr[:2, 3]); sts.append(r.status)
        errs, yerrs = np.array(errs), np.degrees(yerrs)
        sts = np.array(sts)
        trk = sts != "LOST"
        print(f"与参考轨迹对比（雷达位置，{len(errs)} 帧，不含 LOST 帧）：水平误差中位数 {np.median(errs[trk]) * 100:.1f} cm，"
              f"95% {np.percentile(errs[trk], 95) * 100:.1f} cm，最大 {errs[trk].max() * 100:.1f} cm；"
              f"偏航误差中位数 {np.median(yerrs[trk]):.2f}°，95% {np.percentile(yerrs[trk], 95):.2f}°，最大 {yerrs[trk].max():.2f}°")
        summary = {"frames": len(errs), "relocalizations": reloc_log,
                   "first_fix_s": round(float(t_first_fix - t_first), 2) if t_first_fix else None, "err_med_cm": float(np.median(errs[trk]) * 100),
                   "err_p95_cm": float(np.percentile(errs[trk], 95) * 100), "err_max_cm": float(errs[trk].max() * 100),
                   "yaw_med_deg": float(np.median(yerrs[trk])), "yaw_p95_deg": float(np.percentile(yerrs[trk], 95)),
                   "ms_med": float(np.median(ms)) if len(ms) else None, "ms_p95": float(np.percentile(ms, 95)) if len(ms) else None}
        if out:
            Path(f"{out}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
            plot(out, np.array(est_xy), np.array(ref_xy), np.array(tt), errs, yerrs, sts, args.map_dir)


def plot(out, est, ref, tt, errs, yerrs, sts, map_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK TC", "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    c = np.load(Path(map_dir) / "localization" / "loc_map.npz")
    pts = c["pts"]
    band = pts[(pts[:, 2] > 0.25) & (pts[:, 2] < 1.5)]
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.1, 1]})
    ax[0].scatter(band[::3, 0], band[::3, 1], s=0.2, c="0.7")
    ax[0].plot(ref[:, 0], ref[:, 1], "-", c="tab:green", lw=1.2, label="参考（建图轨迹）")
    ax[0].plot(est[:, 0], est[:, 1], "-", c="tab:blue", lw=0.8, label="实时定位")
    rel = sts == "RELOCALIZED"
    ax[0].scatter(est[rel, 0], est[rel, 1], c="red", s=40, marker="*", zorder=5, label="初始化/重定位")
    ax[0].set_aspect("equal"); ax[0].legend(loc="upper right"); ax[0].set_title("地图坐标系中的雷达轨迹（米）")
    ax[1].plot(tt, errs * 100, lw=0.7, label="水平位置误差（cm）")
    ax[1].plot(tt, yerrs * 10, lw=0.7, alpha=0.7, label="偏航误差（0.1°）")
    lost = sts == "LOST"
    if lost.any():
        ax[1].scatter(tt[lost], errs[lost] * 100, c="red", s=4, label="判为丢失")
    ax[1].set_ylim(0, max(30, np.percentile(errs * 100, 99) * 1.2))
    ax[1].set_xlabel("时间（s）"); ax[1].legend(); ax[1].grid(alpha=0.3); ax[1].set_title("与参考轨迹的误差")
    fig.tight_layout()
    fig.savefig(f"{out}.png", dpi=110)
    print(f"图：{out}.png")


if __name__ == "__main__":
    main()
