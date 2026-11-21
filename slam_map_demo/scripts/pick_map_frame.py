"""在地图包的俯视图上点选新的坐标原点和 +x 方向。

俯视图：灰度为离地 0.15~1.8 m 的点数（墙、家具越黑越密），去掉天花板以免遮挡；
蓝线为机身建图轨迹，红/绿箭头为当前地图坐标系的 x/y 轴。

操作：
    左键第 1 次  选新原点
    左键第 2 次  选 +x 方向上的任一点（离原点越远，方向越准）
    右键或 u     撤销上一个点
    s 或回车     保存结果并退出（两个点都选好后才可保存）
    缩放/平移    用窗口下方工具栏的放大镜/十字箭头；使用它们时左键不会选点，再点一次图标退出该模式

用法: pick_map_frame.py <地图包目录> <输出 yaml>
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
import yaml  # noqa: E402

RES = 0.02
Z_RANGE = (0.15, 1.8)


def main(map_dir, out_yaml):
    map_dir = Path(map_dir)
    xyz = np.asarray(o3d.io.read_point_cloud(str(map_dir / "dense_map.pcd")).points)
    traj_file = map_dir / "base_trajectory.txt"
    traj = np.loadtxt(traj_file if traj_file.exists() else map_dir / "trajectory.txt", ndmin=2)

    lo, hi = xyz[:, :2].min(0), xyz[:, :2].max(0)
    shape = np.ceil((hi - lo) / RES).astype(int) + 1
    wall = xyz[(xyz[:, 2] > Z_RANGE[0]) & (xyz[:, 2] < Z_RANGE[1])]
    floor = xyz[np.abs(xyz[:, 2]) < 0.08]
    img = np.zeros(shape[::-1])
    fl = ((floor[:, :2] - lo) / RES).astype(int)
    img[fl[:, 1], fl[:, 0]] = 0.15  # 扫到过的地面画成浅灰
    ij = ((wall[:, :2] - lo) / RES).astype(int)
    cnt = np.zeros(shape[::-1])
    np.add.at(cnt, (ij[:, 1], ij[:, 0]), 1)
    img = np.maximum(img, np.log1p(cnt) / np.log1p(np.percentile(cnt[cnt > 0], 95)))

    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(np.clip(img, 0, 1), cmap="gray_r", origin="lower", vmin=0, vmax=1,
              extent=[lo[0], lo[0] + shape[0] * RES, lo[1], lo[1] + shape[1] * RES])
    ax.plot(traj[:, 0], traj[:, 1], "-", color="tab:blue", lw=1, alpha=0.6, label="建图轨迹")
    ax.annotate("", (1, 0), (0, 0), arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.annotate("", (0, 1), (0, 0), arrowprops=dict(arrowstyle="->", color="green", lw=2))
    ax.text(1.05, 0, "当前 x", color="red")
    ax.text(0, 1.05, "当前 y", color="green")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("当前地图 x (m)")
    ax.set_ylabel("当前地图 y (m)")
    ax.legend(loc="upper right")

    pts, artists = [], []
    title = "左键选新原点 | 右键/u 撤销 | s/回车 保存"

    def redraw():
        for a in artists:
            a.remove()
        artists.clear()
        if pts:
            artists.append(ax.plot(*pts[0], "o", color="magenta", ms=9)[0])
        if len(pts) == 2:
            o, p = np.array(pts[0]), np.array(pts[1])
            d = (p - o) / np.linalg.norm(p - o)
            n = np.array([-d[1], d[0]])
            artists.append(ax.annotate("", o + d, o, arrowprops=dict(arrowstyle="-|>", color="magenta", lw=2.5)))
            artists.append(ax.annotate("", o + n, o, arrowprops=dict(arrowstyle="-|>", color="darkorange", lw=2.5)))
            artists.append(ax.text(*(o + 1.1 * d), "新 x", color="magenta", fontsize=12))
            artists.append(ax.text(*(o + 1.1 * n), "新 y", color="darkorange", fontsize=12))
            artists.append(ax.plot(*p, "x", color="magenta", ms=9)[0])
            yaw = np.degrees(np.arctan2(d[1], d[0]))
            ax.set_title(f"新原点 ({o[0]:.3f}, {o[1]:.3f})，新 x 轴相对当前 x 轴转 {yaw:.1f}°  |  s/回车 保存，右键撤销")
        elif pts:
            ax.set_title("再左键选 +x 方向上的一点（离原点远一些更准）| 右键/u 撤销")
        else:
            ax.set_title(title)
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes is not ax or fig.canvas.toolbar.mode:
            return
        if ev.button == 1 and len(pts) < 2:
            pts.append((ev.xdata, ev.ydata))
            print(f"第 {len(pts)} 个点: ({ev.xdata:.3f}, {ev.ydata:.3f})", flush=True)
        elif ev.button == 3 and pts:
            pts.pop()
        redraw()

    def on_key(ev):
        if ev.key == "u" and pts:
            pts.pop()
            redraw()
        elif ev.key in ("s", "enter") and len(pts) == 2:
            o, p = np.array(pts[0]), np.array(pts[1])
            yaw = float(np.arctan2(*(p - o)[::-1]))
            res = {"map_dir": str(map_dir.resolve()),
                   "origin": [round(float(v), 4) for v in o],
                   "x_toward": [round(float(v), 4) for v in p],
                   "yaw_deg": round(float(np.degrees(yaw)), 3),
                   "note": "坐标均为当前地图坐标系；新坐标 = Rz(-yaw) @ (旧坐标 - origin)"}
            Path(out_yaml).parent.mkdir(parents=True, exist_ok=True)
            Path(out_yaml).write_text(yaml.safe_dump(res, allow_unicode=True, sort_keys=False))
            print("已保存", out_yaml, res, flush=True)
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()


if __name__ == "__main__":
    main(*sys.argv[1:3])
