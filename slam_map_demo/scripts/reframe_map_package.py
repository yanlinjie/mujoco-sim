"""按点选的原点和 +x 方向，把地图包整体换到新坐标系，生成新版本地图包。

新坐标 = Rz(-yaw) @ (旧坐标 - origin)，yaw = atan2(x_toward - origin)；z 不变（地面仍为 z=0）。
变换内容：dense_map.pcd、mesh/scene.obj（只改顶点）、trajectory.txt（x y yaw 列）、map_from_slam.txt（左乘新变换）；
复制 mesh/scene_texture.png、meta.yaml（更新 version）。waypoints.yaml 不复制（由点选结果重新生成），
原点处放一个默认点位，朝向 +x。

用法: reframe_map_package.py <源地图包> <新地图包> <frame.yaml，含 origin: [x, y] 与 x_toward: [x, y]>
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_hilti_map_package import write_pcd  # noqa: E402
from map_to_mujoco import read_pcd_xyz  # noqa: E402


def main(src, dst, frame_yaml):
    src, dst = Path(src), Path(dst)
    fr = yaml.safe_load(open(frame_yaml))
    o = np.array(fr["origin"][:2], float)
    d = np.array(fr["x_toward"][:2], float) - o
    yaw = float(np.arctan2(d[1], d[0]))
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])   # Rz(-yaw)
    A = np.eye(4); A[:3, :3] = Rz; A[:3, 3] = -Rz @ np.array([o[0], o[1], 0.0])
    print(f"新原点 ({o[0]:+.3f}, {o[1]:+.3f})，+x 方向 {np.degrees(yaw):.2f}°")
    (dst / "mesh").mkdir(parents=True, exist_ok=True)

    xyz = read_pcd_xyz(src / "dense_map.pcd").astype(float) @ A[:3, :3].T + A[:3, 3]
    write_pcd(dst / "dense_map.pcd", [("x", "F"), ("y", "F"), ("z", "F")],
              [xyz[:, i].astype(np.float32) for i in range(3)])
    out = []
    for l in open(src / "mesh" / "scene.obj"):
        if l.startswith("v "):
            v = A[:3, :3] @ np.array(l.split()[1:4], float) + A[:3, 3]
            out.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        else:
            out.append(l)
    (dst / "mesh" / "scene.obj").write_text("".join(out))
    shutil.copy(src / "mesh" / "scene_texture.png", dst / "mesh" / "scene_texture.png")

    if (src / "trajectory.txt").exists():
        tr = np.loadtxt(src / "trajectory.txt", ndmin=2)
        tr[:, 1:4] = tr[:, 1:4] @ A[:3, :3].T + A[:3, 3]
        tr[:, 4] = (tr[:, 4] - yaw + np.pi) % (2 * np.pi) - np.pi
        np.savetxt(dst / "trajectory.txt", tr, fmt="%.4f", header="机身轨迹（地图坐标）：时间戳 x y z yaw")
    T = A @ np.loadtxt(src / "map_from_slam.txt")
    np.savetxt(dst / "map_from_slam.txt", T, fmt="%.9f", header="4x4 齐次变换：地图坐标 = T @ 调平后激光/重建坐标")
    shutil.copy(frame_yaml, dst / "frame_pick.yaml")

    meta = yaml.safe_load(open(src / "meta.yaml"))
    meta["version"] = dst.name
    meta["frame"] = (f"由 {src.name} 换系：原点 ({o[0]:.3f}, {o[1]:.3f})、+x 朝向 {np.degrees(yaw):.2f}°（旧坐标系），"
                     f"点选来源见 frame_pick.yaml")
    (dst / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True))
    wps = [{"id": "wp_01", "name": "原点（充电桩）", "stand_pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
            "lookat": {"x": 1.0, "y": 0.0, "z": 0.5}, "arrive_tolerance": {"xy": 0.15, "yaw": 0.2},
            "safety_distance": 0.3}]
    (dst / "waypoints.yaml").write_text(yaml.safe_dump({"waypoints": wps}, sort_keys=False, allow_unicode=True))
    return A


if __name__ == "__main__":
    main(*sys.argv[1:4])
