"""不开窗口跑一遍点位巡航（与 waypoint_tour.py 同一控制器），离屏渲染跟拍画面并录成 MP4。

画面（--view）：
  follow   机身后上方的跟拍相机（朝向平滑跟随机身），右上角叠加俯视小地图（碰撞高度场 + 已走轨迹 + 点位）；
  overview 与局域网推流（sim_stream.py）相同的固定斜侧视角，自动取景放下整张地图，默认只有仿真画面（1280x720）；
           加 --clock 在左上角显示日期时间，加 --panel-lines 7 在下方拼接与推流相同的滚动文字板块。
视频按 SPEEDUP 倍速播放。

用法（slam_map_demo 目录下）：
    MUJOCO_GL=egl $P -I -u scripts/record_tour_video.py <地图包> <场景 xml> <输出.mp4> [--speed 2] [--size 960x540]
        [--view follow|overview] [--pose-log 位姿.txt] [--laps 1]

--pose-log：把推流页面（sim_stream.py）滚动文字里的实时状态行按同样格式写入文本，每仿真秒一行；
时间戳为“录制开始时刻 + 仿真时间”，到达点位等事件也写成单独一行。
--laps：沿全部点位循环的圈数（每圈从充电桩出发，依次经过其余点位再回到充电桩）。
"""
import datetime
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import waypoint_tour as wt  # noqa: E402
from nav_grid import NavGrid  # noqa: E402


def minimap(grid, waypoints, size=260):
    occ = grid.clearance <= 0
    img = np.where(occ, 60, 225).astype(np.uint8)[::-1]          # 行=y，翻转使 +y 朝上
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    s = size / max(img.shape[:2])
    img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    H = img.shape[0]

    def px(xy):
        c, r = (np.asarray(xy) - grid.lo) / grid.res
        return int(c * s), int(H - 1 - r * s)
    for wp in waypoints:
        cv2.circle(img, px([wp["stand_pose"]["x"], wp["stand_pose"]["y"]]), 4, (60, 200, 60), -1)
    return img, px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map_dir"); ap.add_argument("scene"); ap.add_argument("out")
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--size", default="960x540")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--pose-log", default=None)
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--view", choices=["follow", "overview"], default="follow")
    ap.add_argument("--azimuth", type=float, default=None, help="overview 视角方位角，缺省与推流一致")
    ap.add_argument("--elevation", type=float, default=None, help="overview 视角俯仰角，缺省与推流一致")
    ap.add_argument("--panel-lines", type=int, default=0, help="overview 下方滚动文字行数，0 不显示")
    ap.add_argument("--clock", action="store_true", help="overview 左上角显示日期时间和倍速")
    a = ap.parse_args()
    if a.view == "overview" and a.size == "960x540":
        a.size = "1280x720"   # 与推流默认尺寸一致
    W, H = map(int, a.size.split("x"))

    model = mujoco.MjModel.from_xml_path(a.scene)
    model.opt.timestep = wt.DT
    model.vis.global_.offwidth, model.vis.global_.offheight = max(W, 1280), max(H, 960)
    data = mujoco.MjData(model)
    joints = model.actuator_trnid[:, 0]
    qadr, dadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    pos_adr = model.sensor_adr[model.sensor("frame_pos").id]
    quat_adr = model.sensor_adr[model.sensor("imu_quat").id]
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation = 2.2, -22
    opt = mujoco.MjvOption()

    grid = NavGrid.load(Path(a.map_dir) / "mujoco" / "collision_hfield.npz")
    waypoints = yaml.safe_load(open(Path(a.map_dir) / "waypoints.yaml"))["waypoints"]
    mm, mm_px = minimap(grid, waypoints)
    lines = []   # 滚动文字 (文字, 类别)，与推流 StatusFeed 相同
    panel = None
    if a.view == "overview":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from slam_map_demo.scripts import sim_stream as ss   # 复用推流的取景与文字板块，保证视角一致
        mujoco.mj_forward(model, data)
        cam = ss.fit_camera(model, ss.map_points(model, data), W, H,
                            -90.0 if a.azimuth is None else a.azimuth, -40.0 if a.elevation is None else a.elevation, 1.0)

        class _Feed:   # 离线没有滚动动画：t_new 取很早，文字直接就位
            def snapshot(self, n):
                return lines[-n:], -1e9
        if a.panel_lines > 0:
            panel = ss.StatusPanel(_Feed(), W, a.panel_lines)
    FH = H + (panel.height if panel else 0)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{FH}", "-r", str(a.fps), "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "23", a.out], stdin=subprocess.PIPE)
    every = max(1, int(round(a.speed / a.fps / wt.DT)))   # 每隔多少仿真步出一帧

    for _ in range(int(1.0 / wt.DT)):
        mujoco.mj_step(model, data)
    ctl = wt.TourController(grid, wt.load_tour(a.map_dir) * a.laps)
    vel_adr = model.sensor_adr[model.sensor("frame_vel").id]
    t0 = datetime.datetime.now()
    log = open(a.pose_log, "w", encoding="utf-8") if a.pose_log else None
    n_logged, n_results, log_every = 0, 0, int(round(1.0 / wt.DT))

    def stamp(t):
        return (t0 + datetime.timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S")

    def emit(text, t, kind="event"):
        line = stamp(t) + "  " + text
        lines.append((line, kind))
        if log:
            log.write(line + "\n")
    emit("已收到仿真器状态，开始同步画面", 0.0)
    emit(f"开始行走（沿全部 {len(ctl.tour)} 段点位路线，{a.laps} 圈）", 0.0)
    cam_yaw, trail, step, frames = None, [], 0, 0
    while ctl.state != "done" and ctl.time < 900 * a.laps:
        xy = data.sensordata[pos_adr:pos_adr + 2].copy()
        yaw = wt.yaw_from_quat(*data.sensordata[quat_adr:quat_adr + 4])
        q, dq, kp, kd, tau = ctl.step(xy, yaw, data.qpos[qadr])
        data.ctrl[:] = tau + kp * (q - data.qpos[qadr]) + kd * (dq - data.qvel[dadr])
        mujoco.mj_step(model, data)
        step += 1
        for r in ctl.results[n_results:]:
            wp_id, name, dist, yerr, dur = r
            emit(f"行走日志：到达 {name}（误差 {dist:.2f} m，用时 {dur:.0f} s）" if yerr is not None
                 else f"行走日志：{name} 未到达" + (f"（距离 {dist:.2f} m）" if dist is not None else ""), ctl.time,
                 "event" if yerr is not None else "warn")
        n_results = len(ctl.results)
        if step % log_every == 0:
            # 与 sim_stream.py 的 StatusFeed 同一格式（位置是 IMU 处，与 /pose 一致）
            w, qx, qy, qz = data.sensordata[quat_adr:quat_adr + 4]
            tilt = np.degrees(np.arccos(np.clip(1 - 2 * (qx * qx + qy * qy), -1, 1)))
            z = data.sensordata[pos_adr + 2]
            v = data.sensordata[vel_adr:vel_adr + 2]
            emit(f"运行中  位置 ({xy[0]:+.2f}, {xy[1]:+.2f}) m  朝向 {np.degrees(yaw):+.0f}°  "
                 f"速度 {float(np.hypot(*v)):.2f} m/s  机身高度 {z:.2f} m  倾斜 {tilt:.1f}°  "
                 f"行走中 {ctl.time:.0f} s（{ctl.state}）", ctl.time, "status")
            n_logged += 1
        if step % every:
            continue
        if a.view == "overview":
            renderer.update_scene(data, cam, opt)
            img = renderer.render()
            if panel:
                img = np.vstack([img, panel.render()])
            img = np.ascontiguousarray(img)
            if a.clock:
                label = stamp(ctl.time) + f"  x{a.speed:g}"
                cv2.putText(img, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            ff.stdin.write(img.tobytes())
            frames += 1
            continue
        cam_yaw = yaw if cam_yaw is None else cam_yaw + 0.08 * wt.wrap(yaw - cam_yaw)
        cam.lookat[:] = [xy[0], xy[1], 0.35]
        cam.azimuth = np.degrees(cam_yaw)
        renderer.update_scene(data, cam, opt)
        img = renderer.render().copy()
        trail.append(xy)
        m = mm.copy()
        for p0, p1 in zip(trail[:-1:3], trail[3::3]):
            cv2.line(m, mm_px(p0), mm_px(p1), (40, 40, 230), 2)
        cv2.circle(m, mm_px(xy), 5, (230, 60, 30), -1)
        m = m[:, :, ::-1]                                   # BGR → RGB
        h, w = m.shape[:2]
        img[10:10 + h, W - w - 10:W - 10] = m
        label = f"t={ctl.time:5.1f}s  x{a.speed:g}  {ctl.state}"
        cv2.putText(img, label, (12, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        ff.stdin.write(img.tobytes())
        frames += 1
    reached = sum(1 for r in ctl.results if r[3] is not None)
    emit(f"行走结束，用时 {ctl.time:.0f} s：到达 {reached}/{len(ctl.tour)} 个点位", ctl.time,
         "event" if reached == len(ctl.tour) else "warn")
    if panel:   # 结尾帧换上最新的文字板块
        img = np.ascontiguousarray(np.vstack([img[:H], panel.render()]))
    for _ in range(a.fps):   # 结尾停留 1 秒
        ff.stdin.write(img.tobytes())
    ff.stdin.close(); ff.wait()
    if log:
        log.close()
        print(f"位姿文本 {n_logged} 行 -> {a.pose_log}")
    print(f"巡航用时 {ctl.time:.0f} s，到达 {reached}/{len(ctl.tour)}；视频 {frames} 帧（{frames / a.fps:.0f} s）-> {a.out}")


if __name__ == "__main__":
    main()
