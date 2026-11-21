"""机器狗实时定位节点：通过 DDS 订阅 Go2 的激光帧与腿式里程计，在地图中实时定位，并对外提供位姿。

订阅（Go2 原生话题，ROS 2 话题 /utlidar/cloud 在 DDS 中名为 rt/utlidar/cloud）：
  rt/utlidar/cloud        sensor_msgs/PointCloud2（L1 激光帧，含逐点 time 字段）
  rt/utlidar/robot_odom   nav_msgs/Odometry（腿式里程计）
发布：
  rt/slam_map/base_pose   geometry_msgs/PoseStamped，机身在地图坐标系中的位姿（frame_id = map），每帧一条
  HTTP GET /pose          最新位姿 JSON：x y z yaw quat_xyzw status fitness stamp age_s ...
  HTTP POST /initial_pose {"x":..,"y":..,"yaw":..}  给初始位姿提示（只在附近 1 m、±30° 内搜索），可选
  HTTP GET /health

用法（unitree-mujoco 环境，需 unitree_sdk2py + cyclonedds）：
  localize_dds.py <地图包> [--iface eth0] [--domain 0] [--init X Y YAW | --resume] [--http-port 8767] [--log 结果.txt]
  --init：机身在地图中的大致位姿（误差 1 m、30° 以内），推荐；
  --resume：用上次运行保存的最后位姿（地图包 localization/last_pose.json）作提示，适合关机前后机器狗没被挪动的情况；
  都不给时做全图搜索：需要机器狗在开阔处走动几秒，看到的结构有歧义时会一直等待，不会给出错误结果。
  接真机：--iface 为连接机器狗的网卡、--domain 0；
  本机验证：先运行 dds_replay.py 把录制数据按真实时间发到本机回环网卡，再 --iface lo --domain 1。
"""
import argparse
import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from localizer import Localizer, Params  # noqa: E402

HERE = Path(__file__).resolve().parent


def decode_cloud(m):
    off = {f.name: f.offset for f in m.fields}
    a = np.frombuffer(bytes(m.data), np.uint8).reshape(-1, m.point_step)
    f32 = lambda k: a[:, off[k]:off[k] + 4].copy().view("<f4").ravel()  # noqa: E731
    return np.stack([f32("x"), f32("y"), f32("z")], -1), f32("time")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map_dir")
    ap.add_argument("--robot-cfg", default=str(HERE.parent / "config" / "go2_l1"))
    ap.add_argument("--iface", default=None)
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--init", type=float, nargs=3, metavar=("X", "Y", "YAW"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--http-host", default="0.0.0.0")
    ap.add_argument("--http-port", type=int, default=8767)
    ap.add_argument("--log", help="每帧结果写入文件（JSON 行）")
    ap.add_argument("--max-backlog", type=float, default=0.5, help="激光帧积压超过该秒数时丢弃旧帧")
    args = ap.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import geometry_msgs_msg_dds__PoseStamped_
    from unitree_sdk2py.idl.geometry_msgs.msg.dds_ import PoseStamped_
    from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_

    loc = Localizer(args.map_dir, args.robot_cfg, Params(), log=lambda s: print(s, flush=True))
    last_pose_file = Path(args.map_dir) / "localization" / "last_pose.json"
    if args.init:
        loc.set_initial_pose(*args.init)
    elif args.resume:
        if last_pose_file.exists():
            lp = json.loads(last_pose_file.read_text())
            loc.set_initial_pose(lp["x"], lp["y"], lp["yaw"])
            print(f"[定位节点] 使用上次保存的位姿作提示：x={lp['x']} y={lp['y']} yaw={lp['yaw']}（{lp.get('saved_at', '')}）", flush=True)
        else:
            print(f"[定位节点] 没有 {last_pose_file}，改为全图搜索", flush=True)

    q = queue.Queue()
    newest_scan = [0.0]

    def on_cloud(m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        newest_scan[0] = max(newest_scan[0], t)
        q.put(("scan", t, m))

    def on_odom(m):
        p, o = m.pose.pose.position, m.pose.pose.orientation
        q.put(("odom", m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, (p.x, p.y, p.z), (o.x, o.y, o.z, o.w)))

    ChannelFactoryInitialize(args.domain, args.iface)
    pub = ChannelPublisher("rt/slam_map/base_pose", PoseStamped_)
    pub.Init()
    ChannelSubscriber("rt/utlidar/cloud", PointCloud2_).Init(on_cloud, 50)
    ChannelSubscriber("rt/utlidar/robot_odom", Odometry_).Init(on_odom, 500)

    latest = {"status": "INIT", "note": "尚未收到数据"}
    lock = threading.Lock()
    init_req = []

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/pose"):
                with lock:
                    d = dict(latest)
                if "recv_wall" in d:
                    d["age_s"] = round(time.time() - d.pop("recv_wall"), 3)
                self._send(200, d)
            elif self.path.startswith("/health"):
                self._send(200, {"ok": True, "state": loc.state, "queue": q.qsize(), "dropped": loc.n_dropped})
            else:
                self._send(404, {"error": "GET /pose | GET /health | POST /initial_pose"})

        def do_POST(self):
            if not self.path.startswith("/initial_pose"):
                return self._send(404, {"error": "未知路径"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                init_req.append((float(body["x"]), float(body["y"]), float(body.get("yaw", 0.0))))
                self._send(200, {"ok": True})
            except Exception as e:  # noqa: BLE001
                self._send(400, {"ok": False, "error": str(e)})

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer((args.http_host, args.http_port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[定位节点] DDS 域 {args.domain}、网卡 {args.iface or '默认'}；HTTP http://{args.http_host}:{args.http_port}/pose", flush=True)

    logf = open(args.log, "w") if args.log else None
    n_frames, n_skip, t_print, t_save = 0, 0, time.time(), 0.0
    ms_hist = []
    try:
        while True:
            if init_req:
                x, y, yaw = init_req.pop()
                loc.set_initial_pose(x, y, yaw)
                print(f"[定位节点] 收到初始位姿提示 x={x} y={y} yaw={yaw}", flush=True)
            item = q.get()
            if item[0] == "odom":
                out = loc.add_odom(item[1], item[2], item[3])
            else:
                if newest_scan[0] - item[1] > args.max_backlog:      # 处理跟不上时丢弃积压的旧帧
                    n_skip += 1
                    continue
                xyz, trel = decode_cloud(item[2])
                out = loc.add_scan(item[1], xyz, trel)
            for r in out:
                n_frames += 1
                d = r.to_dict()
                ms_hist.append(r.ms)
                with lock:
                    latest.clear()
                    latest.update(d, recv_wall=time.time())
                if r.T_map_base is not None:
                    msg = geometry_msgs_msg_dds__PoseStamped_()
                    msg.header.stamp.sec = int(r.stamp)
                    msg.header.stamp.nanosec = int((r.stamp - int(r.stamp)) * 1e9)
                    msg.header.frame_id = "map"
                    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (float(v) for v in r.T_map_base[:3, 3])
                    qx, qy, qz, qw = d["quat_xyzw"]
                    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = qx, qy, qz, qw
                    pub.Write(msg)
                if logf:
                    logf.write(json.dumps(d, ensure_ascii=False) + "\n")
                if r.status == "TRACKING" and time.time() - t_save > 1.0:
                    t_save = time.time()
                    tmp = last_pose_file.with_suffix(".tmp")
                    tmp.write_text(json.dumps({"x": d["x"], "y": d["y"], "yaw": d["yaw"],
                                               "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}))
                    tmp.replace(last_pose_file)
            if time.time() - t_print > 1.0 and n_frames:
                t_print = time.time()
                with lock:
                    d = dict(latest)
                pos = f"x={d['x']:.2f} y={d['y']:.2f} yaw={np.degrees(d['yaw']):.0f}°" if "x" in d else "-"
                print(f"[定位节点] {d['status']:<11} {pos}  内点 {d['fitness']:.2f}  "
                      f"耗时中位数 {np.median(ms_hist[-15:]):.0f} ms  积压 {q.qsize()}  丢帧 {n_skip + loc.n_dropped}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if logf:
            logf.close()
        srv.shutdown()


if __name__ == "__main__":
    main()
