"""本机验证用：把录制数据按真实时间通过 DDS 发出，话题名、消息类型与 Go2 真机一致（rt/utlidar/cloud、rt/utlidar/robot_odom），
供 localize_dds.py 订阅。数据来自 slam_data 抽取的中间文件（unitree-mujoco 环境没有 rosbags，不直接读包）：
  l1_scans.npz（extract_l1.py：每帧点、逐点 time、帧头时间；已去掉机身点，定位节点会再滤一次，不影响）
  leg_odom.npz（calib_lidar_base.py：/utlidar/robot_odom 的 t, p, q）

用法: dds_replay.py <l1_scans.npz> <leg_odom.npz> [--start 秒] [--duration 秒] [--iface lo] [--domain 1] [--rate 1.0]
"""
import argparse
import heapq
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("odom")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--domain", type=int, default=1)
    ap.add_argument("--rate", type=float, default=1.0, help="回放倍速")
    args = ap.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
    from unitree_sdk2py.idl.default import (nav_msgs_msg_dds__Odometry_, std_msgs_msg_dds__Header_,
                                            builtin_interfaces_msgs_msg_dds__Time_)
    from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_, PointField_

    z = np.load(args.scans)
    d = {k: z[k] for k in ("lens", "stamps", "points", "intensity", "time")}   # npz 每次取键都会重新解压，先读出来
    lens, stamps = d["lens"], d["stamps"]
    off = np.r_[0, np.cumsum(lens)]
    lo = dict(np.load(args.odom))
    t0 = stamps[0] + args.start
    t1 = t0 + args.duration
    events = []
    for k in np.flatnonzero((stamps >= t0) & (stamps < t1)):
        t_end = stamps[k] + (d["time"][off[k]:off[k + 1]].max() if lens[k] else 0)
        events.append((t_end + 0.005, 1, int(k)))       # 激光帧在帧结束后才发出
    for i in np.flatnonzero((lo["t"] >= t0 - 1.0) & (lo["t"] < t1)):
        events.append((lo["t"][i], 0, int(i)))
    heapq.heapify(events)

    ChannelFactoryInitialize(args.domain, args.iface)
    pc = ChannelPublisher("rt/utlidar/cloud", PointCloud2_)
    pc.Init()
    po = ChannelPublisher("rt/utlidar/robot_odom", Odometry_)
    po.Init()
    fields = [PointField_(n, 4 * i, 7, 1) for i, n in enumerate(["x", "y", "z", "intensity", "time"])]

    def stamp_of(t):
        h = std_msgs_msg_dds__Header_()
        h.stamp = builtin_interfaces_msgs_msg_dds__Time_()
        h.stamp.sec, h.stamp.nanosec = int(t), int((t - int(t)) * 1e9)
        return h

    print(f"[回放] {len(events)} 条消息，{args.duration:.0f} s，DDS 域 {args.domain} 网卡 {args.iface}", flush=True)
    time.sleep(2.0)                                     # 等订阅端发现
    t_data0 = events[0][0]
    wall0 = time.time()
    n_scan = 0
    while events:
        t, kind, i = heapq.heappop(events)
        lag = (t - t_data0) / args.rate - (time.time() - wall0)
        if lag > 0:
            time.sleep(lag)
        if kind == 0:
            m = nav_msgs_msg_dds__Odometry_()
            m.header = stamp_of(lo["t"][i])
            m.header.frame_id = "odom"
            p, q = lo["p"][i], lo["q"][i]
            m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = map(float, p)
            m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z, m.pose.pose.orientation.w = map(float, q)
            po.Write(m)
        else:
            s, e = off[i], off[i + 1]
            arr = np.column_stack([d["points"][s:e], d["intensity"][s:e], d["time"][s:e]]).astype("<f4")
            h = stamp_of(stamps[i])
            h.frame_id = "utlidar_lidar"
            m = PointCloud2_(h, 1, len(arr), fields, False, 20, 20 * len(arr), arr.tobytes(), True)
            pc.Write(m)
            n_scan += 1
            if n_scan % 150 == 0:
                print(f"[回放] 已发 {n_scan} 帧，数据时间 {t - t_data0:.0f} s", flush=True)
    print("[回放] 结束", flush=True)


if __name__ == "__main__":
    main()
