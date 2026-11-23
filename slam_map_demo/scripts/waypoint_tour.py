"""让 Go2 按点位库依次巡航所有点位，最后回到充电桩。

每一段：用 A* 在碰撞高度场上规划路线 → 沿路线行走（根据实际位置和朝向闭环修正）
→ 到点后原地转到点位朝向 → 站立停留 → 下一段。

步态沿用 unitree_mujoco/example/python/walk_go2.py 的开环小跑步态，
在其基础上加入转向：支撑期每只脚的移动 = 前进分量 + 绕机身中心转动的切线分量，
其中横向分量由髋关节侧摆实现。

两种运行方式：
    # 接 unitree_mujoco 仿真器（C++ 或 Python 版），通过 DDS 收发；加 --loop 则一圈接一圈巡航，Ctrl+C 停止
    # 加 --back-and-forth 则沿点位顺序走到底再原路返回，而不是从最后一个点直接回充电桩
    python -I waypoint_tour.py dds <地图包目录> [--loop] [--back-and-forth]
    # 不开窗口，直接在 MuJoCo 中快速跑完整个巡航，用于调参和自检
    python -I waypoint_tour.py headless <地图包目录> <场景 xml> [轨迹图输出路径]
"""
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # -I 模式下需显式加入本目录
sys.path.insert(0, str(HERE.parents[1] / "unitree_mujoco" / "example" / "python"))
import walk_go2 as gait  # noqa: E402
from nav_grid import NavGrid  # noqa: E402

gait.SWING_HEIGHT = 0.09  # 原值 0.07 m；客厅地毯边缘高 6–9 cm，抬高一些避免绊脚

DT = 0.002
V_MAX = 0.3            # m/s，与 walk_go2 的前进速度一致
W_MAX = 0.6            # rad/s，原地转身时的角速度上限
W_MAX_WALKING = 0.3    # 边走边转时的角速度上限；开环步态在高速急转时会侧翻
LOOKAHEAD = 0.35       # 沿路线向前看多远作为追踪目标
ARRIVE_XY = 0.12
ARRIVE_YAW = 0.10
TURN_IN_PLACE_ENTER = 0.4  # 朝向误差大于此值时停下原地转
TURN_IN_PLACE_EXIT = 0.25
PAUSE_TIME = 1.5
STUCK_TIMEOUT = 20.0   # 这么久没有接近目标则跳过该点位
ALIGN_TIMEOUT = 15.0   # 原地转向超过这么久仍未对准，则记录误差后继续
REAPPROACH_DIST = 0.30  # 转向时被带离点位超过此距离，重新走回去
V_ALIGN = 0.1          # 原地转向时用于前后修正位置的速度上限
YAW_FILTER_TIME = 0.4  # 朝向滤波时间常数，等于一个步态周期，滤掉小跑时机身的左右摆动
HEADING_GAIN = 1.0     # 朝向误差到转向角速度的比例

# 足端相对机身中心的位置 (x, y)，按电机顺序 FR、FL、RR、RL（来自 go2.xml 的髋、大腿偏移）
FOOT_XY = np.array([[0.1934, -0.142], [0.1934, 0.142], [-0.1934, -0.142], [-0.1934, 0.142]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def gait_joint_pos(t, vx, wz, swing_scale):
    """walk_go2.GaitJointPos 的扩展：机身以前进速度 vx、转向角速度 wz 运动时各关节的目标角度。

    支撑脚相对机身的速度为 -(v + w × r)，walk_go2 的足端轨迹参数“speed”取其相反数。
    """
    q = np.zeros(12)
    tau = np.zeros(12)
    for leg in range(4):
        rx, ry = FOOT_XY[leg]
        stride_x = (vx - wz * ry) * gait.GAIT_PERIOD / 2
        stride_y = (wz * rx) * gait.GAIT_PERIOD / 2
        phase = (t / gait.GAIT_PERIOD + gait.LEG_PHASE_OFFSET[leg]) % 1.0
        x, z = gait.FootTrajectory(phase, stride_x)
        y, _ = gait.FootTrajectory(phase, stride_y)
        z = -gait.BODY_HEIGHT + (z + gait.BODY_HEIGHT) * swing_scale
        # 髋关节绕 x 轴正转把脚推向 +y；侧摆后矢状面内的腿长变为 sqrt(y² + z²)
        hip = np.arctan2(y, -z)
        thigh, calf = gait.LegIK(x, -np.hypot(y, z))
        q[3 * leg: 3 * leg + 3] = [hip, thigh, calf]
        stance_share = 0.5 if phase < 0.5 else 0.0
        share = swing_scale * stance_share + (1 - swing_scale) * 0.25
        tau[3 * leg + 1: 3 * leg + 3] = gait.StanceTorque(thigh, calf, share * gait.ROBOT_MASS * gait.GRAVITY)
    return q, tau


def gait_command(t, vx, wz, swing_scale, eps=1e-3):
    q, tau = gait_joint_pos(t, vx, wz, swing_scale)
    dq = (gait_joint_pos(t + eps, vx, wz, swing_scale)[0]
          - gait_joint_pos(t - eps, vx, wz, swing_scale)[0]) / (2 * eps)
    return q, dq, tau


class TourController:
    """输入当前位姿，输出 12 个电机的 (q, dq, kp, kd, tau)。"""

    def __init__(self, grid, tour, log=print):
        self.grid = grid
        self.tour = tour          # [(id, 名称, xy, yaw), ...]
        self.log = log
        self.leg = 0
        self.state = "stand_up"
        self.state_time = 0.0
        self.time = 0.0
        self.gait_time = 0.0
        self.swing_scale = 0.0
        self.vx = self.wz = 0.0
        self.yaw_filtered = None
        self.turning = False
        self.path = None
        self.start_q = None
        self.best_dist = np.inf
        self.last_progress = 0.0
        self.results = []

    def _goto(self, state):
        self.state, self.state_time = state, 0.0

    def _begin_leg(self, xy):
        wp_id, name, goal, _ = self.tour[self.leg]
        self.path = self.grid.plan(xy, goal)
        if self.path is None:
            self.log(f"[跳过] {wp_id} {name}：规划不出路线")
            self.results.append((wp_id, name, None, None, None))
            self._next_leg()
            return
        self.path_index = 0
        self.leg_start = self.time
        self.best_dist, self.last_progress = np.inf, self.time
        length = sum(np.linalg.norm(b - a) for a, b in zip(self.path[:-1], self.path[1:]))
        self.log(f"[出发] 前往 {wp_id} {name}，路线 {length:.1f} m，{len(self.path) - 1} 段折线")
        self._goto("follow")

    def restart_tour(self):
        """从第一段重新开始巡航；机器狗此时已经站着，跳过起身。"""
        self.leg = 0
        self.results = []
        self._goto("plan")

    def _next_leg(self):
        self.leg += 1
        if self.leg >= len(self.tour):
            self._goto("done")
        else:
            self._goto("plan")

    def _lookahead_point(self, xy):
        # 沿折线前进：已经接近的拐点就跳过
        while (self.path_index < len(self.path) - 1
               and np.linalg.norm(self.path[self.path_index] - xy) < LOOKAHEAD):
            self.path_index += 1
        return self.path[self.path_index]

    def _remaining_path(self, xy):
        """沿规划路线还剩多长；绕路时到终点的直线距离会先变大，不能用它判断进展。"""
        rest = self.path[self.path_index:]
        return np.linalg.norm(rest[0] - xy) + sum(np.linalg.norm(b - a) for a, b in zip(rest[:-1], rest[1:]))

    def _velocity_command(self, xy, yaw):
        wp_id, name, goal, goal_yaw = self.tour[self.leg]
        dist = np.linalg.norm(goal - xy)
        if self.state == "follow":
            remaining = self._remaining_path(xy)
            if remaining < self.best_dist - 0.05:
                self.best_dist, self.last_progress = remaining, self.time
            if self.time - self.last_progress > STUCK_TIMEOUT:
                self.log(f"[跳过] {wp_id} {name}：{STUCK_TIMEOUT:.0f} 秒没有进展，路线剩余 {remaining:.2f} m")
                self.results.append((wp_id, name, dist, None, None))
                self._next_leg()
                return 0.0, 0.0
            if dist < ARRIVE_XY:
                self._goto("align")
                return 0.0, 0.0
            target = self._lookahead_point(xy)
            err = wrap(np.arctan2(*(target - xy)[::-1]) - yaw)
            if abs(err) > TURN_IN_PLACE_ENTER:
                self.turning = True
            elif abs(err) < TURN_IN_PLACE_EXIT:
                self.turning = False
            if self.turning:
                return 0.0, np.clip(HEADING_GAIN * err, -W_MAX, W_MAX)
            wz = np.clip(HEADING_GAIN * err, -W_MAX_WALKING, W_MAX_WALKING)
            # 转得越急走得越慢；接近终点时减速，避免冲过头
            vx = V_MAX * np.cos(err) * (1 - 0.6 * abs(wz) / W_MAX_WALKING) * np.clip(dist / 0.5, 0.4, 1.0)
            return vx, wz
        # align：原地转到点位要求的朝向，同时用小幅前后移动把位置拉回点位
        err = wrap(goal_yaw - yaw)
        if dist > REAPPROACH_DIST:
            self._goto("follow")
            self.path, self.path_index = [goal], 0
            return 0.0, 0.0
        if (abs(err) < ARRIVE_YAW and dist < ARRIVE_XY) or self.state_time > ALIGN_TIMEOUT:
            self.results.append((wp_id, name, dist, abs(err), self.time - self.leg_start))
            self.log(f"[到达] {wp_id} {name}：位置误差 {dist:.3f} m，朝向误差 {np.degrees(abs(err)):.1f}°，"
                     f"用时 {self.time - self.leg_start:.1f} s")
            self._goto("pause")
            return 0.0, 0.0
        # 横向平移的步态不稳定（会倒退并侧倾），这里只沿机身前后方向修正位置
        forward = np.cos(yaw) * (goal - xy)[0] + np.sin(yaw) * (goal - xy)[1]
        vx = np.clip(forward, -V_ALIGN, V_ALIGN)
        return vx, np.clip(HEADING_GAIN * err, -W_MAX, W_MAX)

    def step(self, xy, yaw, motor_q):
        """推进一个控制周期，返回 (q, dq, kp, kd, tau)，均为电机顺序。"""
        self.time += DT
        self.state_time += DT
        if self.yaw_filtered is None:
            self.yaw_filtered = yaw
        self.yaw_filtered = wrap(self.yaw_filtered + DT / YAW_FILTER_TIME * wrap(yaw - self.yaw_filtered))
        stand_q, stand_tau = gait.StandJointPos(), gait.StandTorque()

        if self.state == "stand_up":
            if self.start_q is None:
                self.start_q = np.array(motor_q)
            phase = np.tanh(self.state_time / 0.6)
            q = phase * stand_q + (1 - phase) * self.start_q
            if self.state_time > gait.STAND_UP_TIME + gait.HOLD_TIME:
                self._goto("plan")
            kp = phase * gait.KP_STAND + (1 - phase) * 20.0
            return q, np.zeros(12), kp, gait.KD_STAND, phase * stand_tau
        if self.state == "done":
            return stand_q, np.zeros(12), gait.KP_STAND, gait.KD_STAND, stand_tau

        if self.state == "plan":
            self._begin_leg(xy)
        if self.state in ("follow", "align"):
            vx_cmd, wz_cmd = self._velocity_command(xy, self.yaw_filtered)
            moving = True
        else:  # pause：踏步渐停为站立，停够后进入下一段
            vx_cmd = wz_cmd = 0.0
            moving = False
            if self.state_time > PAUSE_TIME:
                self._next_leg()

        # 抬脚幅度 1 秒内渐变；速度限制加速度，避免步态突变
        target_scale = 1.0 if moving else 0.0
        self.swing_scale += np.clip(target_scale - self.swing_scale, -DT, DT)
        if self.swing_scale < 0.9:
            vx_cmd = wz_cmd = 0.0
        self.vx += np.clip(vx_cmd - self.vx, -0.5 * DT, 0.5 * DT)
        self.wz += np.clip(wz_cmd - self.wz, -1.5 * DT, 1.5 * DT)

        self.gait_time += DT
        q, dq, tau = gait_command(self.gait_time, self.vx, self.wz, self.swing_scale)
        return q, dq, gait.KP_WALK, gait.KD_WALK, tau


def load_tour(map_dir, back_and_forth=False):
    waypoints = yaml.safe_load(open(Path(map_dir) / "waypoints.yaml"))["waypoints"]
    tour = [(wp["id"], wp["name"], np.array([wp["stand_pose"]["x"], wp["stand_pose"]["y"]]),
             wp["stand_pose"]["yaw"]) for wp in waypoints]
    if back_and_forth:
        # 按点位顺序走到最后一个再原路返回第一个：2 → … → N → … → 1；
        # 返程经过的中间点位朝向取反，避免每到一点都转身两次
        back = [(i, n, xy, wrap(yaw + np.pi)) for i, n, xy, yaw in tour[1:-1][::-1]]
        return tour[1:] + back + tour[:1]
    # 从充电桩出发，依次访问其余点位，最后回到充电桩
    return tour[1:] + tour[:1]


def yaw_from_quat(w, x, y, z):
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run_headless(map_dir, scene_path, image_path=None, back_and_forth=False, tour=None):
    """tour：自定义路线 [(编号, 名称, xy, 朝向), ...]，缺省时按点位库巡航（场景图导航 navigate_to.py 用）。"""
    import mujoco

    model = mujoco.MjModel.from_xml_path(scene_path)
    model.opt.timestep = DT
    data = mujoco.MjData(model)
    joints = model.actuator_trnid[:, 0]
    qadr, dadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    pos_adr = model.sensor_adr[model.sensor("frame_pos").id]
    quat_adr = model.sensor_adr[model.sensor("imu_quat").id]

    # 与仿真器启动时一样：先自然落地
    for _ in range(int(1.0 / DT)):
        mujoco.mj_step(model, data)

    controller = TourController(NavGrid.load(Path(map_dir) / "mujoco" / "collision_hfield.npz"),
                                tour if tour is not None else load_tour(map_dir, back_and_forth))
    trajectory, max_tilt, min_dist = [], 0.0, 0.0
    while controller.state != "done" and controller.time < 900:
        xy = data.sensordata[pos_adr:pos_adr + 2].copy()
        yaw = yaw_from_quat(*data.sensordata[quat_adr:quat_adr + 4])
        q, dq, kp, kd, tau = controller.step(xy, yaw, data.qpos[qadr])
        # 与 unitree_sdk2 桥接相同的电机模型
        data.ctrl[:] = tau + kp * (q - data.qpos[qadr]) + kd * (dq - data.qvel[dadr])
        mujoco.mj_step(model, data)
        if int(controller.time / DT) % 50 == 0:
            trajectory.append(xy)
            w, x, y, z = data.sensordata[quat_adr:quat_adr + 4]
            tilt = np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1)))
            max_tilt = max(max_tilt, tilt if controller.state != "stand_up" else 0.0)
        if data.ncon:
            min_dist = min(min_dist, data.contact.dist[:data.ncon].min())
        w, x, y, z = data.qpos[3:7]
        if controller.state != "stand_up" and (1 - 2 * (x * x + y * y)) < 0.5:  # 倾斜超过 60°
            print(f"[摔倒] t={controller.time:.1f}s 位置 ({data.qpos[0]:+.2f}, {data.qpos[1]:+.2f})")
            break

    reached = [r for r in controller.results if r[3] is not None]
    print(f"\n巡航结束：用时 {controller.time:.0f} s，到达 {len(reached)}/{len(controller.tour)} 个点位，"
          f"行走中最大倾斜 {max_tilt:.1f}°，最深穿透 {-min_dist:.3f} m")
    if image_path:
        draw_trajectory(controller.grid, controller.tour, np.array(trajectory), image_path)
    return len(reached) == len(controller.tour)


def draw_trajectory(grid, tour, trajectory, image_path):
    import cv2

    scale = 5
    img = np.full(grid.heights.shape + (3,), 240, np.uint8)
    img[grid.heights > 0.03] = (170, 200, 230)
    img[grid.clearance <= 0] = (70, 70, 70)
    img = cv2.resize(np.flipud(img), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    h = img.shape[0]

    def px(xy):
        c, r = (np.asarray(xy) - grid.lo) / grid.res * scale
        return int(c), int(h - r)

    for a, b in zip(trajectory[:-1], trajectory[1:]):
        cv2.line(img, px(a), px(b), (0, 0, 220), 2)
    for i, (wp_id, name, xy, yaw) in enumerate(tour):
        cv2.circle(img, px(xy), 7, (0, 160, 0), -1)
        tip = xy + 0.3 * np.array([np.cos(yaw), np.sin(yaw)])
        cv2.line(img, px(xy), px(tip), (0, 100, 0), 2)
        cv2.putText(img, wp_id[-2:], (px(xy)[0] + 8, px(xy)[1] - 6), 0, 0.45, (0, 0, 0), 1)
    cv2.imwrite(str(image_path), img)
    print(f"轨迹图已保存到 {image_path}")


def run_dds(map_dir, loop=False, back_and_forth=False, tour=None, hold=False, *, domain=1):
    """hold：走完后保持站立直到 Ctrl+C（不加则站 3 秒后退出，退出后机器狗会趴下）。"""
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
    from unitree_sdk2py.utils.crc import CRC

    ChannelFactoryInitialize(domain, "lo")
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    ChannelSubscriber("rt/lowstate", LowState_).Init(gait.LowStateHandler, 10)
    ChannelSubscriber("rt/sportmodestate", SportModeState_).Init(gait.SportStateHandler, 10)

    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0], cmd.head[1] = 0xFE, 0xEF
    cmd.level_flag = 0xFF
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01
    crc = CRC()

    waited = 0.0
    while gait.low_state is None or gait.sport_state is None:
        if waited >= 10.0:
            print("10 秒没有收到仿真器的状态消息：仿真器可能没有运行（或窗口已关闭），请先启动 unitree_mujoco", flush=True)
            sys.exit(1)
        print("等待 rt/lowstate 和 rt/sportmodestate ...", flush=True)
        time.sleep(0.5)
        waited += 0.5

    controller = TourController(NavGrid.load(Path(map_dir) / "mujoco" / "collision_hfield.npz"),
                                tour if tour is not None else load_tour(map_dir, back_and_forth),
                                log=lambda s: print(s, flush=True))
    last_print = 0.0
    lap, lap_start = 1, 0.0
    try:
        while True:
            step_start = time.perf_counter()
            low, sport = gait.low_state, gait.sport_state
            xy = np.array(sport.position[:2])
            w, x, y, z = low.imu_state.quaternion
            if controller.state != "stand_up" and 1 - 2 * (x * x + y * y) < 0.5:  # 倾斜超过 60°
                print(f"[摔倒] 第 {lap} 圈，位置 ({xy[0]:+.2f}, {xy[1]:+.2f})；请重启仿真器后再运行", flush=True)
                return
            yaw = yaw_from_quat(w, x, y, z)
            motor_q = [low.motor_state[i].q for i in range(12)]
            q, dq, kp, kd, tau = controller.step(xy, yaw, motor_q)
            gait.SetCmd(cmd, q, kp, kd, dq, tau)
            cmd.crc = crc.Crc(cmd)
            pub.Write(cmd)

            if controller.state == "done":
                reached = [r for r in controller.results if r[3] is not None]
                errors = [r[2] for r in reached]
                print(f"[{f'第 {lap} 圈' if loop else '全程'}结束] 用时 {controller.time - lap_start:.0f} s，"
                      f"到达 {len(reached)}/{len(controller.tour)} 个点位，"
                      f"最大位置误差 {max(errors) if errors else float('nan'):.3f} m", flush=True)
                if not loop:
                    if not hold:
                        break
                    ok = len(reached) == len(controller.tour)
                    print(("已到达全部目标" if ok else f"有 {len(controller.tour) - len(reached)} 个目标没有到达（见上方[跳过]）")
                          + "，保持站立；按 Ctrl+C 退出", flush=True)
                    while True:  # done 状态下控制器输出站立指令
                        q, dq, kp, kd, tau = controller.step(xy, yaw, [low.motor_state[i].q for i in range(12)])
                        gait.SetCmd(cmd, q, kp, kd, dq, tau)
                        cmd.crc = crc.Crc(cmd)
                        pub.Write(cmd)
                        time.sleep(DT)
                lap, lap_start = lap + 1, controller.time
                print(f"[第 {lap} 圈开始]", flush=True)
                controller.restart_tour()

            if controller.time - last_print >= 2.0:
                last_print = controller.time
                rpy = gait.QuatToRpy(low.imu_state.quaternion)
                print(f"t={controller.time:5.1f}s [{controller.state:8s}] pos=({xy[0]:+.2f}, {xy[1]:+.2f}) "
                      f"yaw={np.degrees(yaw):+6.1f} roll={rpy[0]:+5.1f} pitch={rpy[1]:+5.1f}", flush=True)

            remaining = DT - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，站稳后退出", flush=True)

    # 保持站立几秒，让仿真器里的机器狗不至于在程序退出后立刻趴下
    stand_q, stand_tau = gait.StandJointPos(), gait.StandTorque()
    for _ in range(int(3.0 / DT)):
        gait.SetCmd(cmd, stand_q, gait.KP_STAND, gait.KD_STAND, tau=stand_tau)
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)
        time.sleep(DT)


if __name__ == "__main__":
    back_and_forth = "--back-and-forth" in sys.argv
    args = [a for a in sys.argv if a != "--back-and-forth"]
    if args[1] == "dds":
        run_dds(args[2], loop="--loop" in args[3:], back_and_forth=back_and_forth)
    else:
        ok = run_headless(args[2], args[3], args[4] if len(args) > 4 else None, back_and_forth)
        sys.exit(0 if ok else 1)
