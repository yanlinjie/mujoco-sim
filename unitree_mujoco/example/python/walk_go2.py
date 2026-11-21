import time
import sys
import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, SportModeState_
from unitree_sdk2py.utils.crc import CRC

# Stand up, trot forward with an open-loop gait, then stop and stand still.
# Motor order: FR, FL, RR, RL, each leg (hip, thigh, calf).

L_THIGH = 0.213
L_CALF = 0.213
ROBOT_MASS = 15.2  # kg
GRAVITY = 9.81
BODY_HEIGHT = 0.32  # foot depth below thigh joint (m)

GAIT_PERIOD = 0.4  # s, one full trot cycle
SWING_HEIGHT = 0.07  # m
FORWARD_SPEED = 0.3  # m/s
# Diagonal pairs (FR+RL, FL+RR) are half a cycle apart
LEG_PHASE_OFFSET = [0.0, 0.5, 0.5, 0.0]

KP_STAND, KD_STAND = 50.0, 3.5
KP_WALK, KD_WALK = 60.0, 2.0

STAND_UP_TIME = 2.0
HOLD_TIME = 1.0
STEP_IN_PLACE_TIME = 1.0
WALK_TIME = 4.0
STOP_TIME = 2.0

dt = 0.002
crc = CRC()

low_state = None
sport_state = None


def LowStateHandler(msg: LowState_):
    global low_state
    low_state = msg


def SportStateHandler(msg: SportModeState_):
    global sport_state
    sport_state = msg


def LegIK(x, z):
    """Foot position (x forward, z up) relative to the thigh joint -> (thigh, calf)."""
    d2 = x * x + z * z
    cos_knee = (d2 - L_THIGH**2 - L_CALF**2) / (2 * L_THIGH * L_CALF)
    calf = -np.arccos(np.clip(cos_knee, -1.0, 1.0))
    thigh = np.arctan2(-x, -z) - calf / 2  # valid because thigh and calf are equal length
    return thigh, calf


def StanceTorque(thigh, calf, foot_force):
    """Joint torques (thigh, calf) that press the foot down with foot_force (N), via J^T * f."""
    # Foot position: x = -L1 sin(a) - L2 sin(a+b), z = -L1 cos(a) - L2 cos(a+b)
    # Only the z row of the Jacobian matters for a vertical force (0, -foot_force)
    dz_dthigh = L_THIGH * np.sin(thigh) + L_CALF * np.sin(thigh + calf)
    dz_dcalf = L_CALF * np.sin(thigh + calf)
    return -foot_force * dz_dthigh, -foot_force * dz_dcalf


def FootTrajectory(phase, stride):
    """phase in [0, 1): first half stance (foot sweeps back), second half swing."""
    if phase < 0.5:
        s = phase / 0.5
        return stride / 2 - stride * s, -BODY_HEIGHT
    s = (phase - 0.5) / 0.5
    x = -stride / 2 + stride * (s - np.sin(2 * np.pi * s) / (2 * np.pi))
    z = -BODY_HEIGHT + SWING_HEIGHT * np.sin(np.pi * s)
    return x, z


def GaitJointPos(t, speed, swing_height_scale):
    stride = speed * GAIT_PERIOD / 2
    q = np.zeros(12)
    tau = np.zeros(12)
    for leg in range(4):
        phase = (t / GAIT_PERIOD + LEG_PHASE_OFFSET[leg]) % 1.0
        x, z = FootTrajectory(phase, stride)
        z = -BODY_HEIGHT + (z + BODY_HEIGHT) * swing_height_scale
        thigh, calf = LegIK(x, z)
        q[3 * leg: 3 * leg + 3] = [0.0, thigh, calf]
        # Weight is shared by the two stance legs; blend in while stepping in place
        stance_share = 0.5 if phase < 0.5 else 0.0
        share = swing_height_scale * stance_share + (1 - swing_height_scale) * 0.25
        tau[3 * leg + 1: 3 * leg + 3] = StanceTorque(thigh, calf, share * ROBOT_MASS * GRAVITY)
    return q, tau


def GaitCommand(t, speed, swing_height_scale, eps=1e-3):
    q, tau = GaitJointPos(t, speed, swing_height_scale)
    dq = (GaitJointPos(t + eps, speed, swing_height_scale)[0]
          - GaitJointPos(t - eps, speed, swing_height_scale)[0]) / (2 * eps)
    return q, dq, tau


def StandJointPos():
    thigh, calf = LegIK(0.0, -BODY_HEIGHT)
    return np.array([0.0, thigh, calf] * 4)


def StandTorque():
    thigh, calf = LegIK(0.0, -BODY_HEIGHT)
    return np.array([0.0, *StanceTorque(thigh, calf, 0.25 * ROBOT_MASS * GRAVITY)] * 4)


def QuatToRpy(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees([roll, pitch, yaw])


def SetCmd(cmd, q, kp, kd, dq=None, tau=None):
    for i in range(12):
        cmd.motor_cmd[i].q = float(q[i])
        cmd.motor_cmd[i].kp = kp
        cmd.motor_cmd[i].dq = 0.0 if dq is None else float(dq[i])
        cmd.motor_cmd[i].kd = kd
        cmd.motor_cmd[i].tau = 0.0 if tau is None else float(tau[i])


if __name__ == '__main__':

    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    ChannelSubscriber("rt/lowstate", LowState_).Init(LowStateHandler, 10)
    ChannelSubscriber("rt/sportmodestate", SportModeState_).Init(SportStateHandler, 10)

    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode

    while low_state is None:
        print("Waiting for rt/lowstate ...")
        time.sleep(0.5)

    start_q = np.array([low_state.motor_state[i].q for i in range(12)])
    stand_q = StandJointPos()
    stand_tau = StandTorque()

    t_hold = STAND_UP_TIME
    t_step = t_hold + HOLD_TIME
    t_walk = t_step + STEP_IN_PLACE_TIME
    t_stop = t_walk + WALK_TIME
    t_end = t_stop + STOP_TIME

    running_time = 0.0
    gait_time = 0.0
    last_print = -1.0
    while running_time < t_end:
        step_start = time.perf_counter()
        running_time += dt

        if running_time < t_hold:
            phase = np.tanh(running_time / 0.6)
            q = phase * stand_q + (1 - phase) * start_q
            SetCmd(cmd, q, phase * KP_STAND + (1 - phase) * 20.0, KD_STAND, tau=phase * stand_tau)
            stage = "stand up"
        elif running_time < t_step:
            SetCmd(cmd, stand_q, KP_STAND, KD_STAND, tau=stand_tau)
            stage = "hold"
        elif running_time < t_stop:
            gait_time += dt
            if running_time < t_walk:
                # Fade in foot lift while stepping in place
                ramp = (running_time - t_step) / STEP_IN_PLACE_TIME
                q, dq, tau = GaitCommand(gait_time, 0.0, ramp)
                stage = "step in place"
            else:
                ramp = min((running_time - t_walk) / 1.0, 1.0)
                q, dq, tau = GaitCommand(gait_time, ramp * FORWARD_SPEED, 1.0)
                stage = "walk"
            SetCmd(cmd, q, KP_WALK, KD_WALK, dq, tau)
        else:
            SetCmd(cmd, stand_q, KP_STAND, KD_STAND, tau=stand_tau)
            stage = "stop"

        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

        if running_time - last_print >= 0.5 and sport_state is not None:
            last_print = running_time
            pos = sport_state.position
            rpy = QuatToRpy(low_state.imu_state.quaternion)
            print(f"t={running_time:4.1f}s [{stage:13s}] "
                  f"pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:.3f}) "
                  f"roll={rpy[0]:+5.1f} pitch={rpy[1]:+5.1f} yaw={rpy[2]:+5.1f}")

        time_until_next_step = dt - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
