"""读取 unitree_mujoco 仿真器中 Go2 的当前位姿，打印一行 JSON：{"ok": true, "x": .., "y": .., "yaw": .., "tilt_deg": ..}

坐标即地图坐标系（场景由 map_to_mujoco.py 生成，世界原点 = 地图原点）。只读，不发控制指令，
可以在 follow_path.py / nav_demo.py 运行时同时调用。仿真器没运行时 3 秒后输出 {"ok": false, ...}，退出码 1。

用法：$P -I scripts/sim_pose.py
"""
import json
import sys
import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_

state = {}


def main():
    ChannelFactoryInitialize(1, "lo")
    ChannelSubscriber("rt/sportmodestate", SportModeState_).Init(lambda m: state.__setitem__("sport", m), 10)
    ChannelSubscriber("rt/lowstate", LowState_).Init(lambda m: state.__setitem__("low", m), 10)
    for _ in range(30):
        if "sport" in state and "low" in state:
            break
        time.sleep(0.1)
    else:
        print(json.dumps({"ok": False, "error": "3 秒没有收到仿真器状态，仿真器可能没有运行"}, ensure_ascii=False))
        sys.exit(1)
    x, y = state["sport"].position[:2]
    w, qx, qy, qz = state["low"].imu_state.quaternion
    yaw = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (qx * qx + qy * qy), -1, 1)))
    print(json.dumps({"ok": True, "x": round(float(x), 3), "y": round(float(y), 3),
                      "yaw": round(float(yaw), 4), "tilt_deg": round(float(tilt), 1)}))


if __name__ == "__main__":
    main()
