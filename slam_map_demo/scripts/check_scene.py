"""对生成的 MuJoCo 场景做自动检查，并渲染截图。

1. 高度场方向：在随机位置向下打射线，与点云栅格高度比较，确认行列没有颠倒。
2. 默认出生：模拟 unitree_mujoco 启动时的状态（原点、腿伸直、无控制），确认不会被弹飞。
3. 点位复用：把 Go2 以站立姿态放到每个点位，关节用 PD 保持 3 秒，
   检查机身离地高度、水平漂移、朝向，以及站稳后是否有穿模（接触距离明显为负）。

用法：
    MUJOCO_GL=egl python -I check_scene.py <场景 xml> <地图包目录> <截图输出目录>
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from PIL import Image

STAND_JOINTS = np.array([0, 0.9, -1.8] * 4)
STAND_HEIGHT = 0.30
KP, KD = 50.0, 3.5
HOLD_SECONDS = 3.0
SETTLE_SECONDS = 2.0  # 穿透量只统计这之后的时间段
COLLISION_GROUP = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
MAP_GEOM = "map_collision"


def ground_height(model, data, x, y):
    """机身周围 0.3 m 范围内地图碰撞体的最高点。"""
    map_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, MAP_GEOM)
    geomid = np.zeros(1, dtype=np.int32)
    best = 0.0
    for dx in (-0.3, 0, 0.3):
        for dy in (-0.3, 0, 0.3):
            dist = mujoco.mj_ray(model, data, np.array([x + dx, y + dy, 3.0]),
                                 np.array([0, 0, -1.0]), COLLISION_GROUP, 1, -1, geomid)
            if dist >= 0 and geomid[0] == map_id:
                best = max(best, 3.0 - dist)
    return best


def check_hfield_orientation(model, data, map_dir):
    info = np.load(Path(map_dir) / "mujoco" / "collision_hfield.npz")
    heights, lo, res = info["heights"], info["lo"], float(info["resolution"])
    mujoco.mj_forward(model, data)
    rng = np.random.default_rng(0)
    rows, cols = np.nonzero(heights > 0.3)
    pick = rng.choice(len(rows), 200, replace=False)
    errors = []
    geomid = np.zeros(1, dtype=np.int32)
    for r, c in zip(rows[pick], cols[pick]):
        x, y = lo[0] + c * res, lo[1] + r * res
        dist = mujoco.mj_ray(model, data, np.array([x, y, 3.0]), np.array([0, 0, -1.0]),
                             COLLISION_GROUP, 1, -1, geomid)
        errors.append(abs((3.0 - dist) - heights[r, c]))
    errors = np.array(errors)
    ok = np.median(errors) < 0.05
    print(f"[高度场方向] 200 个障碍栅格的射线高度误差中位数 {np.median(errors):.3f} m -> {'通过' if ok else '失败'}")
    return ok


def place_robot(model, data, x, y, yaw):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    ground = ground_height(model, data, x, y)
    data.qpos[:3] = [x, y, ground + STAND_HEIGHT + 0.02]
    data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    data.qpos[7:19] = STAND_JOINTS
    mujoco.mj_forward(model, data)
    return ground


def hold_and_measure(model, data):
    # 执行器顺序是 FR、FL、RR、RL，与 qpos 中关节顺序不同，需按执行器实际驱动的关节取值
    joints = model.actuator_trnid[:, 0]
    qadr, dadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    min_dist = 0.0
    for _ in range(int(HOLD_SECONDS / model.opt.timestep)):
        data.ctrl[:] = KP * (STAND_JOINTS - data.qpos[qadr]) - KD * data.qvel[dadr]
        mujoco.mj_step(model, data)
        if data.time > SETTLE_SECONDS and data.ncon:
            min_dist = min(min_dist, data.contact.dist[:data.ncon].min())
    return min_dist


def render(model, data, path, lookat, distance, azimuth, elevation, show_collision=False):
    renderer = mujoco.Renderer(model, 960, 1280)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = show_collision
    opt.geomgroup[1] = not show_collision
    renderer.update_scene(data, cam, opt)
    Image.fromarray(renderer.render()).save(path)
    renderer.close()


def main(scene_path, map_dir, image_dir):
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(scene_path)
    model.opt.timestep = 0.005  # 与 unitree_mujoco 的 SIMULATE_DT 一致
    data = mujoco.MjData(model)

    all_ok = check_hfield_orientation(model, data, map_dir)

    mujoco.mj_resetData(model, data)
    max_speed = 0.0
    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        max_speed = max(max_speed, np.linalg.norm(data.qvel[:3]))
    ok = max_speed < 3.0 and np.hypot(*data.qpos[:2]) < 0.3
    all_ok &= ok
    print(f"[默认出生] 2 秒内机身最大速度 {max_speed:.2f} m/s，最终位置 {data.qpos[:3].round(3).tolist()} -> {'通过' if ok else '失败'}")

    waypoints = yaml.safe_load(open(Path(map_dir) / "waypoints.yaml"))["waypoints"]
    for wp in waypoints:
        p = wp["stand_pose"]
        ground = place_robot(model, data, p["x"], p["y"], p["yaw"])
        min_dist = hold_and_measure(model, data)
        base = data.qpos[:3]
        drift = np.hypot(base[0] - p["x"], base[1] - p["y"])
        w, z = data.qpos[3], data.qpos[6]
        yaw_err = abs((2 * np.arctan2(z, w) - p["yaw"] + np.pi) % (2 * np.pi) - np.pi)
        ok = abs(base[2] - ground - STAND_HEIGHT) < 0.06 and drift < 0.10 and yaw_err < 0.10 and min_dist > -0.02
        all_ok &= ok
        print(f"[点位 {wp['id']} {wp['name']}] 地面高度 {ground:.3f} m，机身离地 {base[2] - ground:.3f} m，水平漂移 {drift:.3f} m，"
              f"朝向误差 {yaw_err:.3f} rad，最深穿透 {-min_dist:.3f} m -> {'通过' if ok else '失败'}")
        render(model, data, image_dir / f"{wp['id']}.png", base, 2.5, 180 + np.degrees(p["yaw"]), -25)

    place_robot(model, data, 0, 0, 0)
    map_geom = model.geom(MAP_GEOM)
    center = map_geom.pos
    span = 2.6 * map_geom.size[:2].max()  # geom.size 对高度场即 hfield 的半边长
    render(model, data, image_dir / "overview_visual.png", center, span, 90, -89)
    render(model, data, image_dir / "overview_collision.png", center, span, 90, -89, show_collision=True)
    render(model, data, image_dir / "perspective.png", [0, 1.5, 0], 7, 135, -40)
    print("全部通过" if all_ok else "存在失败项")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
