# 快速验证：SLAM 地图导入 MuJoCo

用开源扫描场景（Habitat 测试场景中的 `apartment_1.glb`，真实公寓扫描重建的 Mesh）伪造一份 SLAM 地图包，
验证“地图包 → 点位库 → unitree_mujoco 场景 → 机器狗按点位巡航”这一段流程。
这里只用到 Habitat 提供的模型文件，不使用 Habitat 仿真器。

## 目录

```
downloads/   下载的原始数据（不要在此目录内运行脚本）
scripts/     处理脚本
maps/        生成的地图包，结构与 SLAM部署方案.md 第七步一致
output/      自动检查的截图、巡航轨迹图
```

生成的场景文件写在 `../unitree_mujoco/unitree_robots/go2/scene_slam_habitat_apartment_1.xml`，
它通过绝对路径引用 `maps/` 下的资源，移动本目录后需重新生成。

## 流程

所有命令在本目录下、`unitree-mujoco` conda 环境中执行：

```bash
P=/home/oc/miniconda3/envs/unitree-mujoco/bin/python
MAP=maps/habitat_apartment_1/2026-10-27.1
SCENE=../unitree_mujoco/unitree_robots/go2/scene_slam_habitat_apartment_1.xml

# 1. 制作演示地图包，含重力对齐（真实 SLAM 跑通后，这一步由建图流程产出，跳过即可）
$P -I scripts/make_demo_map_package.py downloads/data/scene_datasets/habitat-test-scenes/apartment_1.glb $MAP

# 2. 标注点位库 waypoints.yaml：在家具、拐角、通道口附近自动找站位，并计算相邻关系
$P -I scripts/annotate_waypoints.py $MAP
#    或者人工点选：在 CloudCompare 里每个点位点两下（站位、要看向的位置），导出点列表，
#    名称文件每行「名称 | 标签1, 标签2」；需先完成第 3 步（用到导航栅格），之后重跑第 3 步刷新点位标记。
#    实例见 maps/go2_room_selfbuilt/2026-11-09.2/waypoint_picks.txt 与 waypoint_names.txt
# $P -I scripts/waypoints_from_picks.py $MAP <点列表.txt> <名称.txt> [--snap]

# 3. 地图包转 MuJoCo 场景
$P -I scripts/map_to_mujoco.py $MAP ../unitree_mujoco/unitree_robots/go2

# 4. 自动检查并输出截图
MUJOCO_GL=egl $P -I scripts/check_scene.py $SCENE $MAP output/check

# 5. 不开窗口跑一遍完整巡航，输出轨迹图（约 1 分钟）
$P -I scripts/waypoint_tour.py headless $MAP $SCENE output/tour_headless.png
```

## 在 unitree_mujoco 中运行

C++ 仿真器可以直接用参数指定场景，不需要改配置文件：

```bash
cd ../unitree_mujoco/simulate/build && ./unitree_mujoco -r go2 -s scene_slam_habitat_apartment_1.xml
```

仿真器启动后，另开终端让机器狗按点位库巡航（从充电桩出发，依次访问所有点位后回到充电桩，约 7 分钟）：

```bash
$P -I -u scripts/waypoint_tour.py dds $MAP
```

查看器中按数字键 `3` 可显示碰撞高度场（同时会显示机器人的碰撞体）。
Python 版仿真器则需把 `simulate_python/config.py` 中的 `ROBOT_SCENE` 改为上述场景文件。

## 场景组成

- 视觉：Mesh 带贴图，不参与碰撞。
- 碰撞：由稠密点云生成 0.05 m 栅格的高度场，忽略 1.5 m 以上的点（天花板）。
  地面压成严格平整的 z=0（成片且高于 3 cm 的地毯压成各自的固定高度）。
- 点位库：绿色圆盘为站位，红色短条为朝向。

## 踩过的坑

- **地图必须重力对齐**：原始扫描的地面有约 1° 倾斜，跨 10 m 差出近 10 cm，
  会被误当成地毯台阶。制作地图包时用地面拟合平面把场景转平。
- **高度场地面必须严格平整**：MuJoCo 把高度场每个格子当作独立棱柱计算碰撞，
  相邻格子只差几毫米，足端也会碰到棱边产生倾斜的接触力，
  使机器狗朝某些方向走时打滑、倒退甚至侧翻。
- **扫描缺失区**：高度场里没有数据的格子在仿真中是平地，但导航时一律当作障碍，避免走出地图。
- **出生点**：仿真器总把机器狗以伸直腿的姿态放在世界原点、朝 +x，
  因此地图原点要选在平地上，且 +x 方向留出行走空间。
- **步态**：沿用 `example/python/walk_go2.py` 的开环小跑步态，扩展了转向（髋关节侧摆实现切向迈步）。
  横向平移不稳定，没有使用；高速急转会侧翻，边走边转时限制了角速度。
  电机顺序是 FR、FL、RR、RL，与模型中关节顺序不同。

## 已知限制

- 高度场只能表达“每个位置一个高度”，桌子底下这类悬空空间会被当成实心。
- 导航使用仿真器给出的真实位姿，没有接入 SLAM 定位。
- 开环步态在真实机器人上不可用，仅用于仿真中验证地图和点位库。

## 路径规划服务（按点位名称）

`scripts/nav_service.py`：输入当前位置和一个或多个点位名称，按先后顺序用 A* 依次规划每一段，输出 JSON。
名称可以是点位编号（`wp_02`）、名称、语义标签或名称中的一部分；有歧义或找不到时报错并列出可用点位。
规划用的导航栅格与巡航脚本相同（`mujoco/collision_hfield.npz`）。
完整的接口说明（地址、请求与返回字段、错误处理、调用示例）见 [导航规划服务接口文档.md](导航规划服务接口文档.md)。

现在以 Docker 容器运行（代码和地图都打包在镜像里），启动、切换地图、重新构建见 [../nav_astar_docker/README.md](../nav_astar_docker/README.md)。下面是不用 Docker 时的宿主机命令（调试用）：

```bash
# 命令行规划一次
$P -I scripts/nav_service.py plan maps/go2_room_selfbuilt/2026-11-09.2 --start 0 0 --goals 广告牌 厕所通道 [--step 0.1]

# 启动局域网服务：HTTP 8765 端口，WebSocket 8766 端口，监听所有网卡
$P -I -u scripts/nav_service.py serve maps/go2_room_selfbuilt/2026-11-09.2
```

其他设备调用（把 IP 换成运行服务的电脑）：

```bash
curl http://192.168.104.191:8765/waypoints
curl "http://192.168.104.191:8765/plan?x=0&y=0&goals=广告牌,厕所通道"
curl -X POST http://192.168.104.191:8765/plan -H 'Content-Type: application/json' \
     -d '{"start": {"x": 0, "y": 0, "yaw": 0}, "goals": ["广告牌", "厕所通道"], "step": 0.2}'
```

```python
import asyncio, json
from websockets.asyncio.client import connect

async def main():
    # proxy=None：websockets 15 以后默认走系统代理环境变量，局域网直连时关掉
    async with connect("ws://192.168.104.191:8766", proxy=None) as ws:
        await ws.send(json.dumps({"type": "plan", "request_id": 1, "start": {"x": 0, "y": 0},
                                  "goals": ["广告牌", "厕所通道"]}, ensure_ascii=False))
        print(json.loads(await ws.recv()))

asyncio.run(main())
```

返回中每段（`legs`）包含：目标点位（站位、到达后的朝向 `stand_pose.yaw`、看向的位置），
拉直后的折线 `path`（第一个点为该段起点），每段折线的行进方向 `headings`，长度 `length`；
请求带 `step` 时另有按该间距加密的 `dense_path`。坐标为地图坐标系，单位米、弧度。
起点离障碍比避障半径（0.3 m）还近时，第一段先直线移到最近的可通行处，并在 `warnings` 中说明；
起点在墙外等与目标不连通的区域时返回失败。

## 仿真控制服务（局域网远程操作仿真）

`scripts/sim_service.py`，HTTP 8770 端口，以用户级 systemd 服务 `robot-sim-service` 常驻（单元文件 `systemd/robot-sim-service.service`）。
其他电脑通过它启动/重启仿真器、读位姿、提交路线让 Go2 行走（A* 返回或坐标点列）、查询进度、停止、不开窗口预演。
启动仿真器时会同时拉起下文的浏览器推流服务（`--stream-port`，默认 8771，0 表示不推流），关闭仿真器时一并停止。
接口说明见 [仿真控制服务接口文档.md](仿真控制服务接口文档.md)。A* 规划服务的 `goals` 也已支持坐标目标 `{"x":…, "y":…}`。

本机副本增加持续轨迹控制：`POST /control/start` 默认创建 trajectory 会话，
`POST /control/trajectory` 定期提交航点、坐标系及图像采集时间／位姿，服务内部 MPC 跟踪。
`GET /control` 查询状态，`POST /control/stop` 等待站稳并释放控制权。
轨迹默认自图像采集起 2 秒过期，过期后站立但不释放会话；与 `/walk` 互斥。
持续控制只提供航点输入接口，不提供速度模式。协议及限速见上述接口文档，
客户端和实际验证见 [本地部署说明](../../docs/LOCAL_DEPLOYMENT.md)。

## 浏览器查看仿真画面（MJPEG 推流）

`scripts/sim_stream.py`，HTTP 8771 端口。局域网内浏览器打开 `http://<服务器IP>:8771/` 即可观看；
嵌到别的网页里只需 `<img src="http://<服务器IP>:8771/stream">`，单帧截图为 `/snapshot.jpg`，推流状态为 `/health`，
实时状态与最近的滚动文字为 `/status?n=行数`。

画面左上角显示年月日和时间；3D 画面下方拼接一个 7 行的滚动文字板块（画在视频帧里，帧大小为 1280×932）：
每秒滚入一行实时状态（位置、朝向、速度、机身高度、倾斜、行走用时和阶段），
仿真器连上或断开、开始行走、行走日志、行走结果、摔倒等事件用黄色或红色插入。行走信息读自仿真控制服务
（`--sim-service`，默认 `http://127.0.0.1:8770`）的 `GET /walk`，连不上时只显示机身状态。
`--panel-lines 0` 不显示文字板块，`--status-interval` 调整状态行的间隔。

```bash
$P -I -u scripts/sim_stream.py                     # 默认场景 scene_slam_room1110.xml，1280x720，15 帧/秒
$P -I -u scripts/sim_stream.py scene_slam_xxx.xml --azimuth -60 --elevation -50 --zoom 1.2 --fps 10
```

C++ 仿真器的窗口属于另一个进程，读不到它的像素。本脚本加载同一个场景文件，只读订阅仿真器发出的
`rt/lowstate`（12 个关节角、IMU 姿态）和 `rt/sportmodestate`（机身位置），把机器狗摆到同样的位姿，
再用 `mujoco.Renderer` 离屏渲染（EGL），经 OpenCV 编码为 JPEG 后由 Flask 持续推送。不发控制指令，
仿真器可以随时启停，没收到状态时画面左上角显示 `waiting for simulator state`。
相机是固定的斜侧视角，按地图网格顶点自动调整视线中心和距离，使整张地图恰好放进画面；整图视角下机器狗只有约 30 像素宽。

Go2 模型另有机身固定相机 `go2_front_rgb`，朝机身 +x，输出不带文字和点位标记的第一视角。
浏览器打开 `http://127.0.0.1:8771/camera/front`，JPEG 为 `/camera/front.jpg`，
MJPEG 为 `/camera/front/stream`。模型观测接口 `/observation` 将 JPEG base64、采集主机时间
`stamp_ns`、同一渲染状态的 `pose`（IMU）和 `base_pose`（base_link）一起返回。
尺寸默认 480×270，水平视场角 112°，可用 `--front-width`、`--front-height`、`--front-hfov` 调整。
通过 `--map-dir <地图包>` 将地图编号与版本加入观测。图像或任一路 DDS 状态超过 1 秒时，
观测及第一视角截图接口返回 HTTP 503，不返回旧帧。完整字段和本项目运行命令见
[本地部署说明](../../docs/LOCAL_DEPLOYMENT.md)。
带宽约 1.2 MB/s（每秒 15 帧、JPEG 质量 80），占用约半个 CPU 核。停止：`kill <pid>`（不要用 `pkill -f`，会误杀执行命令的 shell）。

## 沿任意折线行走（按坐标）

`nav_demo.py` 只接受点位名称；下面三个脚本把“规划”和“行走”拆开，行走只认坐标。
Claude Code 技能 `astar-path-planning`、`mujoco-path-following`、`go2-mujoco-navigation`（`../.claude/skills/`）描述了完整用法。

```bash
# 读仿真器中机器狗的当前位姿（只读）：{"ok": true, "x": .., "y": .., "yaw": .., "tilt_deg": ..}
$P -I scripts/sim_pose.py

# 目标是坐标时的 A*（服务只接受名称）：与服务同一套导航栅格，输出 {"ok": true, "points": [[x, y], ...]}
$P -I scripts/plan_to_xy.py $MAP --start 0 0 --goals 2,3 -0.37,4.54 [--final-yaw 3.14] > /tmp/xy_plan.json

# 沿折线行走：--points 直接给拐点，或 --path-json 读点列表 / plan_to_xy.py 输出 / 规划服务 /plan 的完整返回
$P -I -u scripts/follow_path.py dds $MAP --points 0,0 2,1 2,4 --final-yaw 1.57 --result-json /tmp/r.json [--no-hold]
$P -I -u scripts/follow_path.py dds $MAP --path-json /tmp/nav_plan.json --result-json /tmp/r.json
# 不开窗口预演，可指定出生位姿并输出轨迹图
$P -I -u scripts/follow_path.py headless $MAP --scene $SCENE --path-json /tmp/xy_plan.json --spawn 0 0 0 --image /tmp/p.png
```

`follow_path.py` 行走前检查折线，穿过障碍或未扫描区域时拒绝（`--force` 跳过检查）；
机器狗离折线起点超过 0.3 m 时先走过去，直线有障碍则先用 A* 绕行。
`kill <pid>` 与 Ctrl+C 一样会先站稳再退出；不要 `kill -9`，否则仿真器一直执行最后一条行走指令。
机器狗实际停下的位置与目标有几厘米偏差，每次行走前应重新读位姿。
此前 C++ 仿真器中机器狗站立时会以约 1 cm/s 向身后滑动，原因是仿真器自带的 MuJoCo 3.3.6 在高度场栅格边缘产生倾斜接触；
2026-11-10 起 map_to_mujoco.py 生成的场景在 z=0 加平面承担落脚、高度场下移 5 mm，平地上不再滑动（地毯台面上仍可能滑）。

## 激光实时定位（room 地图）

在已建好的地图中，用机器狗实时的激光帧确定它在地图坐标系中的位置：每来一帧 L1 激光，
先用腿式里程计预测位姿，再把这一帧配准到地图点云上，得到机身的 x、y、z、朝向。

| 文件 | 作用 |
|---|---|
| `scripts/localizer.py` | 定位核心（只依赖 numpy、scipy），算法说明见文件开头 |
| `scripts/localize_dds.py` | 实时节点：订阅 Go2 的 `rt/utlidar/cloud`、`rt/utlidar/robot_odom`，发布 `rt/slam_map/base_pose`，并提供 HTTP 查询 |
| `scripts/localize_replay.py` | 用录制包按顺序回放，与建图轨迹对比误差，可模拟“被搬走” |
| `scripts/dds_replay.py` | 本机验证用：把录制数据按真实时间以 Go2 的话题名通过 DDS 发出 |
| `config/go2_l1/` | 雷达到机身的外参、机身自遮挡体素（换机器狗或雷达安装位置后需重新标定） |

第一次运行会预处理地图（约 10 s），结果缓存在地图包的 `localization/loc_map.npz`。
当前使用 2026-11-09.2 版本（与规划服务同一坐标系）；下方测试结果是在 2026-11-06.1 上测的，
切换后在新版本上回放第 100~160 s 复测：水平误差中位数 1.9 cm、最大 7.6 cm，与旧版本一致。

### 接真机

```bash
P=/home/oc/miniconda3/envs/unitree-mujoco/bin/python
MAP=maps/go2_room_selfbuilt/2026-11-09.2
# --iface 换成连接机器狗的网卡；--init 为机身大致位姿（x y 朝向弧度，误差 1 m、30° 以内即可）
$P -I -u scripts/localize_dds.py $MAP --iface eth0 --domain 0 --init -0.11 -2.52 1.85
# 或者 --resume：用上次运行保存的最后位姿（localization/last_pose.json）作提示；都不给时做全图搜索
```

查询与设置：

```bash
curl http://<主机>:8767/pose      # {"status": "TRACKING", "x": .., "y": .., "z": .., "yaw": .., "fitness": .., "age_s": ..}
curl -X POST http://<主机>:8767/initial_pose -H 'Content-Type: application/json' -d '{"x": -0.11, "y": -2.52, "yaw": 1.85}'
```

`status`：`INIT` 正在初始化（还没有位姿）、`RELOCALIZED` 刚完成初始化或重定位、`TRACKING` 正常、
`PREDICT` 本帧配准不可信只用了里程计预测、`LOST` 判为丢失并开始重定位。
坐标与 `nav_service.py` 的规划结果是同一个地图坐标系，可以直接把 `/pose` 的 x、y、yaw 作为规划起点。

### 本机验证

```bash
V=/home/oc/projects/robot/slam_data/.venv-r2b/bin/python      # 有 rosbags
BAG=/mnt/sda2/robot_download/room-19700102-135516-vpn1/bag
REF=/home/oc/projects/robot/slam_data/runs/go2_room_selfbuilt/go2_room_selfbuilt_poses_final_tum.txt
# 整包回放（约 5 分钟），输出误差统计和轨迹图 output/localization/full.png
$V -I scripts/localize_replay.py $MAP $BAG --hint-from-ref 0.6 25 --ref $REF --out output/localization/full
# 从任意时刻开始、不给提示（全图搜索）：--start 240 --duration 40；模拟被搬走：--kidnap 150

# DDS 端到端：一个终端起定位节点，另一个终端回放
$P -I -u scripts/localize_dds.py $MAP --iface lo --domain 1 --init -0.86 -1.0 2.065
W=/mnt/sda2/robot_download/go2_room_selfbuilt
$P -I -u scripts/dds_replay.py $W/l1_scans.npz $W/calib/leg_odom.npz --start 100 --duration 60
```

### 测试结果（2026-11-09，录制包 room-19700102-135516-vpn1）

- 整包 12 分钟回放：全程跟踪不丢失，雷达位置与建图轨迹的水平误差中位数 2.1 cm、95% 分位 5.1 cm、最大 15.8 cm；
  偏航误差中位数 0.4°；每帧耗时中位数 27 ms、最大 42 ms（激光 15 Hz，每帧预算 66 ms）；
- DDS 端到端（60 s，从第 100 s 开始，真实时间）：机身水平误差中位数 2.0 cm、95% 分位 4.8 cm，位姿延迟约 0.05 s；
- 初始化，在包内 11 个时刻起步：
  - 给提示（故意偏 0.6 m、25°）：9 次在 1.4 s 内正确（误差 1~15 cm），1 次一直未定位（第 30 s），
    **1 次错误**（第 520 s，偏 2 m：机器狗静止、贴近墙，离真实位置 2 m 处的另一个地方对这帧的匹配反而更好）；
  - 不给提示（全图搜索）：4 次正确（2.5~19 s 内定位，误差 1~7 cm），7 次 40 s 内一直未定位，没有给出错误位姿；
- 被搬走（第 150 s 跳过 10 s 数据，期间里程计不动）：1 s 内判为丢失，在最后可信位置附近重定位成功，误差 2.5 cm。

注意：参考轨迹和地图来自同一份录制数据，上面的跟踪精度偏乐观；换一天、家具挪动、人多时的表现还没有数据验证。

### 已知限制

- L1 倒装，大部分点打在地面上，离地结构点少、看得近；机器狗静止对着墙时全图搜索无法唯一确定位置，会一直等待。
  实际使用建议给初始位姿提示（`--init`、`/initial_pose` 或 `--resume`）；
- 地图与现场不一致（门开关、家具挪动）会降低初始化的得分，起步处的门就是这种情况；
- 还没有接入巡航脚本：`waypoint_tour.py` 在仿真中仍使用仿真器给出的真实位姿（仿真里没有 L1 激光）；
- 外参平移取 URDF 值，机身自遮挡体素来自这次录制，换机器狗后需重新生成 `config/go2_l1/`。
