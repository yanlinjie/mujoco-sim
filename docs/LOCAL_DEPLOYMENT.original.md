# 本地安装与启动

请阅读项目根目录的 [中文安装与使用说明](../README.zh-CN.md)。

该说明包含系统与 Python 依赖、已有权重配置、手动启动 MuJoCo 和模型、
本机浏览器访问、测试与故障排查。

本源码包不包含原部署机器的局域网转发配置、个人网络信息、账号、
机器路径或运行记录。模型权重与 Python 依赖需要另行准备。

## 本机运行 robot / Go2 地图仿真

该模式复用旁边 `../robot` 中的 `sim_service.py`、`follow_path.py`、
`sim_stream.py` 和 Unitree C++ 仿真器源码。默认地图为
`room1110/2026-11-11.2`，机器人为 12 关节 Go2。
不会启动 LightNav 模型、RPent 或 A* 服务。

### 环境准备

在 LightNav 项目根目录执行：

```bash
bash scripts/setup_robot_mujoco.sh
```

准备脚本创建独立 `.venv-robot-mujoco`，不会修改 LightNav 的 `.venv`。
它使用 MuJoCo 3.3.6、CycloneDDS Python 0.10.2，并复用 robot 中现有的
CycloneDDS 0.10.2 C/C++ 库。Unitree Python SDK 采用 editable 安装，
以保留普通 wheel 打包时遗漏的 CRC `.so` 文件。

C++ 仿真器在 `output/robot-mujoco-local/native/build/` 重新编译，
链接本机的 Boost、yaml-cpp 等库，不覆盖复制来的二进制和原有源码。
脚本下载 MuJoCo 官方开发包、GLFW 头文件，并通过 `apt-get download`
下载和解包 Eigen 头文件到本项目输出目录；不使用 sudo、不安装系统包。
当前脚本针对已验证的 Ubuntu 22.04 x86_64 主机，需要已有 uv、CMake、
C++ 编译器、Boost program_options、yaml-cpp 和 fmt 开发库。
下载需要代理时，仅在当前终端配置代理环境变量。

生成的本机场景 XML 保存在 `output/robot-mujoco-local/`。
场景中的地图资源路径被重定位，Go2 的 assets 使用符号链接；不会复制或
改写原始地图。Python 包版本快照保存在该目录的 `requirements.freeze.txt`。

### 启动与访问

```bash
bash scripts/run_robot_mujoco.sh start
bash scripts/run_robot_mujoco.sh status
```

`start` 后台启动服务，完整场景与推流通常需要约 1–3 分钟加载，状态与错误见
`output/robot-mujoco-local/stack.log`，生成 `ready` 文件后表示两项服务就绪。
重复 `start` 不会重复启动已有的本任务进程。

| 端口／资源 | 用途 |
| --- | --- |
| `127.0.0.1:8770` | 原有仿真 HTTP 控制接口 |
| `127.0.0.1:8771/` | 浏览器查看地图和 Go2 |
| `127.0.0.1:8771/snapshot.jpg` | 当前画面 JPEG |
| DDS domain 1、`lo` | 原有控制器与仿真器之间的本机通信 |

无桌面环境时，启动脚本用 `xvfb-run` 提供仅 Unix socket 的虚拟显示，
供 C++ 查看器使用。浏览器画面由原有 `sim_stream.py` 使用 EGL 独立渲染。
两个 HTTP 服务均只监听回环地址。为限制虚拟显示的软件渲染线程，
启动脚本设置 `LP_NUM_THREADS=4`、`OMP_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=1`。

此入口禁用控制服务自动拉起的推流，改由启动脚本单独启动原有推流程序，
显式传入 `--host 127.0.0.1`。因此控制服务 `/health` 中
`stream.enabled=false` 是预期行为，实际推流状态看 `8771/health`。

在这台主机的浏览器打开 `http://127.0.0.1:8771/`。
如果浏览器在另一台电脑上，需要使用已授权的私有访问方式；例如在操作者
电脑的普通终端建立 SSH 本地转发：

```bash
ssh -N -L 8771:127.0.0.1:8771 -L 8770:127.0.0.1:8770 用户@服务器
```

然后在操作者电脑打开同一回环地址。主机端接口可用以下命令检查：

```bash
curl --noproxy '*' http://127.0.0.1:8770/health
curl --noproxy '*' http://127.0.0.1:8770/pose
curl --noproxy '*' http://127.0.0.1:8771/health
```

### 停止与排错

```bash
bash scripts/run_robot_mujoco.sh stop
```

停止脚本先通过 `/simulator/stop` 关闭仿真器，再结束本次启动的控制服务、
推流和虚拟显示进程组。不要同时在 DDS domain 1 上运行其他 Unitree 仿真；
上游关闭接口按进程名枚举仿真器，启动脚本会拒绝与已有仿真器共用该域。
不安装或启用 systemd 服务；启动记录在 `stack.pid`。

| 文件／现象 | 排查方式 |
| --- | --- |
| `stack.log` | 整体启动失败或子进程退出 |
| `service.log` | HTTP、DDS 初始化错误 |
| `work/simulator.log` | 场景加载、C++ 查看器、显示与 OpenGL 错误 |
| `stream.log` | EGL、图像依赖、推流错误 |
| `work/walk_*.log` | 行走程序及动作结果 |
| `Address already in use` | 核对 8770、8771 的占用，不要结束未知进程 |
| `crc_amd64.so` 缺失 | 重新运行准备脚本，确认 SDK 以 editable 模式安装 |
| 原路径或共享库缺失 | 使用本地启动入口，勿直接运行复制来的二进制或旧场景 |

仅检查场景和 100 步物理推进，可单独执行：

```bash
.venv-robot-mujoco/bin/python scripts/robot_mujoco_local.py --validate
```

该检查不证明行走或模型推理成功。实际服务验证结果应分别核对
`simulator_running`、位姿、推流状态及行走结果；本模式没有模型推理。

### 本机验证记录（2026-11-14）

- 本机重新编译的 C++ 仿真器和 Python MuJoCo 均为 3.3.6；`ldd` 没有缺失库。
- 默认场景加载成功：`nq=19`、`nv=18`、`nu=12`、`ngeom=59`，100 步物理状态有限。
- `/pose` 收到真实仿真 DDS 状态，`fallen=false`；控制服务地图身份与场景一致。
- `/walk` 已实际执行一个前方 0.4 米目标的短距离检查，返回
  `finished`、`exit_code=0`、`result.ok=true`、`reached=true`、`fell=false`。
  服务报告到达时位置误差为 0.120 米，按结束位姿复算为 0.152 米；两者都小于 0.25 米。
- 浏览器推流 `/health` 返回 `ok=true`、`simulator_state=true`，实测约 10 FPS；
  已获取并查看地图和 Go2 的实际截图 `output/robot-mujoco-local/snapshot.jpg`。
- 停止命令已确认释放两个监听端口；启动探测使用 `SO_REUSEADDR`，避免端口
  TIME_WAIT 使快速重启被误判为仍被占用。停止后已重新启动，并再次确认
  仿真与推流健康；最新结果保存在 `verified-health.json`。

行走请求、受理回执和完整结果分别位于输出目录的
`smoke-walk-request.json`、`smoke-walk-accepted.json`、`smoke-walk-result.json`。
上述记录仅验证原有地图仿真服务，没有执行 LightNav 模型推理。

## 本机运行 A* 导航服务

A* 使用同一个 `.venv-robot-mujoco` 环境，直接运行复制目录中的
`robot/slam_map_demo/scripts/nav_service.py`。准备脚本已包含所需的
`websockets==16.1.1`；现有环境可单独补装：

```bash
uv pip install --python .venv-robot-mujoco/bin/python websockets==16.1.1
```

启动、检查、停止均在本项目根目录执行：

```bash
bash scripts/run_robot_astar.sh start
bash scripts/run_robot_astar.sh status
bash scripts/run_robot_astar.sh stop
```

启动器默认选用 `room1110/2026-11-11.2`，与上述 MuJoCo 仿真一致；
可用 `ROBOT_ROOT` 和 `ROBOT_MAP` 环境变量指定复制目录和地图。
切换地图前先停止 A*，重新启动后核对两项服务的 `map_id`、`map_version`。
停止 A* 不会停止仿真；停止仿真也不会停止 A*。
修改点位或导航栅格后，需要重启 A* 才能重新加载。

| 地址 | 用途 |
| --- | --- |
| `http://127.0.0.1:8765/health` | 地图身份与点位数量 |
| `http://127.0.0.1:8765/waypoints` | 实时点位名称、编号、站位与朝向 |
| `http://127.0.0.1:8765/plan` | GET 或 POST 路径规划 |
| `ws://127.0.0.1:8766` | WebSocket 查询和规划 |

HTTP 和 WebSocket 仅监听回环地址。默认地图有 5 个点位：充电桩、广告牌、
厕所通道、男厕所门口、母婴室；此地图没有场景图。
规划请求中的坐标为地图坐标、米和弧度，实际起点应从仿真的 `/pose` 获取。
以下为从地图原点规划到厕所通道的接口示例，只生成路线：

```bash
curl --noproxy '*' http://127.0.0.1:8765/waypoints
curl --noproxy '*' -X POST http://127.0.0.1:8765/plan \
  -H 'Content-Type: application/json' \
  -d '{"start":{"x":0,"y":0,"yaw":0},"goals":["厕所通道"],"step":0.2}'
```

WebSocket 消息使用相同字段，并添加 `type: "plan"`；也支持
`type: "health"`、`type: "waypoints"`。Python WebSocket 客户端直连本机时，
可给 `connect()` 显式设置 `proxy=None`。

RPent 的 MuJoCo 模式对应两个端点：

```text
--nav-endpoint http://127.0.0.1:8765
--sim-endpoint http://127.0.0.1:8770
```

如果操作者在另一台电脑查看接口，使用已授权的私有访问方式。
SSH 本地转发可在操作者电脑上增加以下端口：

```bash
ssh -N -L 8765:127.0.0.1:8765 -L 8766:127.0.0.1:8766 用户@服务器
```

日志与 PID 分别在 `output/robot-astar-local/service.log` 和 `service.pid`。
端口占用时先核对占用进程；启动器不会结束其他服务。准备脚本不会改写
原地图、原导航代码或 Docker 配置，也不会修改系统服务。

本机已验证（2026-11-14）：两项服务地图身份相同；从实时仿真位姿到全部
5 个点位的 HTTP 规划成功；WebSocket 健康查询、中文名称规划和 request_id
回传正常；未知目标返回 HTTP 400 和 `ok=false`；A* 停止、重启及重复启动检查通过。
完整结果在 `output/robot-astar-local/verified-checks.json`，路线报告在同目录的
`plan-wp_*.json` 和 `plan-websocket.json`。以上验证没有提交新的行走任务。

## 用 RPent 调用本机两个服务做集成测试

前提是 A* 和 MuJoCo 服务已运行，RPent 位于旁边的 `../rpent/RPent`，
其 `.venv` 已安装依赖，私有 `.env.local` 中配置了可用的模型 API。
本次复用了已有的 RPent 环境和私有配置，没有修改 RPent 源码或已有记忆。

在普通终端运行，保持 TTY：

```bash
bash scripts/test_rpent_local.sh
```

默认任务先读取点位和实时位姿，记录起点，沿起始朝向规划一个前方 0.6 米的
坐标目标，确认到达后，再独立规划并返回起点和原朝向。它显式使用本机
`8765` 和 `8770` 端点，调用已配置的真实 API 模型；不启用前端回调。
每段行走预算 90 秒，会话预算 240 秒，模型请求上限 12 次。
CLI 因任务成功、失败、预算耗尽或取消而退出，外部两个服务继续运行。

每次使用新的输出和隔离记忆目录，最新目录路径记录在
`output/rpent-local/latest-run.txt`。目录中包含：

- `transcript_unitree_*.json`：模型响应、工具输入输出和最终 `finish`。
- `states.json`：完成目标数、活动行走、停止错误和最终位姿。
- `navigation/plans/` 与 `navigation/walks/`：每段 A* 路线及行走报告。
- `run.log`：运行日志。

可给脚本传入其他有限任务，例如：

```bash
bash scripts/test_rpent_local.sh '查询实时点位和当前位姿，报告两项服务的地图版本，然后结束。'
```

`RPENT_ROOT`、`RPENT_CONFIG_FILE` 可指定其他 RPent 源码和私有配置路径。
配置中的模型变量为 `RPENT_MODEL`；认证仍由原配置提供，不要把密钥写进脚本。
Ctrl+C 使用 RPent 原有的取消与自有行走停止流程，取消后应核对服务状态。

### 本次真实运行结果（2026-11-14）

模型为 `openai:deepseek-flash`，最小 API 检查成功。随后 CLI 实际运行 69 秒，
使用 7 次模型请求和 8 次工具调用，顺序为：

```text
list_navigation_goals → get_navigation_pose
→ plan_navigation → follow_navigation
→ plan_navigation → follow_navigation
→ get_navigation_status → finish
```

两段均保存了独立路线并实际执行，最终 `finish.status=success`、`error=null`，
`completed_navigation_goals=2`、`active_walk_id=null`、`submission_uncertain=false`、
`stop_error=null`，机器人未摔倒。

| 段落 | 服务报告位置误差 | 按结束位姿复算的位置误差 | 按结束位姿复算的朝向误差 |
| --- | --- | --- | --- |
| 前方 0.6 米目标 | 0.119 米 | 0.148 米 | 0.0014 弧度 |
| 返回最初起点 | 0.033 米 | 0.088 米 | 0.0103 弧度 |

RPent 的验收阈值是位置不超过 0.25 米、指定朝向时角度不超过 0.2 弧度，
两段均通过。去程实际位移约 0.452 米，因此通过容差验收不代表精确移动了
0.6 米。模型总结有一处将服务报告误差误称为容差；此处采用代码中的阈值和
结束位姿独立复算，没有修改原始 transcript。

本次报告在
`output/rpent-local/run-20261114T042844Z-C3Mr39/verification.json`，
该文件包含核验后的指标和实际工具顺序。测试只涉及 RPent 的 API 规划模型
和本机导航、仿真服务，没有调用 LightNav 模型。
