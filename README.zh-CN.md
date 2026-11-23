# Go2 MuJoCo 仿真和 MCP

仿真器、Go2 模型和 DDS 控制代码位于本仓库。地图包部署到 `slam_map_demo/maps/<地图>/<版本>/`，并通过 `ROBOT_MAP` 选择。A* 与仿真使用相同的地图 ID 和版本。

## 安装依赖

Ubuntu x86_64 上需要 `uv`、`curl`、CMake 3.16+、C++17 编译器、Xvfb 和 `xauth`，以及 Boost.ProgramOptions、yaml-cpp 和 fmt 的开发包。脚本首次下载 Eigen 时，APT 源须提供 `libeigen3-dev=3.4.0-2ubuntu2`。

在仓库根目录运行准备脚本。脚本默认创建本仓库的 Python 3.11 环境 `.venv-robot-mujoco`，安装 Python 依赖和 Unitree SDK，并编译仿真器。 [依赖清单](requirements.txt)

```bash
bash scripts/setup_robot_mujoco.sh
```

## 启动

仿真已运行时可以直接启动 MCP。

```bash
ROBOT_MAP=room1110/2026-11-11.2 bash scripts/run_robot_mujoco.sh start
.venv-robot-mujoco/bin/python -m mujoco_mcp.server
```

`start` 会一起启动仿真和地图网页。要在局域网浏览器打开网页，给启动命令设置 `ROBOT_VIEWER_HOST` 为本机的局域网 IPv4 地址，访问该地址的 `8771` 端口。

MCP 默认在 `http://127.0.0.1:8791/mcp` 监听，连接本机 8770 端口的仿真服务，并从 A* 的 `http://127.0.0.1:8790/mcp` 取完整规划。使用 `--sim-endpoint`、`--astar-mcp-url` 和 `--port` 可设置其他地址。

- `get_pose()` 查询地图位姿。
- `get_navigation_status()` 检查控制器及与 A* 的地图一致性。
- `follow_plan(plan_id)` 跟踪完整路径并返回到达和停车证据。
- `get_walk_status()` 查询当前或最近一次行走。
- `preview_plan(plan_id)` 用独立仿真预演同一路径。

同一仿真器的地图行走和连续航点控制共享控制权。收到取消或调用方断开后，MuJoCo MCP 按其拥有的行走编号请求停车并确认状态。调用方可在 `tools/call` 的 `_meta` 中提供 `rpent.dev/run-id`，随后用同一元数据调用只读的 `get_run_report()`，查询本次执行和停车证据。

本机运维仍使用 `bash scripts/run_robot_mujoco.sh status` 和 `stop`。第一视角观测与浏览器预览由仿真栈的 8771 端口提供。
