# Go2 MuJoCo 仿真 MCP

MuJoCo 从 A* MCP 取得路线，在本机启动 Go2 仿真器并通过 DDS 控制。地图包位于 `slam_map_demo/maps/<地图>/<版本>/`。两端使用相同的地图 ID 和版本。

## 安装依赖

Ubuntu x86_64 需要 `uv`、`curl`、CMake 3.16+、C++17 编译器、Xvfb、xauth，以及 Boost.ProgramOptions、yaml-cpp、fmt 的开发包。准备脚本安装 [Python 依赖](requirements.txt) 和 Unitree SDK，并编译 Go2 仿真器。

```bash
bash scripts/setup_robot_mujoco.sh
```

## 启动

先启动 A* MCP，再在 MuJoCo 仓库根目录执行。按 Ctrl+C 停止服务和它启动的仿真器。

```bash
bash scripts/run_robot_mujoco.sh --map room1110/2026-11-11.2
```

MuJoCo 的 MCP 地址为 `http://127.0.0.1:8791/mcp`，地图网页为 `http://127.0.0.1:8791/`。使用 `--host` 和 `--port` 设置两者的监听地址。绑定局域网地址时，同一地址提供地图网页和 MCP。加 `--no-first-person` 只显示地图全景，加 `--no-viewer` 关闭网页渲染。

A* 地址默认为 `http://127.0.0.1:8790/mcp`，可用 `--astar-mcp-url` 修改。单独测试时可设置独立的 `--dds-domain` 和 `--port`。默认 DDS 域为 1。

| MCP 工具 | 用途 |
|---|---|
| `get_pose()` | 查询机器狗在地图上的位姿 |
| `get_navigation_status()` | 检查导航可用性和 A* 地图版本 |
| `follow_plan(plan_id)` | 执行 A* 保存的路线并确认到达 |
| `get_walk_status()` | 查询当前或最近一次行走 |
| `preview_plan(plan_id)` | 用独立的离屏仿真预演路线 |
| `get_run_report()` | 根据 `rpent.dev/run-id` 查询执行和停车证据 |

收到取消或客户端断开时，服务按当前行走编号请求停车并确认进程退出。仿真器和行走日志位于 `output/robot-mujoco-local/work/`。
