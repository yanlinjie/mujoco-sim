#!/usr/bin/env bash
# 让 Go2 在 SLAM 演示场景里按点位库一圈接一圈地巡航，Ctrl+C 停止。
# 如果仿真器没在运行，会先在后台启动 C++ 版 unitree_mujoco（日志写到 /tmp/umj_slam_scene.log）。
#
# 用法: patrol.sh [地图包目录] [传给 waypoint_tour.py 的其他参数，如 --back-and-forth]
#   不带参数时用 Habitat 公寓地图；场景文件名按地图包 meta.yaml 的 map_id 推出（与 map_to_mujoco.py 一致）。
#   例：scripts/patrol.sh maps/r2b_storage_ns/2026-11-01.3 --back-and-forth
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SIM_DIR="$DEMO_DIR/../unitree_mujoco/simulate/build"
PYTHON=/home/oc/miniconda3/envs/unitree-mujoco/bin/python
MAP_DIR="$(realpath "${1:-$DEMO_DIR/maps/habitat_apartment_1/2026-10-27.1}")"
shift $(( $# > 0 ? 1 : 0 ))
MAP_ID=$(sed -n 's/^map_id: *//p' "$MAP_DIR/meta.yaml")
SCENE=scene_slam_$MAP_ID.xml

if pgrep -f "unitree_mujoco -r go2 -s $SCENE" > /dev/null; then
    echo "仿真器已在运行，直接开始巡航"
elif pgrep -f "^./unitree_mujoco|unitree_mujoco.py" > /dev/null; then
    echo "有另一个 unitree_mujoco 仿真器在运行（可能加载的是别的场景），两者会在同一 DDS 域冲突，请先关掉它" >&2
    exit 1
else
    echo "启动仿真器 ..."
    (cd "$SIM_DIR" && nohup ./unitree_mujoco -r go2 -s "$SCENE" > /tmp/umj_slam_scene.log 2>&1 &)
    for _ in $(seq 60); do
        grep -q "Link_index" /tmp/umj_slam_scene.log 2>/dev/null && break
        sleep 0.5
    done
    sleep 2  # 等机器狗落地
fi

exec "$PYTHON" -I -u "$DEMO_DIR/scripts/waypoint_tour.py" dds "$MAP_DIR" --loop "$@"
