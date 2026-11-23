#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python="${ROBOT_PYTHON:-$PWD/.venv-robot-mujoco/bin/python}"
exec xvfb-run -a -s '-screen 0 1600x1000x24 -nolisten tcp' \
  "$python" -u -m mujoco_mcp.server "$@"
