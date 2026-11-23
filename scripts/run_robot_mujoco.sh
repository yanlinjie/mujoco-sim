#!/usr/bin/env bash
# Run the upstream HTTP service, native simulator and browser stream.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_ROOT="${ROBOT_ROOT:-$ROOT}"
ROBOT_MAP="${ROBOT_MAP:-family_house/v1}"
RUN_DIR="$ROOT/output/robot-mujoco-local"
PYTHON="${ROBOT_PYTHON:-${ROBOT_ENV_DIR:-$ROOT/.venv-robot-mujoco}/bin/python}"
SCRIPT="$ROOT/scripts/run_robot_mujoco.sh"
VIEWER_HOST="${ROBOT_VIEWER_HOST:-127.0.0.1}"
VIEWER_URL="http://$VIEWER_HOST:8771"
if [[ "$VIEWER_HOST" == "0.0.0.0" ]]; then
  VIEWER_URL="http://127.0.0.1:8771"
fi
mkdir -p "$RUN_DIR"
export NO_PROXY="localhost,127.0.0.1,::1${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
export LD_LIBRARY_PATH="$ROBOT_ROOT/unitree_sdk2/thirdparty/lib/x86_64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export LP_NUM_THREADS=4

owned_stack() {
  [[ -f "$RUN_DIR/stack.pid" ]] || return 1
  local stack_pid
  stack_pid="$(cat "$RUN_DIR/stack.pid")"
  [[ "$stack_pid" =~ ^[0-9]+$ && -r "/proc/$stack_pid/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/$stack_pid/cmdline" | rg -Fq "$SCRIPT"
}

case "${1:-start}" in
  start)
    if owned_stack; then
      printf 'Stack is running. Use status to check it.\n'
      exit 0
    fi
    [[ -x "$RUN_DIR/native/build/unitree_mujoco" ]] || {
      printf 'Run bash scripts/setup_robot_mujoco.sh first.\n' >&2; exit 1;
    }
    "$PYTHON" - "$VIEWER_HOST" <<'PY'
import socket
import sys

for host, port in (('127.0.0.1', 8770), (sys.argv[1], 8771)):
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
PY
    if pgrep -x unitree_mujoco >/dev/null; then
      printf 'An existing Unitree simulator is using the shared DDS domain; stop it first.\n' >&2
      exit 1
    fi
    rm -f "$RUN_DIR/ready"
    # Popen establishes the new session before returning, so a short-lived
    # launcher cannot exit before its background child has detached.
    stack_pid="$("$PYTHON" - "$SCRIPT" "$RUN_DIR/stack.log" <<'PY'
import subprocess, sys
with open(sys.argv[2], 'ab', buffering=0) as log:
    process = subprocess.Popen(['bash', sys.argv[1], 'run'], stdin=subprocess.DEVNULL,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print(process.pid)
PY
)"
    printf '%s\n' "$stack_pid" > "$RUN_DIR/stack.pid"
    printf 'Starting stack, PID %s; viewer %s/; log: %s/stack.log\n' "$stack_pid" "$VIEWER_URL" "$RUN_DIR"
    ;;
  run)
    exec xvfb-run -a -s '-screen 0 1600x1000x24 -nolisten tcp' bash "$SCRIPT" _run
    ;;
  _run)
    service_pid=''
    stream_pid=''
    cleanup() {
      trap - EXIT INT TERM
      if [[ -n "$service_pid" ]] && kill -0 "$service_pid" 2>/dev/null; then
        curl --noproxy '*' -fsS --max-time 20 -X POST \
          http://127.0.0.1:8770/simulator/stop >"$RUN_DIR/last-stop.json" || true
      fi
      [[ -z "$stream_pid" ]] || kill "$stream_pid" 2>/dev/null || true
      [[ -z "$service_pid" ]] || kill "$service_pid" 2>/dev/null || true
      wait || true
      rm -f "$RUN_DIR/ready"
    }
    trap cleanup EXIT
    trap 'exit 0' INT TERM
    cd "$ROOT"
    "$PYTHON" "$ROOT/scripts/robot_mujoco_local.py" --serve \
      --robot-root "$ROBOT_ROOT" --map "$ROBOT_MAP" >"$RUN_DIR/service.log" 2>&1 &
    service_pid=$!
    for _ in {1..120}; do
      kill -0 "$service_pid" || exit 1
      if curl --noproxy '*' -fsS --max-time 2 http://127.0.0.1:8770/health >"$RUN_DIR/health.json" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done
    curl --noproxy '*' -fsS --max-time 120 -X POST \
      http://127.0.0.1:8770/simulator/start >"$RUN_DIR/start.json"
    MUJOCO_GL=egl "$PYTHON" "$ROBOT_ROOT/slam_map_demo/scripts/sim_stream.py" \
      "$RUN_DIR/scene_slam_${ROBOT_MAP%%/*}.xml" --host "$VIEWER_HOST" --port 8771 --fps 10 \
      --map-dir "$ROBOT_ROOT/slam_map_demo/maps/$ROBOT_MAP" \
      --sim-service http://127.0.0.1:8770 >"$RUN_DIR/stream.log" 2>&1 &
    stream_pid=$!
    "$PYTHON" - "$stream_pid" "$VIEWER_URL/health" <<'PY'
import json, sys, time, urllib.request
from pathlib import Path
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
deadline = time.monotonic() + 120
while time.monotonic() < deadline:
    stat = Path('/proc') / sys.argv[1] / 'stat'
    if not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
        raise SystemExit('Stream process exited; see stream.log')
    try:
        with opener.open(sys.argv[2], timeout=3) as response:
            state = json.load(response)
        if state.get('ok') and state.get('simulator_state') and state.get('first_person', {}).get('ready'):
            break
    except (OSError, ValueError):
        pass
    time.sleep(0.5)
else:
    raise SystemExit('Stream not ready; see stream.log')
PY
    printf '%s\n' "$VIEWER_URL" >"$RUN_DIR/ready"
    printf 'Ready: API http://127.0.0.1:8770; viewer %s/\n' "$VIEWER_URL"
    wait -n "$service_pid" "$stream_pid"
    ;;
  status)
    viewer_url="$VIEWER_URL"
    if owned_stack && [[ -s "$RUN_DIR/ready" ]]; then
      viewer_url="$(< "$RUN_DIR/ready")"
    fi
    curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:8770/health
    printf '\n'
    curl --noproxy '*' -fsS --max-time 5 --url "$viewer_url/health"
    printf '\n'
    ;;
  stop)
    if ! owned_stack; then
      printf 'No owned stack is running.\n'; exit 0
    fi
    curl --noproxy '*' -fsS --max-time 20 -X POST \
      http://127.0.0.1:8770/simulator/stop >"$RUN_DIR/last-stop.json"
    kill -TERM -- "-$(cat "$RUN_DIR/stack.pid")"
    printf 'Stop requested for the owned stack.\n'
    ;;
  *) printf 'Usage: %s {start|status|stop}\n' "$0" >&2; exit 2 ;;
esac
