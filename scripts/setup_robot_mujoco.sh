#!/usr/bin/env bash
# Prepare an isolated runtime for the copied upstream C++ simulator.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_ROOT="${ROBOT_ROOT:-$ROOT}"
RUN_DIR="$ROOT/output/robot-mujoco-local"
DEPS="$RUN_DIR/deps"
ENV_DIR="${ROBOT_ENV_DIR:-$ROOT/.venv-robot-mujoco}"
mkdir -p "$DEPS/cyclonedds/bin" "$DEPS/glfw-include/GLFW" "$DEPS/lib"
ln -sfn "$ROBOT_ROOT/unitree_sdk2/thirdparty/include" "$DEPS/cyclonedds/include"
ln -sfn "$ROBOT_ROOT/unitree_sdk2/thirdparty/lib/x86_64" "$DEPS/cyclonedds/lib"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  uv venv "$ENV_DIR" --python "${ROBOT_PYTHON:-3.11}"
fi
export CYCLONEDDS_HOME="$DEPS/cyclonedds"
# CycloneDDS embeds its build-time library prefix; do not reuse another project's wheel.
uv pip install --python "$ENV_DIR/bin/python" --no-cache --reinstall-package cyclonedds \
  -r "$ROOT/requirements.txt"
# Editable mode retains the SDK's CRC shared libraries, omitted by its wheel metadata.
uv pip install --python "$ENV_DIR/bin/python" --no-deps --editable "$ROBOT_ROOT/unitree_sdk2_python"

if [[ ! -f "$DEPS/mujoco-3.3.6-linux-x86_64.tar.gz" ]]; then
  curl -fL --retry 2 --connect-timeout 15 --max-time 180 \
    https://github.com/google-deepmind/mujoco/releases/download/3.3.6/mujoco-3.3.6-linux-x86_64.tar.gz \
    -o "$DEPS/mujoco-3.3.6-linux-x86_64.tar.gz"
fi
if [[ ! -d "$DEPS/mujoco-3.3.6/simulate" ]]; then
  tar -xzf "$DEPS/mujoco-3.3.6-linux-x86_64.tar.gz" -C "$DEPS"
fi
if [[ ! -f "$DEPS/glfw-include/GLFW/glfw3.h" ]]; then
  curl -fL --retry 2 --connect-timeout 15 --max-time 60 \
    https://raw.githubusercontent.com/glfw/glfw/3.4/include/GLFW/glfw3.h \
    -o "$DEPS/glfw-include/GLFW/glfw3.h"
fi
if [[ ! -f "$DEPS/sysroot/usr/include/eigen3/Eigen/Dense" ]]; then
  (cd "$DEPS" && apt-get download libeigen3-dev=3.4.0-2ubuntu2)
  dpkg-deb -x "$DEPS/libeigen3-dev_3.4.0-2ubuntu2_all.deb" "$DEPS/sysroot"
fi
GLFW_LIBRARY="$("$ENV_DIR/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/glfw/x11/libglfw.so"
ln -sfn "$GLFW_LIBRARY" "$DEPS/lib/libglfw.so.3"
cmake -S "$ROOT/scripts/robot_mujoco" -B "$RUN_DIR/native/build" \
  -DCMAKE_BUILD_TYPE=Release -DROBOT_ROOT="$ROBOT_ROOT" \
  -DMUJOCO_ROOT="$DEPS/mujoco-3.3.6" -DGLFW_INCLUDE_DIR="$DEPS/glfw-include" \
  -DGLFW_LIBRARY="$GLFW_LIBRARY" -DGLFW_LIBRARY_DIR="$DEPS/lib" \
  -DLOCAL_INCLUDE_DIR="$DEPS/sysroot/usr/include"
cmake --build "$RUN_DIR/native/build" --parallel 4
cp "$ROBOT_ROOT/unitree_mujoco/simulate/config.yaml" "$RUN_DIR/native/config.yaml"
"$ENV_DIR/bin/python" "$ROOT/scripts/robot_mujoco_local.py" --robot-root "$ROBOT_ROOT" --validate
uv pip freeze --python "$ENV_DIR/bin/python" > "$RUN_DIR/requirements.freeze.txt"
