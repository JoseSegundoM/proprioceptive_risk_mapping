#!/usr/bin/env bash
# Isolated Python environment. Sees the system's ROS 2 packages, changes nothing outside this folder.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d mine_sentinel_autonomy ] || { git clone https://github.com/JoseSegundoM/mine_sentinel_autonomy.git mine_sentinel_autonomy \
  && git -C mine_sentinel_autonomy checkout eca13b025aee82b2bd3593918d61af0ba60281ed \
  && git -C mine_sentinel_autonomy submodule update --init; }
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
