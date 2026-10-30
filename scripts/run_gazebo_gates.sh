#!/usr/bin/env bash
# Runs the repo's Gazebo gates on Jazzy in the isolated clone. Each gate manages its own
# process session (ms_eval.Run); nothing here kills processes by name.
W="$(cd "$(dirname "$0")/.." && pwd)"
source /opt/ros/jazzy/setup.bash; source "$W/ws_build/install/setup.bash"; source "$W/.venv/bin/activate"
export ROS_DOMAIN_ID=42 GZ_VERSION=harmonic MUJOCO_GL=egl
cd "$W/mine_sentinel_autonomy"
for g in "verify_simulation.py --variant point_foot" "verify_simulation.py --variant wheel_legged" "verify_sensors.py"; do
  n=$(echo "$g" | tr -c 'a-zA-Z0-9\n' '_'); echo "=== $g $(date +%T)"
  timeout 500 python scripts/$g > "$W/logs/gate_jazzy_$n.log" 2>&1; echo "exit=$?"; tail -5 "$W/logs/gate_jazzy_$n.log" | cut -c1-160
done
