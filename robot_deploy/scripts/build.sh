#!/usr/bin/env bash
set -eo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd -- "${script_dir}/.." && pwd)"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    echo "ROS 2 Humble was not found at /opt/ros/humble" >&2
    exit 1
fi

source /opt/ros/humble/setup.bash
cd "${workspace_dir}"
colcon build --symlink-install "$@"
