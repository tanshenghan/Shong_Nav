#!/usr/bin/env bash
set -euo pipefail
# Usage: bash data_pipeline/run_lerobot.sh NEW_OUTPUT_DIR [EPISODES] [collector flags...]
BETA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BETA_EXPORT=$(realpath -m "${1:?supply a NEW LeRobot output directory}")
BETA_EPISODES=${2:-1}
BETA_LEROBOT_PYTHON=${LEROBOT_PYTHON:-"$BETA_REPO/data_pipeline/.venv/lerobot/bin/python"}
if [[ -e "$BETA_EXPORT" ]]; then
  echo "Output already exists: $BETA_EXPORT" >&2
  exit 1
fi
"$BETA_LEROBOT_PYTHON" -c 'from lerobot.datasets.lerobot_dataset import LeRobotDataset'
mkdir -p "$(dirname "$BETA_EXPORT")/_staging"
BETA_STAGE=$(mktemp -d "$(dirname "$BETA_EXPORT")/_staging/lerobot_capture.XXXXXX")
echo "Intermediate capture: $BETA_STAGE/raw"
bash "$BETA_REPO/data_pipeline/run.sh" "$BETA_STAGE/raw" "$BETA_EPISODES" "${@:3}"
cd "$BETA_REPO"
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE="$BETA_REPO/data_pipeline/.venv/cache/datasets"
export PYTHONDONTWRITEBYTECODE=1
"$BETA_LEROBOT_PYTHON" data_pipeline/validate_r2r.py "$BETA_STAGE/raw"
"$BETA_LEROBOT_PYTHON" data_pipeline/export_lerobot.py "$BETA_STAGE/raw" --output-dir "$BETA_EXPORT"
for BETA_EPISODE in "$BETA_STAGE/raw/episodes/"*; do
  "$BETA_LEROBOT_PYTHON" visual/render_episode_gt.py "$BETA_EPISODE" \
    --output "$BETA_EXPORT/review/episode_$(basename "$BETA_EPISODE")_gt.mp4" --fps 5 --horizon 10 --scale 2
done
# Only this invocation's scratch capture is removed, after official round-trip validation.
rm -rf -- "$BETA_STAGE"
"$BETA_LEROBOT_PYTHON" data_pipeline/validate_lerobot.py "$BETA_EXPORT"
