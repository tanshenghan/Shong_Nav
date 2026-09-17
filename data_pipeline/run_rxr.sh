#!/usr/bin/env bash
set -euo pipefail
# Usage: bash data_pipeline/run_rxr.sh OUTPUT_DIR [--split train --role guide --languages all ...]
BETA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BETA_RXR_OUTPUT=${1:?supply a NEW RxR LeRobotDataset directory}
shift
exec bash "$BETA_REPO/data_pipeline/run_r2r_train.sh" "$BETA_RXR_OUTPUT" --dataset rxr "$@"
