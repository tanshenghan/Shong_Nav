#!/usr/bin/env bash
set -euo pipefail
BETA_TRAIN_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Compatibility entry point; training setup and its environment live outside this directory.
exec bash "$BETA_TRAIN_REPO/scripts/setup_betanav_train.sh"
