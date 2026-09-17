#!/usr/bin/env bash
set -euo pipefail
BETA_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BETA_ENV="$BETA_REPO/data_pipeline/.venv/lerobot"
"$BETA_REPO/.venv/bin/python" -m venv --system-site-packages "$BETA_ENV"
# Local dataset dependencies take priority; borrow the existing compatible torch wheels.
"$BETA_ENV/bin/python" - "$BETA_REPO" <<'PY'
import sys
import sysconfig
from pathlib import Path
repo = Path(sys.argv[1])
(Path(sysconfig.get_path("purelib")) / "lightnav_runtime.pth").write_text(
    str(repo / ".venv/lib/python3.11/site-packages") + "\n")
PY
"$BETA_ENV/bin/python" -m pip install --index-url https://pypi.org/simple --no-deps lerobot==0.4.4
"$BETA_ENV/bin/python" -m pip install --index-url https://pypi.org/simple -r "$BETA_REPO/data_pipeline/requirements-lerobot.txt"
"$BETA_ENV/bin/python" - <<'PY'
import torch
import torchvision
from lerobot.datasets.lerobot_dataset import LeRobotDataset
print("Dataset runtime ready:", LeRobotDataset.__name__, torch.__version__, torchvision.__version__)
PY
