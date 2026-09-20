#!/usr/bin/env bash
set -euo pipefail
BETA_TRAIN_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BETA_TRAIN_ENV="$BETA_TRAIN_REPO/.venv-train"
BETA_BASE_SITE="$("$BETA_TRAIN_REPO/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
"$BETA_TRAIN_REPO/.venv/bin/python" -m venv --system-site-packages "$BETA_TRAIN_ENV"
"$BETA_TRAIN_ENV/bin/python" - "$BETA_TRAIN_REPO" "$BETA_BASE_SITE" <<'PY'
import sys
import sysconfig
from pathlib import Path
repo = Path(sys.argv[1])
(Path(sysconfig.get_path("purelib")) / "lightnav_train_runtime.pth").write_text(
    sys.argv[2] + "\n" + str(repo / "src") + "\n" + str(repo) + "\n"
)
PY
# Reuse Torch/Transformers without changing the inference or dataset-export envs.
"$BETA_TRAIN_ENV/bin/python" -m pip install \
    --index-url https://pypi.org/simple --cache-dir "$BETA_TRAIN_ENV/pip-cache" \
    --no-deps --retries 0 --timeout 20 pyarrow==25.0.1 accelerate==1.12.0
"$BETA_TRAIN_ENV/bin/python" - <<'PY'
import accelerate
import pyarrow
import torch
import transformers
from lightnav.training.train_betanav import main
print("BetaNav training runtime:", accelerate.__version__, pyarrow.__version__,
      torch.__version__, transformers.__version__)
assert callable(main)
PY
