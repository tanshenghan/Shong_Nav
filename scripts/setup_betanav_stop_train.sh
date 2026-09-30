#!/usr/bin/env bash
# Dedicated executable training environment; reuse installed packages without downloads.
set -Eeuo pipefail

BETA_STOP_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BETA_STOP_ENV="$BETA_STOP_REPO/.venv-train-stop"
BETA_STOP_BOOTSTRAP_PYTHON="${BETA_STOP_BOOTSTRAP_PYTHON:-$BETA_STOP_REPO/.venv-tools/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11}"

# Serialize first-time preparation across jobs sharing this checkout.
mkdir -p "$BETA_STOP_ENV"
exec 9>"$BETA_STOP_ENV/.setup.lock"
flock 9
if [[ ! -x "$BETA_STOP_ENV/bin/python" ]]; then
    if [[ ! -x "$BETA_STOP_BOOTSTRAP_PYTHON" ]]; then
        echo "Set BETA_STOP_BOOTSTRAP_PYTHON to an accessible Python 3.11 interpreter." >&2
        exit 1
    fi
    "$BETA_STOP_BOOTSTRAP_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 required"'
    "$BETA_STOP_BOOTSTRAP_PYTHON" -m venv --without-pip "$BETA_STOP_ENV"
fi

"$BETA_STOP_ENV/bin/python" - "$BETA_STOP_REPO" <<'PY'
import sys
import sysconfig
from pathlib import Path

repo = Path(sys.argv[1])
sites = [repo / ".venv/lib/python3.11/site-packages",
         repo / ".venv-train/lib/python3.11/site-packages"]
for site in sites:
    if not site.is_dir():
        raise SystemExit(f"Missing installed training dependencies: {site}")
# Only the new environment is changed. Existing runtime/package installations stay intact.
target = Path(sysconfig.get_path("purelib")) / "lightnav_train_runtime.pth"
target.write_text("\n".join(map(str, [*sites, repo / "src", repo])) + "\n")
PY

"$BETA_STOP_ENV/bin/python" - <<'PY'
import accelerate
import pyarrow
import torch
import transformers
from lightnav.training.train_betanav import train

assert callable(train)
print("FM+Stop training environment:", "torch=" + torch.__version__,
      "transformers=" + transformers.__version__, "accelerate=" + accelerate.__version__,
      "pyarrow=" + pyarrow.__version__, flush=True)
PY
