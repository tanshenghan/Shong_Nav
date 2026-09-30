#!/usr/bin/env python3
"""Compatibility entry point; maintain collection code in data_pipeline/R2R/ only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.R2R.collect_r2r import main  # noqa: E402

if __name__ == "__main__":
    main()
