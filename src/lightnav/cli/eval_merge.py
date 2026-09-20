"""``lightnav-eval-merge``: merge parallel Habitat evaluation shards into one summary.

Usage::

    lightnav-eval-merge output/r2r                 # merges output/r2r/*/results.jsonl -> output/r2r/
    lightnav-eval-merge output/r2r/shard_0 output/r2r/shard_1 --output output/r2r/merged
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lightnav-eval-merge",
        description="Merge the per-shard results.jsonl of a parallel Habitat evaluation "
        "into one results.jsonl + summary.json.",
    )
    p.add_argument(
        "paths",
        nargs="+",
        help="shard output directories (each holding results.jsonl), or a parent directory "
        "whose immediate children are the shards",
    )
    p.add_argument(
        "--output",
        default=None,
        help="where to write the merged results.jsonl + summary.json "
        "(default: the single parent directory given, else ./merged)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from lightnav.habitat.merge import find_shard_dirs, merge_results

    shards = find_shard_dirs(args.paths)
    if not shards:
        print(f"no results.jsonl found under: {', '.join(args.paths)}", file=sys.stderr)
        return 1
    if args.output:
        output = Path(args.output)
    elif len(args.paths) == 1 and Path(args.paths[0]).resolve() not in {s.resolve() for s in shards}:
        output = Path(args.paths[0])
    else:
        output = Path("merged")
    summary = merge_results(shards, output)
    return 0 if summary is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
