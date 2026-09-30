"""Prepare and launch a foreground, single-node FM + StopHead platform job."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from lightnav.training.config import load_config
from lightnav.training.prepare_data import make_manifest
from lightnav.training.runtime import read_data_manifest

REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = "configs/betanav/r2r_fm_stop_h30_prefix_kv_all36_beta1_1p5.yaml"
NORMALIZATION_SOURCE = "configs/betanav/data/r2r_all_train_20260913.json"


def publish_manifest(path, manifest):
    """Publish a complete JSON atomically, preserving a concurrent writer's file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass  # The caller validates whichever complete file won publication.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY / DEFAULT_CONFIG)
    parser.add_argument("--num-gpus", type=int, help="visible local GPUs; defaults to CUDA device count")
    parser.add_argument("--batch-size", type=int, help="per-GPU microbatch; defaults to the YAML")
    parser.add_argument("--global-batch", type=int, default=512)
    parser.add_argument("--output-dir")
    parser.add_argument("--data-manifest")
    parser.add_argument("--resume", nargs="?", const="auto", help="explicit exact resume; default is a fresh run")
    parser.add_argument("--prepare-only", action="store_true",
                        help="audit data and print the launch command without CUDA; requires --num-gpus")
    args = parser.parse_args(argv)
    if args.prepare_only and args.num_gpus is None:
        parser.error("--prepare-only requires an explicit --num-gpus")
    config = load_config(args.config, dict(batch_size=args.batch_size, output_dir=args.output_dir,
                                           data_manifest=args.data_manifest))
    if config.training_mode != "fm_stop" or config.cpu or config.tiny_model:
        raise ValueError("platform launcher requires fm_stop with the real LightNav backbone on CUDA")
    if config.num_epochs != 3 or config.eval_every or config.save_best:
        raise ValueError("R2R platform training requires num_epochs=3, eval_every=0 and save_best=false")
    for name in ("dataset_root", "model_path", "data_manifest", "output_dir"):
        path = Path(getattr(config, name))
        setattr(config, name, str((REPOSITORY / path).resolve()))
    output = Path(config.output_dir)
    if not args.resume and output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("fresh output_dir must be empty; select --output-dir or explicitly --resume auto")
    visible = None
    if not args.prepare_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; the platform must allocate GPUs before launching")
        visible = torch.cuda.device_count()
    count = args.num_gpus if args.num_gpus is not None else visible
    if count is None or count < 1:
        raise ValueError("--num-gpus must be positive")
    if visible is not None and count > visible:
        raise ValueError(f"requested {count} GPUs but only {visible} are visible to this platform job")
    per_window = count * config.batch_size
    if args.global_batch < 1 or args.global_batch % per_window:
        raise ValueError("--global-batch must be positive and divisible by num_gpus * batch_size")
    config.gradient_accumulation_steps = args.global_batch // per_window
    config.validate()
    if not args.prepare_only and config.mixed_precision == "bf16":
        for index in range(count):
            with torch.cuda.device(index):
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError(f"BF16 is unavailable on visible GPU {index}")

    manifest_path = Path(config.data_manifest)
    if not manifest_path.exists():
        if args.resume:
            raise FileNotFoundError("resume requires the original STOP manifest; do not regenerate it")
        print(f"Auditing STOP labels: {manifest_path}", flush=True)
        manifest = make_manifest(config.dataset_root, seed=config.seed, val_fraction=0,
                                 normalization_from=REPOSITORY / NORMALIZATION_SOURCE,
                                 with_stop_labels=True)
        publish_manifest(manifest_path, manifest)
    manifest = read_data_manifest(config)
    if manifest.get("split_by") != "all" or manifest["val_episodes"] or manifest["val_frames"]:
        raise ValueError("this R2R launcher requires all episodes for training and no validation split")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    launch = REPOSITORY / "runs/betanav_launch" / f"{stamp}_{uuid.uuid4().hex[:8]}_fm_stop_platform"
    launch.mkdir(parents=True)
    effective = launch / "effective_train_config.yaml"
    effective.write_text(yaml.safe_dump(asdict(config), sort_keys=False))
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc_per_node={count}", str(REPOSITORY / "train_betanav.py"), "--config", str(effective)]
    if args.resume:
        command.extend(["--resume", args.resume])
    metadata = dict(status="prepared" if args.prepare_only else "dispatching", training_mode="fm_stop",
                    initialization="resume" if args.resume else "fresh", resume=args.resume,
                    repository=str(REPOSITORY), configuration=str(effective), output_dir=config.output_dir,
                    data_manifest=config.data_manifest, model_path=config.model_path,
                    interpreter=sys.executable, num_gpus=count, visible_gpus=visible,
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    batch_size=config.batch_size, gradient_accumulation_steps=config.gradient_accumulation_steps,
                    global_batch=args.global_batch, num_epochs=config.num_epochs, command=command)
    (launch / "launch.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    print(shlex.join(command), flush=True)
    if args.prepare_only:
        return metadata
    os.chdir(REPOSITORY)
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
