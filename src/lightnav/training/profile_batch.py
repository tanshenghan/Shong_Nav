"""Bounded full-model batch/accumulation profiling; writes a report, no checkpoints.

Repeats one real long-history training frame on each rank. This measures model
capacity and compute throughput, not shuffled DataLoader or full-run throughput.
Launch with torchrun for the same DDP/ZeRO-1 behavior as train_betanav.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import torch

from lightnav.dataloader.Train_Lerobot_DataLoader import Train_Lerobot_DataLoader
from lightnav.training.checkpoint import atomic_json, on_main
from lightnav.training.config import load_config
from lightnav.training.performance import (
    GpuTelemetry,
    NativeFlopCounter,
    flop_rates,
    gpu_hardware,
)
from lightnav.training.runtime import (
    backward_sums,
    build_model,
    clip_and_step,
    make_optimizers,
    move_batch,
    normalize_gradients,
    read_data_manifest,
)


def select_frame(dataset, history_frame=None):
    """Select the longest episode with a complete FM-30 target, in dataset-local indices."""
    offset, candidates = 0, []
    for episode in dataset.episodes:
        frame = episode.length - 31 if history_frame is None else history_frame
        if 0 <= frame < episode.length - 30:
            candidates.append((episode.length, -episode.index, offset + frame))
        offset += episode.length
    if not candidates:
        raise ValueError("no training episode has the requested history and 30 future actions")
    return max(candidates)[2]


def run_update(model, cpu_batch, accelerator, config, accumulation, groups, optimizers, *,
               update_weights=True):
    """One update, with trace ranges; FLOP probes discard gradients instead of updating Adam."""
    counts = torch.zeros(2, dtype=torch.int64, device=accelerator.device)
    for micro in range(accumulation):
        with torch.profiler.record_function("betanav/host_to_device"):
            batch = move_batch(cpu_batch, accelerator.device)
        with accelerator.no_sync(model) if micro < accumulation - 1 else nullcontext():
            with torch.profiler.record_function("betanav/forward_ce_fm"):
                out = model(batch, lambda_fm=config.lambda_fm)
            if not torch.isfinite(out.loss):
                raise FloatingPointError("nonfinite profile loss")
            with torch.profiler.record_function("betanav/backward_and_ddp"):
                backward_sums(out, accelerator, config.lambda_fm)
        counts += counts.new_tensor([out.ce_token_count, out.fm_step_count if config.lambda_fm > 0 else 0])
        loss = float(out.loss.detach())
        del out, batch
    with torch.profiler.record_function("betanav/gradient_norm_and_adam"):
        counts = accelerator.reduce(counts, reduction="sum").tolist()
        normalize_gradients(groups, counts, accelerator.num_processes)
        if update_weights:
            clip_and_step(groups, optimizers, counts, config)
        else:
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
    return loss


def main():
    from accelerate import Accelerator
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        InitProcessGroupKwargs,
        set_seed,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--per-rank-batch", type=int, default=8,
                        help="microbatch * accumulation, held fixed across candidates")
    parser.add_argument("--no-accumulation", action="store_true",
                        help="test each candidate as a complete batch; effective batch varies by candidate")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2, help="measured optimizer updates per candidate")
    parser.add_argument("--history-frame", type=int,
                        help="zero-based t; default uses the longest complete-FM history")
    parser.add_argument("--count-flops", action="store_true", help="separate forward/backward shape-formula FLOP probe")
    parser.add_argument("--peak-bf16-tflops", type=float,
                        help="verified dense BF16 rated peak PER GPU; omitted means no HFU percentage")
    parser.add_argument("--trace-dir", type=Path, help="new directory for one extra update trace per candidate/rank")
    parser.add_argument("--telemetry", action="store_true", help="sample nvidia-smi to report-name.gpu.csv every 500 ms")
    args = parser.parse_args()
    if (args.per_rank_batch < 1 or args.warmup < 1 or args.steps < 1
            or any(b < 1 or (not args.no_accumulation and args.per_rank_batch % b) for b in args.batches)
            or args.history_frame is not None and args.history_frame < 0):
        parser.error("positive batch divisors, warmup/steps >= 1 and nonnegative history required")
    if args.peak_bf16_tflops is not None and (not args.count_flops or not math.isfinite(args.peak_bf16_tflops)
                                            or args.peak_bf16_tflops <= 0):
        parser.error("--peak-bf16-tflops requires --count-flops and a positive finite value")
    config = load_config(args.config)
    if config.cpu or config.tiny_model or not torch.cuda.is_available():
        parser.error("this profiler requires CUDA and the full pretrained model")
    torch.set_num_threads(config.cpu_threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    accelerator = Accelerator(mixed_precision=config.mixed_precision, gradient_accumulation_steps=1,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True,
                         broadcast_buffers=False, gradient_as_bucket_view=True),
                         InitProcessGroupKwargs(timeout=timedelta(minutes=10))])
    if accelerator.distributed_type not in {DistributedType.NO, DistributedType.MULTI_GPU}:
        raise ValueError("requires plain DDP/ZeRO-1; no DeepSpeed/FSDP launcher")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = accelerator.device
    set_seed(config.seed)

    def reserve_report():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            stream.write('{"status": "initializing"}\n')

    on_main(accelerator, reserve_report)
    if args.trace_dir:
        on_main(accelerator, lambda: args.trace_dir.mkdir(parents=True, exist_ok=False))
    manifest = on_main(accelerator, lambda: read_data_manifest(config))
    loader = Train_Lerobot_DataLoader(config.dataset_root, config.model_path, num_workers=0,
                                      training_mode=config.training_mode,
                                      episode_indices=manifest["train_episodes"], shuffle=False)
    item = loader.dataset[select_frame(loader.dataset, args.history_frame)]
    one = loader.collate_fn([item])
    report = dict(status="running", config=asdict(config), world_size=accelerator.num_processes,
                  torch_version=torch.__version__, gpu=torch.cuda.get_device_name(device),
                  device_total_gib=torch.cuda.get_device_properties(device).total_memory / 2**30,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  sample=item["info"], sequence_length=one.labels.shape[1],
                  history_frames=sum(len(s["frame_ids"]) for s in item["info"]["history"]),
                  effective_batch_size=None if args.no_accumulation else args.per_rank_batch * accelerator.num_processes,
                  hardware=on_main(accelerator, lambda: gpu_hardware(device)),
                  warmup_updates=args.warmup, measured_updates=args.steps, results=[],
                  limitations="Repeated real frame, cached CPU input; excludes data-loading time. "
                  "Optimizer states/gradients included. Tested world size only; no training checkpoint saved.")
    del one
    on_main(accelerator, lambda: atomic_json(args.report, report))
    accelerator.print("Loading full pretrained weights; sample:",
                      {k: report[k] for k in ("sequence_length", "history_frames", "world_size")}, flush=True)
    model = accelerator.prepare_model(build_model(config, manifest["normalization"]["action_scale"]))
    bare = accelerator.unwrap_model(model)
    groups = bare.optimizer_parameters()
    optimizers = make_optimizers(bare, config, accelerator.num_processes)
    report["parameters"] = dict(total=sum(p.numel() for p in bare.parameters()),
                               **{k: sum(p.numel() for p in ps) for k, ps in groups.items()})
    set_seed(config.seed + accelerator.process_index)
    for optimizer in optimizers.values():
        optimizer.zero_grad(set_to_none=True)
    telemetry = None
    try:
        if args.telemetry:
            report["telemetry_csv"] = str(args.report.with_suffix(".gpu.csv"))
            def start_telemetry():
                nonlocal telemetry
                telemetry = GpuTelemetry(report["telemetry_csv"])
                telemetry.__enter__()
            on_main(accelerator, start_telemetry)
        for batch_size in args.batches:
            accumulation = 1 if args.no_accumulation else args.per_rank_batch // batch_size
            effective_batch = batch_size * accumulation * accelerator.num_processes
            accelerator.print(f"Profiling batch={batch_size}, accumulation={accumulation}", flush=True)
            start = time.perf_counter()
            cpu_batch = loader.collate_fn([item] * batch_size)
            collation_seconds = time.perf_counter() - start
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            durations, losses = [], []
            for step in range(args.warmup + args.steps):
                accelerator.wait_for_everyone()
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                loss = run_update(model, cpu_batch, accelerator, config, accumulation, groups, optimizers)
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                times = accelerator.gather(torch.tensor([elapsed], device=device)).tolist()
                if step >= args.warmup:
                    durations.append(max(times))
                    losses.append(loss)
                accelerator.print(f"  update {step + 1}: {max(times):.3f}s, loss={loss:.5f}", flush=True)
            memory = torch.tensor([[torch.cuda.max_memory_allocated(device) / 2**30,
                                    torch.cuda.max_memory_reserved(device) / 2**30,
                                    torch.cuda.memory_allocated(device) / 2**30]], device=device)
            memory = accelerator.gather(memory).tolist()
            result = dict(batch_size=batch_size, accumulation=accumulation,
                          effective_batch_size=effective_batch,
                          update_seconds=durations, mean_update_seconds=sum(durations) / len(durations),
                          samples_per_second=effective_batch * len(durations) / sum(durations),
                          peak_allocated_gib_by_rank=[row[0] for row in memory],
                          peak_reserved_gib_by_rank=[row[1] for row in memory],
                          persistent_allocated_gib_by_rank=[row[2] for row in memory],
                          cpu_collation_seconds=collation_seconds, last_losses_rank0=losses)
            # Keep completed timing results even if optional instrumentation fails.
            report["results"].append(result)
            on_main(accelerator, lambda: atomic_json(args.report, report))
            if args.count_flops:
                # Counting runs separately: TorchDispatchMode overhead must not inflate measured time.
                with NativeFlopCounter() as counter:
                    run_update(model, cpu_batch, accelerator, config, accumulation, groups, optimizers,
                               update_weights=False)
                flops = accelerator.gather(torch.tensor([counter.get_total_flops()], dtype=torch.int64,
                                                        device=device)).tolist()
                result["flops"] = dict(per_update_by_rank=flops,
                    rank0_by_operator={str(k): v for k, v in counter.get_flop_counts()["Global"].items()},
                    includes="registered matmul/conv/attention forward, backward and checkpoint recomputation",
                    excludes="unsupported/elementwise ops, optimizer arithmetic, transfers and NCCL; "
                             "masked attention uses dense formula; not hardware counters or strict MFU",
                    **flop_rates(flops, result["mean_update_seconds"], args.peak_bf16_tflops))
            if args.trace_dir:
                accelerator.wait_for_everyone()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as trace:
                    run_update(model, cpu_batch, accelerator, config, accumulation, groups, optimizers)
                    torch.cuda.synchronize(device)
                trace_path = args.trace_dir / f"batch-{batch_size}-rank-{accelerator.process_index}.json"
                trace.export_chrome_trace(str(trace_path))
                result["rank0_trace"] = str(args.trace_dir / f"batch-{batch_size}-rank-0.json")
                table_path = trace_path.with_suffix(".txt")
                table_path.write_text(trace.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
            on_main(accelerator, lambda: atomic_json(args.report, report))
            accelerator.print(json.dumps(result), flush=True)
            del cpu_batch
        report["status"] = "complete"
        on_main(accelerator, lambda: atomic_json(args.report, report))
    except BaseException as exc:
        if accelerator.is_main_process:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            atomic_json(args.report, report)
        raise
    finally:
        if telemetry is not None:
            telemetry.__exit__(None, None, None)
        # An OOM aborts the torchrun job; earlier completed candidates remain in the report.
        # Do not attempt collective recovery while a peer may still be inside backward.
        accelerator.end_training()


if __name__ == "__main__":
    main()
