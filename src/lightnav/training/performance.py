"""GPU baselines and explicit FLOP accounting for bounded training measurements."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def attention_flops(query, key, value, *, backward=False):
    """Dense SDPA matmul accounting with GQA; backward includes score recomputation.

    Mirrors PyTorch 2.10's SDPA formula, allowing Hq to be a multiple of Hkv.
    Causal/padding masks do not reduce this mathematical upper-bound count.
    """
    b, h, sq, dq = query
    bk, hk, sk, dk = key
    bv, hv, sv, dv = value
    if b != bk or b != bv or hk != hv or h % hk or dq != dk or sk != sv:
        raise ValueError("incompatible SDPA/GQA shapes")
    return 2 * b * h * sq * sk * (3 * dq + 2 * dv if backward else dq + dv)


class NativeFlopCounter(TorchDispatchMode):
    """Count registered formulas without decomposing/replacing the actual fused ops.

    PyTorch 2.10 FlopCounterMode decomposes unregistered ops and adds module hooks;
    that can change the execution graph/memory footprint. Keep unsupported ops at
    zero instead and report this explicit coverage limit.
    """

    def __init__(self):
        from torch.utils.flop_counter import flop_registry

        super().__init__()
        self.counts = defaultdict(int)
        self.formulas = dict(flop_registry)

        def forward(q, k, v, *args, **kwargs):
            return attention_flops(q.shape, k.shape, v.shape)

        def backward(grad, q, k, v, *args, **kwargs):
            return attention_flops(q.shape, k.shape, v.shape, backward=True)

        for backend in ("efficient", "flash", "cudnn"):
            self.formulas[getattr(torch.ops.aten, f"_scaled_dot_product_{backend}_attention")] = forward
            self.formulas[getattr(torch.ops.aten, f"_scaled_dot_product_{backend}_attention_backward")] = backward
        self.formulas[torch.ops.aten._scaled_dot_product_flash_attention_for_cpu] = forward
        self.formulas[torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward] = backward

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        packet = func._overloadpacket
        if packet in self.formulas:
            self.counts[packet] += int(self.formulas[packet](*args, **kwargs, out_val=out))
        return out

    def get_total_flops(self):
        return sum(self.counts.values())

    def get_flop_counts(self):
        return {"Global": dict(self.counts)}


def flop_rates(flops_by_rank, seconds, peak_bf16_tflops=None):
    """Shape-formula FLOPs / uninstrumented wall time, not hardware instruction counters."""
    if not seconds > 0 or not math.isfinite(seconds) or not flops_by_rank or min(flops_by_rank) < 0:
        raise ValueError("positive finite duration and nonnegative per-rank FLOPs required")
    if peak_bf16_tflops is not None and (not math.isfinite(peak_bf16_tflops) or peak_bf16_tflops <= 0):
        raise ValueError("peak must be a positive finite dense BF16 TFLOPS value per GPU")
    cluster = sum(flops_by_rank) / seconds / 1e12
    return dict(estimated_cluster_tflops=cluster, estimated_per_gpu_tflops=cluster / len(flops_by_rank),
                estimated_hfu=None if peak_bf16_tflops is None else cluster / (len(flops_by_rank) * peak_bf16_tflops),
                peak_dense_bf16_tflops_per_gpu=peak_bf16_tflops)


def gpu_hardware(device):
    p = torch.cuda.get_device_properties(device)
    bus = f"{p.pci_domain_id:08x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
    fields = "name,pci.device_id,pci.sub_device_id,driver_version,power.limit,clocks.max.sm,clocks.max.memory"
    query = subprocess.run(["nvidia-smi", "-i", bus, f"--query-gpu={fields}",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return dict(cuda_name=p.name, compute_capability=[p.major, p.minor], multiprocessors=p.multi_processor_count,
                total_gib=p.total_memory / 2**30, pci_bus_id=bus, torch_version=torch.__version__,
                nvidia_smi=dict(zip(fields.split(","), [x.strip() for x in query.stdout.strip().split(",")], strict=True)))


def gemm_baseline(device, sizes=(4096, 8192), iterations=50):
    """Dense BF16 cuBLAS GEMM, FMA=2 FLOPs; a measured local reference, not a rated peak."""
    if iterations < 1 or not sizes or any(n < 1 for n in sizes):
        raise ValueError("positive matrix sizes and iteration count required")
    result = []
    with torch.cuda.device(device), torch.no_grad(), torch.random.fork_rng(devices=[device.index]):
        torch.cuda.manual_seed(123)
        for n in sizes:
            a = torch.randn((n, n), device=device, dtype=torch.bfloat16)
            b, c = torch.randn_like(a), torch.empty_like(a)
            for _ in range(20):
                torch.mm(a, b, out=c)
            torch.cuda.synchronize(device)
            times = []
            for _ in range(3):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    torch.mm(a, b, out=c)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end) / iterations)
            median = statistics.median(times)
            result.append(dict(size=n, iterations=iterations, milliseconds=times,
                               median_ms=median, dense_bf16_tflops=2 * n**3 / (median / 1e3) / 1e12))
            del a, b, c
    return result


class GpuTelemetry:
    """Low-rate nvidia-smi CSV capture; all GPU rows keep their physical indices."""

    def __init__(self, path):
        self.path, self.process, self.stream = Path(path), None, None

    def __enter__(self):
        self.stream = self.path.open("x")
        fields = ("timestamp,index,pci.bus_id,utilization.gpu,utilization.memory,memory.used,"
                  "power.draw,power.limit,temperature.gpu,clocks.current.sm,clocks.current.memory")
        try:
            self.process = subprocess.Popen(["nvidia-smi", f"--query-gpu={fields}",
                "--format=csv,nounits", "--loop-ms=500"], stdout=self.stream, stderr=subprocess.DEVNULL)
        except BaseException:
            self.stream.close()
            raise
        return self

    def __exit__(self, *_args):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.stream is not None:
            self.stream.close()


def main():
    parser = argparse.ArgumentParser(description="Measure local dense BF16 GEMM throughput without loading a model")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0, help="logical CUDA index after CUDA_VISIBLE_DEVICES")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        device = torch.device("cuda", args.device)
        result = dict(hardware=gpu_hardware(device), gemm=gemm_baseline(device, args.sizes, args.iterations),
                      caveat="Measured dense BF16 GEMM reference, not vendor peak or training MFU.")
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
