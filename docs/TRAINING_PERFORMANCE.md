# BetaNav：观察训练性能、TFLOPS 与峰值差距

对冻结 VLM 的 FM-only 实验，使用 `configs/betanav/r2r_fm_only.yaml`。
容量测量可传 `--no-accumulation --batches 8 16 32`，此时每个候选 batch 只更新一次，
有效 batch 随候选值变化，`--per-rank-batch` 不生效。该开关用于比较增大 batch 后的
吞吐与显存；没有设置时仍按原先固定有效 batch、改变梯度累积的方式比较。
每个候选都包含 FM 反向与 Adam 更新，但 VLM 无反向。长历史探测后仍需留出随机批次的
长度差异和 CUDA/NCCL 缓冲空间，不能把 reserved 显存或 GPU-util 直接等同于有效算力。

训练入口为 [train_betanav.py](../train_betanav.py)，配置仍是
[r2r.yaml](../configs/betanav/r2r.yaml)。本页所有命令在仓库根目录使用 `.venv-train/`。
保持 CE + 2.0×FM、每卡 batch=8、8卡有效 batch=64。

## 三种利用率分别看什么

| 指标 | 能回答的问题 | 不能直接推出的结论 |
|---|---|---|
| `nvidia-smi` GPU-Util | 采样时间内有 kernel 在执行的时间比例 | 100% 不代表达到理论 TFLOPS |
| 显存 used / allocated / reserved | 容量余量、张量和缓存占用 | 显存占满不代表算得快 |
| step 秒数 / samples/s / input tokens/s | 同一模型、数据和 batch 下实际速度 | 不能跨不同图像历史、模型直接比较 |
| 训练 FLOPs ÷ 秒数 | 指定计算量口径下的平均 TFLOPS | FLOPs 公式不是硬件指令计数 |
| MFU / HFU | 相对同精度硬件峰值的计算效率 | 不能拿 BF16 训练对比 FP4/INT8 或稀疏峰值 |

[NVIDIA 定义](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization)中，
GPU-Util 是至少一个 kernel 在执行的时间占比；memory utilization 是显存读写活跃的时间占比，
也不是显存容量占比或带宽达到峰值的比例。NCCL 通信 kernel 也可能让 GPU 保持“忙碌”。

## 日常训练：先看系统状态和日志

另开终端，每秒显示状态（Ctrl-C 只结束观察）：

```bash
nvidia-smi \
  --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.current.sm \
  --format=csv --loop=1
```

需要记录成 CSV 时，使用一个新的文件名，并重定向输出即可。
若功耗/时钟明显受限，使用 `nvidia-smi -q -d PERFORMANCE,POWER,CLOCK` 看当前限制原因；
该命令也包含自设备初始化以来的累计计数，不应把历史累计值当成本轮发生的限频。

训练日志在配置的 `output_dir/metrics.jsonl`。当前路径：

```bash
tail -f /mnt/cfs/Models/beta_nav/r2r_ce_fm_01_start/metrics.jsonl
```

除 CE、FM、梯度范数和学习率外，训练器现有以下字段：

| 日志字段 | 定义 |
|---|---|
| `update_seconds` | 记录的这次 optimizer 更新，各 rank 耗时的最大值；记录步骤前后同步 CUDA |
| `samples_per_second` | 本次全局样本数 / 上述秒数 |
| `input_tokens_per_second` | 全局非 padding 输入 token 数 / 秒；包含视觉占位 token，不是 CE 标签数 |
| `update_seconds_by_rank` | 每个 rank 的更新耗时，用于观察慢卡/不同长度输入 |
| `data_wait_seconds_by_rank` / `data_wait_seconds_max` | 主训练线程等待 DataLoader 返回 batch 的 CPU 墙钟时间 |
| `peak_memory_gib` | 本进程启动以来、所有 rank 中最大的 PyTorch allocated 峰值 |
| `peak_reserved_memory_gib` | 相同口径的 allocator reserved 峰值 |
| `padding_fraction` | 这次更新的 padding 占比；视频历史差异大时尤其值得看 |

默认每10步记录一次，这些是抽样更新的耗时，不是最近10步总平均。
时间覆盖数据等待、传输、前反向、通信、梯度裁剪和 AdamW，不包括随后验证/保存/日志写入。
worker 可能提前准备下一批，所以数据等待不是完整解码成本，也不能直接当成 GPU 空闲时间。
每卡数据等待最大值与耗时最大值可能来自不同 rank，不能简单相减得到纯 GPU 计算时间。
TensorBoard/WandB 可用原有 `tracker` 配置启用；每卡数组在平台中展开为 `/rank_0` 等标量曲线。

## 一次短测：完整模型耗时、FLOPs、GPU 状态和 CUDA 时间线

```bash
CUDA_VISIBLE_DEVICES=1,2 NCCL_DEBUG=WARN \
.venv-train/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 \
  -m lightnav.training.profile_batch \
  --config configs/betanav/r2r.yaml \
  --batches 8 --per-rank-batch 8 --warmup 2 --steps 5 \
  --count-flops --telemetry \
  --trace-dir runs/betanav_profile/NEW_TRACE \
  --report reports/betanav/NEW_PERFORMANCE.json
```

只加载完整模型短测，临时权重在退出时丢弃，不使用训练输出目录，不写模型 checkpoint。
默认从最长训练 episode 中选择保留完整 FM-30 的长历史帧，也可加 `--history-frame 20`。
每卡重复同一条真实样本，CPU batch 已缓存，因此测的是模型计算能力；正式数据读取速度看训练日志。
对比8卡通信开销，需要在相同输入/配置下真正启动8个进程，不能把双卡吞吐简单乘4。

性能测量分开执行：

1. 预热后计时，不开 FLOPs 计数或 CUDA profiler；包含反向和两套 AdamW。
2. 额外一次相同 batch 的前反向，仅计数并丢弃梯度，保留融合算子，不改变权重/优化器状态。
3. 若指定 trace，再额外执行一次更新，导出每个 rank 的时间线和算子耗时表。

| 追加参数 | 说明 |
|---|---|
| `--count-flops` | 估算注册算子的前反向 FLOPs，包含 checkpoint 重计算 |
| `--peak-bf16-tflops` | 可选，需要同时开启计数；必须是已确认的**单卡 dense BF16**理论峰值 |
| `--trace-dir` | 可选，新目录；每个候选/rank 生成 `batch-8-rank-0.json` 和 `.txt` |
| `--telemetry` | 可选，500ms采样一次，生成报告同名 `.gpu.csv`；CSV 包含全部物理 GPU 的 index |

报告记录 `flops.per_update_by_rank`、`estimated_cluster_tflops`、
`estimated_per_gpu_tflops`、各算子 FLOPs、输入长度、硬件标识和计时条件。
未提供官方峰值时 `estimated_hfu=null`，不根据显示名称猜测百分比。

基本公式：

```text
单卡平均估算 TFLOPS = 所有 rank 本次更新 FLOPs 之和 / 更新秒数 / GPU 数量 / 10^12
估算 HFU = 上述 TFLOPS / 单卡 dense BF16 理论峰值
MFU = 不含重计算的模型有效 FLOPs / 秒数 / GPU 数 / 单卡理论峰值
```

这里的计数使用 PyTorch 注册的 matmul/conv/attention 公式，额外支持 Qwen 的 GQA，
FMA 计2次运算。原生算子直接执行；不通过分解未知算子改变运行图。
包括冻结 ViT 的前向、语言主干/专家前反向、激活 checkpoint 重计算和 attention backward
公式中的 score 重算。未注册算子、逐元素操作、Adam 算术、数据复制和 NCCL 不计入 FLOPs；
attention 使用不扣除 causal/padding mask 的 dense 公式。它是明确口径的**估算值**，
不是硬件实际执行指令数，也不是严格 MFU。不能直接用 `6×全部参数×token数` 代替：
本模型冻结视觉部分，视觉与文本长度不同，LM head 只投影受监督位置。

[PyTorch FLOP 计数说明](https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py)
解释了公式/算子覆盖的限制；[激活重计算说明](https://pytorch.org/blog/activation-checkpointing-techniques/)
解释了显存与额外计算的关系。

## 如何查看耗时瓶颈

打开 [Perfetto UI](https://ui.perfetto.dev/)，选择 Open trace file，加载生成的 rank JSON。
格式来自 [PyTorch Profiler](https://docs.pytorch.org/docs/2.10/profiler.html)，
使用方式见 [Perfetto 文档](https://perfetto.dev/docs/visualization/perfetto-ui)。
本机 `.txt` 文件可直接看最耗时算子，不需要安装 UI。

时间线标注 `betanav/host_to_device`、`forward_ce_fm`、`backward_and_ddp`、
`gradient_norm_and_adam`；它们是 CPU 调用区间，GPU kernel 轨道显示真正执行和重叠关系。
重点看 GPU 空隙、GEMM/attention kernel、NCCL all-reduce/broadcast、拷贝和 AdamW。
CPU 调用耗时不是纯 GPU 执行时间；多个 stream 的 kernel 时长相加也不等于总墙钟时间。

| 观察到的现象 | 下一步检查 |
|---|---|
| GPU 经常空闲，DataLoader 等待长 | CPU 解码、共享存储读取、worker 数量及预取 |
| padding 高，rank 耗时差异大 | 视频长度分布；可再评估按长度分组采样 |
| NCCL 占比大 | 卡间拓扑、梯度通信、ZeRO-1 参数广播；用相同 workload 对比单/双/八卡 |
| GPU-Util 高但多数时间是小 kernel/逐元素操作 | kernel 启动开销、优化器、可融合操作 |
| GEMM 明显低于本机同精度基线 | 矩阵形状、dtype、Tensor Core、功耗和时钟 |

本机已安装 `ncu`（Nsight Compute），可进一步查看单 kernel 的 Tensor Core/SM/显存带宽及
roofline，见 [NVIDIA Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。
这是比 GPU-Util 更细的硬件观测，需要驱动允许性能计数器访问。本轮 CUDA trace 不使用这些硬件计数器。

## 本机的硬件与比较基线

2026-09-13，驱动显示 `NVIDIA RPBZZZ6`；完整 PCI 标识是 `2BB9 10DE 2091`，
在 [NVIDIA 官方设备表](https://download.nvidia.com/XFree86/Linux-x86_64/590.48.01/README/supportedchips.html)
对应 **NVIDIA RTX 6000D**。CUDA 报告 sm_120、188 SM、可见约94.97 GiB，功耗上限600W。
`nvidia-smi topo -m` 显示卡间 PHB/SYS，未显示 NVLink；8卡横跨两个 NUMA 域。
本轮没有确认该 SKU 的官方 dense BF16 峰值，所以不套用 H100/B200 或 RTX PRO 6000 的规格。

本地 BF16 基线命令（不加载训练模型）：

```bash
CUDA_VISIBLE_DEVICES=1 .venv-train/bin/python -m lightnav.training.performance \
  --report reports/betanav/NEW_GEMM.json
```

参数：`--report` 必填新文件，`--device` 默认0（可见卡中的逻辑编号），
`--sizes` 默认4096/8192，`--iterations` 默认50。每种矩阵预热20次、用 CUDA events
测量3组并取中位数，按 `2*M*N*K/time` 计算，输入为 BF16 dense GEMM。

[2026-09-13 基线](../reports/betanav/gemm_baseline_20260913.json)：
4096方阵为378.55 TFLOPS，8192方阵为358.42 TFLOPS。
这是所测两个形状的本机实测参考，不是该 GPU 的理论峰值或最优 GEMM 极限。
训练包含注意力、视觉处理、参数更新和通信，其平均 TFLOPS 低于大 GEMM 很正常，
差距的原因要结合时间线定位，不能直接归为“GPU 没有利用起来”。

## 完整 BetaNav 实测结果（2026-09-13）

[原始报告](../reports/betanav/training_performance_native_2gpu_20260913.json)：
物理 GPU 1、2，完整预训练模型，BF16/DDP/ZeRO-1，batch=8、累积1；
episode 4112 的 t=152，82张历史 RGB、1411输入 token、有效 FM-30。
预热2个更新、计时5个更新，不含随机数据读取。

| 指标 | 结果 |
|---|---:|
| 双卡每次更新样本数 | 16 |
| 每次更新平均墙钟时间 | 3.58165秒 |
| 双卡吞吐 | 4.46721样本/秒 |
| 每卡每次更新估算 FLOPs | 454.267万亿次 |
| 双卡合计估算 TFLOPS | 253.664 |
| 每卡平均估算 TFLOPS | 126.832 |
| 相对本机358.42–378.55 TFLOPS的GEMM参考 | 33.5%–35.4% |
| 官方峰值 HFU / 严格 MFU | 未给出；分别需要确认硬件峰值和不含重计算的模型计算量 |

126.832来自 `454.267 / 3.58165`。计时与计数分开，计数保留原生融合算子。
完整前反向计数和 CUDA trace 均实际跑通，不使用 tiny model 推算这些数字。

分析产物：

- [rank 0 算子耗时表](../runs/betanav_profile/performance_native_20260913/batch-8-rank-0.txt)
- [rank 0 CUDA trace](../runs/betanav_profile/performance_native_20260913/batch-8-rank-0.json)
- [rank 1 CUDA trace](../runs/betanav_profile/performance_native_20260913/batch-8-rank-1.json)
- [GPU 采样 CSV](../reports/betanav/training_performance_native_2gpu_20260913.gpu.csv)
- [观测摘要](../reports/betanav/performance_observation_20260913.json)

单次插桩 trace 中，rank 0 的 NCCL kernel 累计约1.85秒，其中 all-reduce 约1.527秒，
broadcast 约0.322秒，此外还有大量逐元素和拷贝操作。NCCL 是值得优先分析的开销。
这些是有 profiler 开销的 kernel 时长累计；stream 可能重叠，不能用1.85除以未插桩3.58
直接宣布“通信占了52%的训练时间”。GPU-Util 也会包含通信 kernel。
trace 文件每卡约140MB，保存在忽略的大产物目录 `runs/`。

计数工具验证包括 GQA 与显式 KV-head 展开的计算量一致、冻结视觉层只计前向、
checkpoint 重计算会增加计数而不改变梯度。全仓库 CPU 测试759通过、5跳过；
双卡真实 DataLoader + tiny model 额外验证了每卡计时汇总、日志、验证和 checkpoint 保存。
tiny 检查仅证明日志集成，不用于上表的性能结论。
