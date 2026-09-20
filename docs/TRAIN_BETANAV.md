# BetaNav：Token CE + Flow Matching 分布式训练

入口：[train_betanav.py](../train_betanav.py)。训练实现位于
[src/lightnav/training/](../src/lightnav/training/)，读取器位于
[src/lightnav/dataloader/](../src/lightnav/dataloader/)。
训练环境为 `.venv-train/`，划分与动作尺度文件为 `configs/betanav/data/`，检查报告为
`reports/betanav/`。已有 LeRobot 数据可直接训练，不依赖仓库的数据生产目录。

使用 Accelerate 管理设备、BF16 autocast、DDP 和可选日志平台；PyTorch 原生
`ZeroRedundancyOptimizer` 为两个 AdamW 优化器分别分片状态。训练循环保留 BetaModel
特有的 CE/FM mask、梯度隔离和停止语义，不使用 Hugging Face Trainer。
当前支持单进程、DDP 和 ZeRO-1；不能用 DeepSpeed/FSDP launcher 配置启动本入口。
ZeRO-1 只分片优化器状态，模型参数和梯度仍在每张卡上保留完整副本。
GPU 实时监控、每卡耗时/数据等待、TFLOPS 估算和 CUDA 时间线见
[TRAINING_PERFORMANCE.md](TRAINING_PERFORMANCE.md)。

## 冻结原始 VLM、仅训练 FM

配置为 [r2r_fm_only.yaml](../configs/betanav/r2r_fm_only.yaml)，从
`/mnt/cfs/Models/Light` 初始化冻结的 VLM，重新初始化动作专家，不加载上一轮 BetaNav。
`training_mode: fm_only` 同时控制模型和读取器：只使用 prompt 的 hidden states 与原始
连续轨迹监督，不读取 RVQ 码本或 token GT。只有 FM AdamW/ZeRO 状态，当前配置的
`lambda_fm: 1.0`，loss 为 `1.0 × FM`。已完成的 2026-09-16 实验使用 `2.0 × FM`，
其日志与 checkpoint 配置快照记录的是当时的训练设置。
冻结主干以 BF16 执行无梯度前向；FM 投影与专家仍保留 FP32 参数。无需主干梯度检查点。

`num_epochs: 3` 按实际分布式 DataLoader 长度计算 `max_steps`，包括每轮最后不足一个
batch 的数据。每卡 batch 为 B、8 卡、无梯度累积时，每轮更新数为
`ceil(ceil(631244 / 8) / B)`，总更新数为其三倍。DistributedSampler 每轮补齐 4 个重复
样本；所有 631,244 个训练帧均被遍历。这个新实验不沿用上一轮的全局 29,589 步停止点。
更大的 batch 不自动缩放学习率；FM LR=1e-4、warmup=1,000、cosine floor=0.1。

本机最终选用每卡 64、累积 1、全局 512，每轮 1,233 步，三轮 **3,699 步**；
每轮尾部 batch 为全局 464 个样本。CPU 读取 workers 增至每卡 4、prefetch=1。
八卡真实 82 帧历史测试峰值 allocated=66.97 GiB/card、reserved=88.31 GiB/card，
测量的完整更新平均 13.01 秒（39.34 samples/s）；这是缓存长历史的容量测试，
实际随机训练吞吐以 `metrics.jsonl` 为准。详细报告见
[fm_only_20260916](../reports/betanav/fm_only_20260916/README.md)。

不做验证；每 1,000 步异步保存完整模型、FM Adam/RNG/数据游标，保留最近 3 份；每 3,000 步
永久保存模型归档；结束时也保存完整 checkpoint。实验写入独立目录，原实验保持原状。
`training_mode` 和 `num_epochs` 参与精确恢复契约，不能将 CE+FM optimizer 状态当作 FM-only
恢复；旧契约缺省按 `ce_fm`、`num_epochs=null` 解释。

下文的双优化器、RVQ、STOP 和主干学习率说明适用于默认 CE+FM 配置。

### 未来 10 步的三组 FM-only 实验（2026-09-18）

`fm_horizon: 10` 同时配置读取器和动作专家。原始数据保留 30 步，读取器校验后取前
10 个连续 SE(2) 增量及 mask；没有抽样或插值，也不减少样本帧数。每条训练 GT、
专家流场和推理动作均为 `10×3`，组合后的局部航点也是 `10×3`。
`num_flow_steps: 10` 仍表示积分次数；两个参数互相独立。

按以下顺序分别使用八卡训练，每组从冻结的原始 LightNav 和全新 FM 专家开始：

| 配置 | 时间分布 |
|---|---|
| [r2r_fm_only_h10_beta1_1p5.yaml](../configs/betanav/r2r_fm_only_h10_beta1_1p5.yaml) | Beta(1,1.5) |
| [r2r_fm_only_h10_beta12.yaml](../configs/betanav/r2r_fm_only_h10_beta12.yaml) | Beta(1,2) |
| [r2r_fm_only_h10_beta13.yaml](../configs/betanav/r2r_fm_only_h10_beta13.yaml) | Beta(1,3) |

三组均使用 seed=42、`1.0 × FM`、batch=64/card、全局 512、三轮 3,699 步，
沿用相同 LR、warmup、动作尺度、数据清单及 checkpoint 策略。预计每版约十小时，
实际按 epoch 完成，不设十小时强制停止。后续任务等待前一个退出和 GPU 空闲。
不覆盖以前的 H=30 实验；训练契约缺省 horizon 按 30 解释，跨 horizon 禁止精确续训。

### Beta(1,2) 时间采样实验

[r2r_fm_only_beta12.yaml](../configs/betanav/r2r_fm_only_beta12.yaml) 为独立实验：
冻结原始 LightNav、重新初始化专家、`lambda_fm: 1.0`，使用 GPU 0–7、每卡 batch=64，
全局 batch=512，三轮 3,699 步。输出目录为
`/mnt/cfs/Models/beta_nav/r2r_fm_only_beta12_frozen_lightnav_20260917`。

```yaml
flow_time_distribution: beta
flow_time_alpha: 1.0
flow_time_beta: 2.0
```

`tau=0` 对应噪声，`tau=1` 对应干净 GT。Beta(1,2) 的平均 tau 为 1/3，
噪声系数至少为 0.7 的概率为 51%。GT/mask 仍来自原始连续轨迹，训练目标为 FM MSE。
三个采样配置字段参与精确恢复契约，旧 checkpoint 缺省解释为均匀采样；切换分布应新建
实验，不能混用优化器恢复状态。若启用现有验证器，它显式使用固定 RNG 的均匀时间采样，
便于比较不同训练采样方案；本 R2R 配置仍关闭验证。

### Beta(1,3) 与延迟启动

[r2r_fm_only_beta13.yaml](../configs/betanav/r2r_fm_only_beta13.yaml) 沿用 Beta(1,2)
的全部训练设置，仅将 `flow_time_beta` 改为 `3.0` 并使用独立输出目录
`/mnt/cfs/Models/beta_nav/r2r_fm_only_beta13_frozen_lightnav_20260918`。
仍从原始 LightNav 和新专家开始；平均 tau 为 1/4，噪声系数至少为 0.7 的概率为 65.7%。

服务器一次性队列使用 `python -m lightnav.training.scheduled_launch --schedule PATH`。
`--check` 仅验证冻结文件和当前可启动条件，不创建训练进程。
准备好包含 `supervisor.py`、`train_config.yaml` 和状态为 `prepared` 的 `launch.json`
的启动目录后，传入以下 JSON（路径必须为实际绝对路径）：

```json
{
  "repository": "/absolute/repository",
  "launch_dir": "/absolute/prepared_launch",
  "not_before_utc": "2026-09-17T17:08:44+00:00",
  "wait_for_launch_dir": "/absolute/previous_launch",
  "gpu_indices": [0, 1, 2, 3, 4, 5, 6, 7],
  "poll_seconds": 30,
  "prepared_sha256": {
    "/absolute/prepared_launch/train_config.yaml": "SHA256",
    "/absolute/prepared_launch/supervisor.py": "SHA256"
  }
}
```

队列最早在指定时间触发；此前一轮结束后，再等待无训练/分析进程、无 GPU compute
进程且所需八卡显存各不超过 1 GiB。检查失败继续等待，不中断已有任务。
`wait_for_launch_dir` 可省略；`poll_seconds` 默认为 30，支持 1–60 秒。
文件锁与 `schedule_state.json` 防止重复启动，启动前再次核对冻结文件哈希；
状态文件记录排队 PID、等待原因、启动后的 supervisor PID 和最终结果。
此队列是服务器后台进程，须保持服务器/容器在线，不自动恢复重启后的任务。

## 参考与取舍（CE+FM）

- [LeRobot 多 GPU 训练](https://huggingface.co/docs/lerobot/multi_gpu_training)：
  参考其 Accelerate 接入、每卡 batch、主进程日志和显式学习率设置。
- [OpenPI PyTorch 训练器](https://github.com/Physical-Intelligence/openpi/blob/main/scripts/train_pytorch.py)：
  参考其 DDP、AdamW、学习率 warmup/衰减和恢复训练的组织方式。
  [OpenPI optimizer 配置](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/optimizer.py)
  默认 cosine schedule 的 peak LR 为 2.5e-5、warmup 为 1,000、decay 为 30,000，
  AdamW betas 为 0.9/0.95、clip 为 1；各实验配置可以覆盖这些值。
- [OpenVLA-OFT 微调](https://github.com/moojink/openvla-oft/blob/main/vla-scripts/finetune.py)：
  参考动作 chunk、梯度累积和可选验证。其 LoRA/动作头微调配置里的学习率不能直接
  用于这里全量更新的语言主干。
- [PyTorch ZeRO 优化器](https://docs.pytorch.org/docs/2.10/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer)：
  与 DDP 配合分摊 Adam 状态，仍允许 CE/FM 使用独立优化器和独立裁剪。

这些是训练基础设施和优化习惯的参考。下面的数值是针对当前 LightNav/R2R 提出的
首轮实验配置，不是已有论文验证过的 BetaNav 最优超参数，也不是对 π0.5 训练配方的完整复现。

## 首轮超参数

完整配置：[configs/betanav/r2r.yaml](../configs/betanav/r2r.yaml)。

| 参数 | 首轮值 | 说明 |
|---|---:|---|
| 每卡 batch | 8 | 完整权重在双卡约96GB GPU上通过长/短历史测量 |
| 梯度累积 | 1 | 8 GPU 时有效 batch=64；不是每卡 64 |
| 停止点 | 全局29,589步 | 用户确认按9,863×3的数值停止；全量数据实际每轮9,864步 |
| warmup | 1,000 | 按 optimizer 更新窗口计数，不按 microbatch/帧计数 |
| 主干/LM head LR | 1e-5 | 已预训练参数，先较小幅度更新 |
| FM 专家/投影 LR | 1e-4 | 新初始化约 36.5M 参数，独立优化器 |
| schedule | warmup + cosine | 衰减至各自 peak LR 的 10% |
| AdamW betas / eps | [0.9,0.95] / 1e-8 | 两个分支分别维护 moments |
| weight decay | 0.01 | bias 和一维参数不衰减 |
| 梯度裁剪 | VLM=1、FM=1 | 分支独立，先完成全局有效数量归一化再裁剪 |
| λFM | 2.0 | CE + 2.0×FM；主干不会收到 FM 梯度 |
| 视觉编码器 | 冻结 | 语言主干、LM head、FM 专家和投影训练 |
| precision | BF16 autocast | 参数与 Adam 状态保持 FP32；支持 `no`，暂不支持 FP16 scaler |
| gradient checkpointing | 开启、non-reentrant | 降低主干 activation 显存 |
| 专家结构 | 宽512、6层、8头、FFN2048 | dropout=0 |
| FM horizon / RVQ horizon | 30 / 10 | 与当前数据契约固定对应 |
| FM 采样积分次数 | 10 | 验证/推理参数；训练每次采样一个连续 flow time |
| CPU workers / prefetch | 2 / 1，每卡 | worker 使用 spawn，只加载 CPU 数据/processor |
| 日志 / 验证 | 每10步 / 关闭 | eval_every=0、save_best=false |
| checkpoint 模式 | dual | CPU一致快照后，各rank后台写模型/Adam分片/RNG |
| 完整checkpoint | 每1,000步，最近3份 | checkpoints/；结束或优雅停止时也保存完整状态并参与轮换 |
| 永久模型归档 | 每3,000步 | model_checkpoints/；仅模型，不受keep_last影响 |

先调整 `lr_vlm`（3e-6～2e-5）、`lr_fm`（5e-5～3e-4）和有效 batch（32/64），
再比较是否解冻视觉编码器。λFM 可比较 0.5/1/2，但分支本身不共享梯度，
FM 的学习率和裁剪往往更直接影响其更新。改变卡数不会自动放大学习率。
不要同时改动多个因素，否则难以定位 CE 遗忘或 FM 不收敛的原因。

超参数由配置设置和实验选择；被梯度学习的是权重：语言主干/LM head、动作专家、
context projection。RVQ 码本、动作归一化尺度、SlowFast 采样规则和 mask 都固定。

## Batch 与显存实测（2026-09-13）

当前机器有8张名义96GB GPU；测试使用空闲的物理 GPU 1、2，完整 LightNav 预训练权重、
实际 R2R RGB/指令/GT、BF16、冻结 ViT、gradient checkpointing、DDP + ZeRO-1。
包含前向、反向、梯度归一化/裁剪和 AdamW 更新；不是 tiny-model 或推理显存。
每卡每个更新固定处理8帧样本，因此双卡测量有效 batch=16，正式8卡配置为64。

长历史样本为训练 episode 4112 的 t=152：82张历史图像，输入序列1411 token，
FM 有效未来30步。每种组合预热1次、测量2次完整更新，峰值包含预热/优化器状态初始化：

| 每卡 batch | 累积次数 | 每卡峰值 allocated GiB（两卡最大） | 双卡样本/秒 | 每次更新秒数 |
|---:|---:|---:|---:|---:|
| 1 | 8 | 70.89 | 3.36 | 4.763 |
| 2 | 4 | 71.57 | 4.07 | 3.930 |
| 4 | 2 | 72.86 | 4.42 | 3.622 |
| 8 | 1 | 60.40 | 4.48 | 3.574 |

同一 episode 的 t=20 有22张配对后历史图像、588 token；测量3次更新：
`1×8` 为5.01样本/秒，`4×2` 为8.50，`8×1` 为9.67。长历史提升约33%，短历史约93%。
从全新进程直接使用 `8×1` 再测长历史，峰值 allocated=60.31 GiB、reserved=64.39 GiB，
吞吐4.49样本/秒，排除了必须先用小 batch 初始化才能运行的情况。
原始记录见 [长历史](../reports/betanav/batch_profile_long_2gpu_20260913.json)、
[t=20](../reports/betanav/batch_profile_t20_2gpu_20260913.json)、
[batch=8 新进程](../reports/betanav/batch_profile_bs8_fresh_2gpu_20260913.json)。

这些结果使用缓存好的 CPU batch，重复同一真实样本，衡量模型容量和计算吞吐；
不包含随机 Parquet 读取、worker 等待、验证和 checkpoint I/O，不能当成正式8卡端到端吞吐。
长历史样本也不是对全数据每条指令长度的穷举上界。正式训练应继续观察不同长度 batch 的
峰值和 `samples_per_second`，为输入差异及运行时缓冲留余量。

梯度累积增大的是每次更新的样本总量，不会同时增大单次前向的 batch，见
[Accelerate 文档](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation)。
显存由参数、Adam 状态、activation 和 DDP 梯度缓冲共同决定，
[DDP 的 bucket view](https://docs.pytorch.org/docs/2.10/generated/torch.nn.parallel.DistributedDataParallel.html)
也影响梯度内存占用；表中 batch=8 同时关闭累积，因此不能解读成“batch 越大显存越少”。
`allocated` 是活跃张量峰值，`reserved` 包含 PyTorch allocator 缓存；它们也不等于
`nvidia-smi` 显示的整个进程占用。目标是稳定提高吞吐，不是填满显存。

可复现的短测命令（不会写入训练 output_dir 或保存模型）：

```bash
CUDA_VISIBLE_DEVICES=1,2 NCCL_DEBUG=WARN \
.venv-train/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 \
  -m lightnav.training.profile_batch \
  --config configs/betanav/r2r.yaml \
  --batches 1 2 4 8 --per-rank-batch 8 --warmup 1 --steps 2 \
  --report reports/betanav/NEW_batch_profile.json
```

| 测量参数 | 说明 |
|---|---|
| `--config` | 必填，使用完整训练模型、精度和优化器配置，拒绝 CPU/tiny model |
| `--report` | 必填，新 JSON 文件路径，拒绝覆盖；每个候选完成后更新报告 |
| `--batches` | 默认1/2/4/8，每卡 microbatch 候选 |
| `--per-rank-batch` | 默认8，候选必须整除它；累积次数=本参数/batch |
| `--warmup` / `--steps` | 默认1/2，每个候选的预热/计时更新数，均至少1 |
| `--history-frame` | 可选 t，例如20；默认选训练集中最长 episode 留足 FM-30 的最后一个帧 |

测量只在内存中做临时权重更新，使用配置的固定 LR，不用于判断 loss/导航质量。
同一进程按候选顺序连续测试；OOM 会终止该 torchrun 任务，已完成候选留在报告中。
退出时释放测试 GPU，无须清理训练 checkpoint。

## 数据划分与归一化

更换数据或划分时运行一次 [prepare_data.py](../src/lightnav/training/prepare_data.py)：

```bash
.venv-train/bin/python -m lightnav.training.prepare_data \
  /mnt/cfs/data/beta_nav/lerobot_r2r_train_470x280_20260912 \
  --output configs/betanav/data/NEW_training_data.json \
  --seed 42 --val-fraction 0 --scale-floor 0.05 0.02 0.05
```

当前配置使用 [r2r_all_train_20260913.json](../configs/betanav/data/r2r_all_train_20260913.json)：

- 全部61个场景、10,819条episode、631,244个时刻样本用于训练，没有验证划分。
- 一般每步8卡×每卡8样本×累积1次=64个样本；每个样本包含截至当前时刻的SlowFast图像历史、指令、token标签和未来最多30步FM GT，不能把64个样本理解为64张RGB图。
- DistributedSampler每卡78,906个样本，共631,248次样本访问（补齐4个重复样本）。
- 每轮9,864步：9,863步×64样本，末步每卡2样本、全局16样本。
- 本次在全局3,047步恢复后从全量数据游标0开始；用户确认在全局29,589步停止，因此该数值不代表切换后的全量数据完整遍历三轮。
- 从旧状态扩展全量数据时保留原来的action_scale；`normalization_source`记录统计来源，避免改变已训练动作专家的尺度。

历史 [r2r_train_20260913.json](../configs/betanav/data/r2r_train_20260913.json) 保留用于恢复来源核对：

- 训练：55 个场景、10,042 条 episode、584,564 帧。
- 验证：6 个场景、777 条 episode、46,680 帧。
- 训练有效 FM 动作：12,868,911 步；不包含验证部分或 masked padding。
- 固定 `action_scale=[0.1145287109, 0.02, 0.1431552994]`，单位为米/米/弧度。

上述历史划分按场景进行，因此同一路线不同指令也不会跨集合。历史验证集来自当前 train 导出中的
场景留出，不是官方 R2R val_unseen；本训练器不会启动 Habitat 评测。
尺度使用训练 episode 的 mask-aware mean/std/count 合并计算，但实际变换仅执行
`action / scale`，不减均值。侧向动作 std 约 0.0062 m，使用 0.02 m 下限以免过度放大。
读取器和模型都不裁剪真实动作值。

`--val-fraction 0`生成全训练manifest；正数仍支持按场景留出验证。
继续使用已有模型时，可加`--normalization-from configs/betanav/data/r2r_train_20260913.json`
保留此前的固定动作尺度；它要求元数据hash完全一致。新训练不传该参数时对全部训练episode拟合尺度。

准备命令仅读取元数据/统计，不重建图像数据，拒绝覆盖已有 manifest。
训练启动检查 manifest 记录的元数据 hash，并核对 checkpoint 的 RVQ 资产。
当前配置已经引用冻结 manifest，无需重复准备，GT 不需要重新生成。
训练运行只依赖已有数据集、原始 checkpoint 和训练代码/config，独立于数据生产目录。
数据集内部 `meta/provenance` 是来源记录，应随数据保留，不属于训练代码依赖。

## 安装、启动与恢复

在仓库根目录运行：

```bash
bash scripts/setup_betanav_train.sh
```

它创建独立的 `.venv-train/`，复用仓库 `.venv/` 的 Torch/Transformers，
在新环境安装 PyArrow 25.0.1 与 Accelerate 1.12.0，不改动原环境。
通用安装也可以在独立训练环境执行 `pip install -e '.[training]'`。
可选日志依赖通过 `.[training-logs]` 安装，然后将 `tracker` 设置为 `tensorboard`
或 `wandb`。默认只写本地 JSONL，不要求登录；WandB 默认 offline，可显式配置 online。

先检查数据和配置，完全不加载模型权重：

```bash
.venv-train/bin/python train_betanav.py \
  --config configs/betanav/r2r.yaml --cpu --dry-run
```

单进程 dry run 的 world_size=1，因此显示有效 batch=8；正式8卡启动时有效 batch=64。

8 GPU 新实验启动（output_dir须为空）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv-train/bin/python -m accelerate.commands.launch \
  --config_file configs/betanav/accelerate_8gpu.yaml \
  train_betanav.py --config configs/betanav/r2r.yaml
```

本次从历史完整checkpoint迁移到全量训练的命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv-train/bin/python -m accelerate.commands.launch \
  --config_file configs/betanav/accelerate_8gpu.yaml \
  train_betanav.py --config configs/betanav/r2r.yaml \
  --resume /mnt/cfs/Models/beta_nav/r2r_ce_fm_01_start/checkpoints/step-00003047 \
  --expand-training-data --override-max-steps
```

`--expand-training-data`只允许原train+val的完整并集，严格核对数据hash、归一化、模型、batch、
优化器和schedule。保留全局step/Adam/RNG，把epoch和batch_offset重置为0，明确重新遍历全量数据。
这次从全局3047开始，全量第一轮完成时全局step为12,911。
`--override-max-steps`显式允许把旧checkpoint的30000改为29589，后续cosine衰减终点随之改变；
该选项自身不重置数据游标，也不允许修改batch、学习率或其他schedule参数。
后续从当前配置保存的完整状态恢复，使用`--resume auto`即可，不再传上述两个迁移选项。
永久模型导出不含Adam，不能精确续训；`--resume`只接受完整checkpoint。
当最新模型导出比完整checkpoint更新时，`--resume auto`会拒绝静默回退到较早训练状态。
`--stop-after`用于保存并退出，不要通过修改max_steps模拟中断，以免改变cosine schedule。

同样支持 `torchrun`，进程内部仍然使用 Accelerate：

```bash
.venv-train/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 \
  train_betanav.py --config configs/betanav/r2r.yaml
```

多机使用 Accelerate 的 `num_machines`、`machine_rank`、`main_process_ip/port`
配置，各节点必须访问同一数据、模型和输出目录。当前验证覆盖单机；多机网络吞吐未测。
单 GPU 直接运行 Python 即可，`zero1` 自动使用普通 AdamW，有效 batch 变为8。
正式训练不会因 CUDA 不可用而自动退回 CPU。

## 数据和梯度的基础设施细节

DataLoader 已根据 rank/world_size 分配数据，不再传给 `accelerator.prepare()` 二次分片。
训练 DistributedSampler 会补齐至各 rank 等长，至多重复 world_size-1 个帧样本；
每个 epoch 调用 set_epoch。验证使用没有补齐的独立分片，统计时不重复计算样本。

每个 microbatch 先计算 CE/FM 的本地有效均值；反向时分别乘本地有效 token/动作步数，
累积梯度和总数。窗口结束后 all-reduce 计数：由于 DDP 梯度是按卡平均，
主干梯度乘 `world_size / 全局有效 CE token 数`，专家梯度乘
`world_size / 全局有效 FM 步数`。FM loss 内部的三维平均已经包含在模型中。
此过程利用了两个分支的梯度隔离，也适用于最后不足完整累积长度的窗口。
开启梯度累积时，不需要在 RAM 中提前保留整个更新窗口的视频 batch 来计算分母。

非最后 microbatch 使用 no_sync 减少参数梯度通信。DDP 开启 find_unused_parameters，
覆盖某个 rank 或整个 batch 没有有效 FM 动作的情况。
某分支全局没有标签时，梯度设为 None，并跳过该 optimizer.step，因此不会产生
Adam moments 更新或 weight decay。两个分支都检查/裁剪完成后才更新权重。

## 日志、验证与 checkpoint

当前`checkpoint_mode: dual`同时执行两套保留规则：

| 类型 | 频率 | 内容 | 保留规则 |
|---|---|---|---|
| 完整训练状态 | 每1000个全局step | 模型、两套Adam分片、每rank RNG、计数器和数据游标 | 同一run_id仅保留最近3份 |
| 永久模型归档 | 每3000个全局step | 同一步的完整BetaModel权重及配置 | 永久保留，不参与轮换 |

新完整状态格式为`betanav_checkpoint_v2`：

```text
checkpoints/step-00004000/
  model.pt          # 完整BetaModel FP32权重
  rank-00000.pt ... # 每个rank的Adam分片与Python/NumPy/Torch/CUDA RNG；八卡共8份
  timing-00000.json ... # 每个rank的CPU快照和后台写盘耗时
  trainer.json      # 全局step、epoch、下一batch位置、样本数与各分支更新计数
  metadata.json     # 配置、数据/恢复契约、参数划分格式、world_size、run_id
  COMPLETE
latest.json
```

例如第6000步完整保存成功后，滚动目录最终只留4000/5000/6000，模型归档仍保留3000/6000。
正常结束或优雅停止时还会保存当前步完整状态，并参与最近3份轮换；非3000倍数不会额外创建永久模型归档。
永久归档结构如下：

```text
model_checkpoints/step-00003000/
  model.pt       # 完整BetaModel FP32权重：VLM/LM head、FM expert、context projection及模型buffer
  metadata.json  # 模型/训练配置、数据契约、step、run_id；明确不含完整训练状态
  COMPLETE
latest_model.json
```

保存先在同一个更新边界把rank 0模型权重、各rank本地Adam分片和RNG复制成独立CPU快照。
不调用ZeRO的`consolidate_state_dict`，不会把所有Adam状态汇总到rank 0。
随后训练继续，每个rank的后台线程只写CPU副本，不访问CUDA、活动优化器或执行分布式通信。
每rank只允许一个未完成的后台快照，避免无限堆积内存；如果下次保存到来时前一次未完成，会等待。
所有rank写入成功后才提交目录与latest指针，再清理旧完整状态；后台错误在下一更新边界通知所有rank。
正常退出会等待写入完成。永久归档与同一步完整状态使用同一份权重；支持硬链接的文件系统不重复写模型字节，
否则后台复制。删除旧完整状态不会删除永久归档。
这里的异步指磁盘I/O与后续训练重叠，并非权重快照完全无停顿。
`full_checkpoint_queued`、`full_checkpoint_committed`日志记录快照、写盘耗时和被轮换的目录。
`keep_last: 3`约束完整状态；永久模型文件永远不参与轮换。
`model_save_every`必须为`save_every`的整数倍，dual模式要求`save_best: false`。
仍可单独配置`model_only`进行永久模型导出，或`full`使用旧版同步完整保存。

下列完整恢复格式保留兼容，用于历史checkpoint或显式配置`checkpoint_mode: full`的实验：

```text
config.json / data_manifest.json / run.json  # 初次启动快照、数据来源、依赖版本、代码 hash
active_config.json                         # 最近一次启动/恢复实际使用的配置
active_data_manifest.json                  # 最近一次启动/恢复实际使用的数据范围
resumes/*.json                             # 每次恢复的来源、配置和训练代码 hash
metrics.jsonl                              # CE/FM、LR、梯度范数、有效数量、吞吐、显存
latest.json / best.json                    # best仅历史或启用验证的full模式使用
checkpoints/step-00001000/
  model.pt                                # 完整 BetaModel state_dict，FP32
  optimizer-vlm.pt / optimizer-fm.pt        # 两套完整 Adam 状态
  trainer.json                            # step、epoch、下一 batch 位置、最佳指标
  rng-00000.pt ...                         # 每个 rank 的 Python/NumPy/Torch/CUDA RNG
  metadata.json                           # BetaConfig、world size、恢复契约
  COMPLETE
```

旧版v1保存先让所有 rank 写 RNG、汇总 ZeRO 状态，然后主进程写临时目录，最后原子 rename
并更新 latest。没有 COMPLETE 的目录不会被恢复。完整 Adam 状态保存在 CPU/checkpoint 中，
恢复v1时每个 ZeRO rank 读取后保留自己的分片；恢复v2时直接读取本rank分片并核对参数名称和顺序。
两种完整格式均可恢复Adam、数据游标和RNG。模型/Adam文件较大，CPU快照及磁盘仍须留足空间。

恢复要求相同 world_size、batch/累积设置、模型/数据和 schedule；只允许改变日志、
worker、prefetch、保留数量及验证开关/频率等不影响训练更新的选项。
改变训练终点必须显式传`--override-max-steps`，且新终点必须大于恢复步数；其余契约仍严格检查。
旧版 checkpoint 契约里的验证选项允许在恢复时变更，不修改历史 checkpoint；
数据、batch、学习率和训练 schedule 等约束仍然严格检查。通过跳过 batch sampler 的索引
恢复游标，不重新解码整个 epoch 已训练过的 RGB。当前读取器没有随机图像增强；
将来添加随机增强时必须一起扩展 worker RNG 恢复。

SIGINT/SIGTERM 请求在完成当前更新窗口后协调保存；硬崩溃保留最近已提交的 checkpoint。
训练进程不会发布模型或启动机器人。这里保存完整 Beta 状态与配置，推理加载可复用
原始 LightNav checkpoint 构建模型后加载 model.pt；不是直接供 lightnav-serve 使用的 HF 发布目录。

当前 R2R 训练关闭验证：`eval_every: 0` 时不创建验证 DataLoader、不运行验证，
`save_best: false`关闭验证改善触发的额外保存，当前全量manifest已经把全部episode纳入训练。
运行中的进程不热读YAML；变更后须重启，并按完整状态或模型初始化语义选择来源。

启用验证时记录 CE 和 FM loss。默认每卡最多64个 batch，用固定噪声/flow time，另外对前4个
batch 执行 FM 积分并计算有效未来轨迹的 ADE/FDE（米）。这些是离线轨迹误差，
不是闭环导航成功率；要计算全验证集可以设置 eval_max_batches=0、相应增加采样 batch 数。

## CLI 参数

| 参数 | 说明 |
|---|---|
| `--config` | 必填，严格 YAML；未知键报错，全部字段见 training/config.py |
| `--output-dir` | 覆盖输出目录；非空目录必须显式 resume |
| `--data-manifest` | 覆盖已准备的划分/归一化 manifest 路径 |
| `--resume [auto或路径]` | 恢复完整训练状态；路径须属于该 output_dir 的 checkpoints |
| `--expand-training-data` | 显式从旧train+val扩展为全量训练；须配合resume，保留优化器/step并重置数据游标 |
| `--override-max-steps` | 须配合resume；仅放行max_steps变更并调整后续cosine终点，保留Adam/RNG/数据游标并记录审计 |
| `--dry-run` | 校验配置和数据，显示有效 batch/更新数量，不加载权重或创建 run |
| `--cpu` | 显式 CPU 执行；不适合正式大模型训练 |
| `--tiny-model` | 小型随机 Qwen3-VL，保留真实 tokenizer/视频处理；只验证接口 |
| `--stop-after` | 在指定全局 step 保存并停止，保留原 schedule |

## 验证方式

[smoke.yaml](../configs/betanav/smoke.yaml) 是 CPU 小型随机模型配置；
[smoke_gpu.yaml](../configs/betanav/smoke_gpu.yaml) 是双 GPU BF16/ZeRO 配置，均只运行4步。
这两份配置使用真实数据/processor，但没有加载预训练 VLM 权重；不能据此评价导航质量
或推断完整模型吞吐/峰值显存。

CPU 回归包括全局 mask 加权、空 FM 分支、checkpoint/RNG 恢复和 best 保留；
两进程 Gloo 测试覆盖不同 rank 标签数不等、最后 microbatch 全无 FM、ZeRO 保存/恢复。
双卡异步完整保存测试覆盖后台写入期间继续更新、恢复后下一步权重一致、Adam/RNG恢复、
完整状态最近3份轮换、永久模型保留以及单rank写盘失败通知所有rank。
本次更新后CPU测试768通过、5跳过、2个GPU测试排除；双GPU异步完整状态测试通过。
真实完整模型的八卡核验在第3050步保存v2状态，CPU快照约5.6秒、后台写盘约50.2秒，
随后从该状态恢复并继续训练，数据游标接续正确。该次因核验退出而等待写盘；
正常周期保存时磁盘I/O与后续更新重叠。记录见
[dual_checkpoint_transition_20260913.json](../reports/betanav/dual_checkpoint_transition_20260913.json)。
完整测试入口：`make test PYTHON=.venv-train/bin/python`。

2026-09-13 的检查结果：全仓库 CPU 测试753通过、5跳过；真实数据的小型模型 CPU
中断/恢复与连续训练得到逐位相同的最终权重；双 GPU BF16 + ZeRO-1 + spawn workers
完成两步训练/保存，并通过 Accelerate launcher 恢复到第4步。
检查摘要见 [betanav_trainer_smoke_20260913.json](../reports/betanav/betanav_trainer_smoke_20260913.json)。
该历史摘要保持原始内容，其中路径记录当时的环境位置。
随后训练目录解耦与 batch 测量工具检查：CPU 测试754通过、5跳过，
新环境 dry run 通过，训练模块/配置不再导入或引用数据生产目录。
