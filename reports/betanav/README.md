# BetaNav 训练检查记录

训练代码位于 `src/lightnav/training/`，读取器位于 `src/lightnav/dataloader/`，
环境为 `.venv-train/`，配置/数据划分位于 `configs/betanav/`。
已有 LeRobot 数据可以直接训练，不依赖数据生产代码。使用说明见
[TRAIN_BETANAV.md](../../docs/TRAIN_BETANAV.md)。大型 checkpoint 写入训练 `output_dir`。

2026-09-13 的完整预训练模型容量与吞吐测量：

- [长历史，batch 1/2/4/8](batch_profile_long_2gpu_20260913.json)
- [t=20，batch 1/4/8](batch_profile_t20_2gpu_20260913.json)
- [全新进程直接使用 batch=8](batch_profile_bs8_fresh_2gpu_20260913.json)
- [本机 BF16 GEMM 参考](gemm_baseline_20260913.json)
- [完整模型 TFLOPS 估算与 CUDA trace 索引](training_performance_native_2gpu_20260913.json)
- [NCCL kernel 与 GPU 采样摘要](performance_observation_20260913.json)

均为物理 GPU 1、2 上的双卡 BF16/DDP/ZeRO-1，包含反向和 AdamW 状态。
为控制输入长度，重复同一条真实训练样本；计算吞吐不包含随机数据读取/验证/保存。
模型权重更新只发生在测量进程内，退出即丢弃，没有保存训练 checkpoint。
性能观测用法与指标口径见 [TRAINING_PERFORMANCE.md](../../docs/TRAINING_PERFORMANCE.md)。
早期 `training_performance_2gpu_20260913.json` 和
`training_performance_gqa_2gpu_20260913.json` 记录分析工具的失败试验，状态为 failed；
以 `training_performance_native_2gpu_20260913.json` 中的完整结果为准。

以下早期接口检查摘要为逐字节副本，保留当时路径/参数和测试结果；不是完整模型性能依据：

- [读取器与模型接口](train_lerobot_loader_smoke_20260913.json)
- [读取器双 worker](train_lerobot_loader_workers_20260913.json)
- [训练器 CPU、双 GPU tiny model 与恢复检查](betanav_trainer_smoke_20260913.json)
