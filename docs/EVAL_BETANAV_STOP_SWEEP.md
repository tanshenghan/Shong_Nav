# 固定 checkpoint 的 STOP 八组开发集对比

一条平台任务启动四个阈值与两种确认次数的完整交叉实验：

| GPU（任务内编号） | STOP 阈值 | 连续确认次数 |
| --- | --- | --- |
| 0 | 0.5 | 1 |
| 1 | 0.5 | 2 |
| 2 | 0.6 | 1 |
| 3 | 0.6 | 2 |
| 4 | 0.7 | 1 |
| 5 | 0.7 | 2 |
| 6 | 0.8 | 1 |
| 7 | 0.8 | 2 |

这是 **8 组完整闭环评估**，每组对同一开发集的所有 episode 运行一次；不是对同一张图片调用八次模型。
所有组固定原 LightNav backbone、FM + StopHead checkpoint、FM 积分步数、噪声种子、环境参数和 episode 集合。

## 平台启动命令

申请单机八卡后提交：

```bash
exec bash /mnt/pfs/tanshenghan/vln4project/LightNav-0/scripts/eval_betanav_stop_sweep_8gpu.sh
```

默认 checkpoint 为：

```text
/mnt/cfs/Models/beta_nav/r2r_fm_stop_h30_prefix_kv_all36_beta1_1p5/checkpoints/step-00003699
```

每组独占一张已分配的 GPU，每卡默认六个常驻 worker，共 48 个 worker；各组端口和输出目录独立。
遵守平台的 `CUDA_VISIBLE_DEVICES`（包括 GPU UUID），子进程内部使用逻辑 GPU 0。
所有组 `fm_seed=42`、`fm_steps=10`、`max_steps=500`、成功半径 3 米，关闭 oracle STOP 和视频。
每个 worker 加载一次模型，动态领取本组分片。默认全局最多两个模型同时加载，已经加载的组继续推理。
六 worker 的实际显存峰值仍需要在分配节点确认；本任务沿用用户指定的并发，不自动降级。

## 开发集独立性

本次固定使用官方 **R2R `val_seen`** 作为开发集。它允许场景与训练集重合，但要求 episode 内容和完整参考路线
不出现在该 FM + StopHead checkpoint 的训练数据中。这不代表未见场景评估，也不证明基础 VLM 的预训练数据没有重合。

预检读取 `val_seen/val_seen.json.gz`、`val_seen/val_seen_gt.json.gz`、`train/train.json.gz` 和 MP3D 场景，
并将仓库冻结的 `r2r_all_train_stop_v1.json` 内容哈希与 checkpoint 的 `contract.data_sha256` 对齐。
训练 manifest 的源 scene/episode 身份必须完整对应原始 train；跨 split 比较实际起点、姿态、目标、规范化指令和参考路线，
不会把不同 split 中恰好相同的数字 episode ID 当成数据泄漏。

当前开发账户没有 CFS 数据目录读取权限，实际文件存在性、数量和独立性由排队任务账户的 CPU 预检确认。
任一必需文件缺失、权限不足或发现重合都会退出，**不会改用训练集或 `val_unseen`**。
已有 `val_unseen` 结果用于诊断；从开发集选出的配置再单独报告其泛化评估，八组本身的分数均标记为开发集结果。

## 连续确认的精确定义

`confirm=1` 保留原行为：本轮 STOP score 达到阈值立即终止。

`confirm=2` 第一次达到阈值后，仍从本轮共享的 VLM 编码生成并执行 FM 动作；下一轮使用新观测重新计算分数。
若仍达到阈值则 STOP，否则清零计数。例：阈值 0.6 时，`0.65 → 0.55 → 0.70 → 0.72` 在第四轮停止。
新 episode 清零；确认后保持停止。每轮仅一次 VLM 编码。

待确认时若 FM 给出 Habitat 会解释为 STOP 的零速度，使用已有的后续 waypoint/恢复转向，防止绕过确认。
该事件单独记录为 `stop_confirmation_zero_command_suppressed`。正常 FM 零命令和预算强制 STOP 仍单独统计。
因此第二次确认可能改变机器人位置与后续路线；是否提高 SR 必须以实际闭环结果判断。

## 查看计划与预检

```bash
# 只生成配置计划：不读取模型/数据，不需要 GPU。
bash scripts/eval_betanav_stop_sweep_8gpu.sh --plan-only

# 在有数据权限的任务账户下检查 checkpoint、独立开发集和 Habitat 环境。
bash scripts/eval_betanav_stop_sweep_8gpu.sh --check-only
```

推理复用 `.venv-train-stop`，Habitat 继续使用独立 Python 3.9 环境；已有推理环境可用 `BETA_EVAL_PYTHON` 指定。
所有参数见 [CONFIGURATION.md](CONFIGURATION.md)。默认结果位于
`outputs/betanav_stop_sweeps/<UTC时间>_r2r_val_seen/`，顶层 `progress.json` 显示已完成的组，
每组目录中保留 `job.json`、`coverage.json`、`results.jsonl`、`summary.json`、`stop_diagnostics.json` 和 worker 日志。
八组全部完成、episode 覆盖完整且固定配置一致后，生成 `comparison.csv`、`comparison.json`、`comparison.md`。
对比表提供 SR、OS、SPL、nDTW、NE、平均步数、误停分解以及预算 STOP；按开发集 SR、SPL、平均步数依次选出配置，
不将开发集最优分数当作独立测试结果。
失败会保留日志和已有结果，并清理本任务的进程；不会将部分组当作完整八组实验。
