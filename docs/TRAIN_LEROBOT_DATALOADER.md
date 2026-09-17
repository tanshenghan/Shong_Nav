# Train_Lerobot_DataLoader

入口：[Train_Lerobot_DataLoader.py](../src/lightnav/dataloader/Train_Lerobot_DataLoader.py)。
训练读取器位于 `src/lightnav/dataloader/`，环境为 `.venv-train/`，报告位于 `reports/betanav/`。
训练直接消费已有 LeRobot 数据，不需要数据生产代码。
它把本项目 image-backed LeRobot v3.0 数据读取成 `BetaTrainBatch`，可以直接传给
`src/lightnav/inference/beta_model.py` 中的 `BetaModel`。模型训练循环、optimizer
和 checkpoint 调度由 [train_betanav.py](../train_betanav.py) 的训练器负责，
详见 [TRAIN_BETANAV.md](TRAIN_BETANAV.md)；读取器不会启动训练或加载 GPU 权重。

## 使用

新增 `training_mode="fm_only"` 参数（默认 `"ce_fm"`）：只读取原始连续动作 GT、有效
mask、索引及图像历史，不要求 RVQ/apos/opos 列或码本文件。SlowFast 历史与连续 GT 的
坐标、30 步 horizon 和尾部 mask 与原训练保持一致。prompt 不再要求输出离散 token；
collator 在 assistant 回答起始位置截断，移除空回答的 EOS，`labels` 全部为 `-100`。
`answer_start` 指向每行补齐后的序列末端，VLM 只编码输入上下文。此模式未新增停止监督。

```bash
bash scripts/setup_betanav_train.sh
# 之后使用 .venv-train/bin/python 运行训练脚本。
```

训练读取环境复用已有的 Torch / Transformers 5.8，增加 PyArrow 25.0.1 和 Accelerate 1.12.0。
它与 `.venv/lerobot` 导出环境分开，避免后者固定的旧 huggingface-hub 破坏
Transformers 导入；不修改推理或 Habitat 环境。

```python
from lightnav.dataloader.Train_Lerobot_DataLoader import Train_Lerobot_DataLoader

def main():
    loader = Train_Lerobot_DataLoader(
        dataset_root="/mnt/cfs/data/beta_nav/lerobot_r2r_train_470x280_20260912",
        model_path="/mnt/cfs/Models/Light",
        batch_size=2,
        num_workers=2,
        shuffle=True,
        seed=42,
    )
    loader.set_epoch(0)
    batch = next(iter(loader))
    print(batch.model_inputs["input_ids"].shape)
    print(batch.fm_clean_se2_diff.shape)  # [2,30,3]
    print(batch.fm_valid_mask.sum(dim=1))
    print(batch.sample_info[0])
    # out = model(batch)  # 已加载并设为 train() 的 BetaModel
    # out.loss.backward()

if __name__ == "__main__":
    main()  # spawn workers 需要 main guard；交互式环境先使用 num_workers=0。
```

模型与读取器应指向同一 checkpoint 的 tokenizer、eval_config 和 RVQ 码本。
若模型加载时显式覆盖了图像尺寸/池化等参数，需要同时对齐读取器配置，不能只改一侧。
当前读取器默认严格使用 checkpoint 的 vlnce 配置。

## 一条训练样本怎么形成

1. **按 `(episode,t)` 定位。**初始化只读取 `meta/info.json` 和 episode 索引的必要列，
   建立全局帧范围与 Parquet 路径。每次只读取所在 row group 的监督列和 RGB 列，
   不读取深度/位姿，也不构建包含数百 GB 图像的 Hugging Face Arrow 副本。
2. **复用 SlowFast。**调用原有 `slowfast_segments(t,t+1,checkpoint_tiers)`，保持
   Current/Fast/Mid/Long/Anchor、相邻帧配对、起始帧去重及时间顺序与推理一致。
   只解码选中的历史 RGB；每个 segment 的 `total_frames=t+1`，不暴露 episode 的未来长度。
3. **复用空间预处理和 prompt。**调用 `rgb_frame_to_model_tensor`；当前 checkpoint
   将 470×280 RGB 转成宽 448×高 256 的 `[-1,1]` 张量。使用原有 unified RVQ prompt，
   保留实际指令。processor 使用 `_skip_normalize=True`，不重复归一化。
4. **拼接五个回答位置。**顺序固定为 apos、opos、RVQ l0/l1/l2。通过真实 tokenizer
   把字符串转为全局词表 ID，检查每个动作/pointing token 恰好对应一个词表项。
5. **构造独立 loss mask。**从完整 assistant header 确定 `answer_start`，随后再对
   五个位置应用各通道 mask。prompt 和 padding 的 labels 为 -100；回答结束的
   `<|im_end|>` 参与 CE。因此普通完整样本通常有 6 个监督位置，即五个任务 token 加结束 token。
6. **保留连续 GT。**直接取当前行 `action.fm.clean_se2_diff` 和 `action.fm.valid_mask`，
   保持 `[30,3]` / `[30]`。不再次堆叠未来 30 行、不用 RVQ 重建值替换、也不在读取器中归一化。

RGB 的压缩列按 row group 读入时可能包含未使用的其他帧字节；模型输入只解码和选取
`<=t` 的历史帧。多个指令 episode 即使复用同一物理路线的图像，仍保持各自 episode 索引与指令。

当前 checkpoint 的 Fast 范围是 age 2..33，Mid 是 34..89；不要直接套用
slowfast.py 中另一套默认示例。训练视觉时间戳沿用 checkpoint 的 `video_fps=4` 和
相对时间配置；LeRobot 的 fps=1 是存储索引约定，两者都不能当成实际控制周期。

## Mask、STOP 与缺失值

| 通道 | 输入/GT 处理 | CE/FM 监督 |
|---|---|---|
| apos 有效点 | 原始 `<apos_k>` | 对应位置参与 CE |
| apos unknown | 合法 `<apos_0>` 输入占位 | 对应 label=-100，不改成 apos STOP |
| opos 确认不显示/不可见 | 原始 `<opos_0>` | mask=true 时参与 CE，包括后半段窗口之前的帧 |
| opos unknown | 原始 `<opos_0>` | mask=false 时 label=-100 |
| RVQ 完整未来十步 | 原始三个 token | 三个位置参与 CE |
| RVQ 尾部不足十步 | 合法词表 token 作为输入占位，原始 -1 codes 不改变 | 三个位置全部 label=-100 |
| 显式 STOP | 原始预留 RVQ 三元组，例如 `[6,122,174]` | RVQ CE 有效；FM 全 false |
| FM 部分未来 | 保留真实增量和固定长度数组 | 只监督 mask=true 的步 |

RVQ 无效时使用 `<act_l0_0><act_l1_0><act_l2_0>` 作为输入占位；如果某个 bundle 的
L0=0 是预留 STOP，则改用合法的 L0=1。占位不产生 RVQ GT，也不参与三个位置的 CE。
当前读取器检查数据和 checkpoint 的实际码本文件哈希相同，并验证 STOP、未来帧号和各通道
有效性与 episode 边界一致。损坏数据直接报错，不静默跳过或伪造标签。

`answer_start` 必须在 loss mask 前确定，并随左侧 padding 偏移。它与第一个有效
label 的位置可能不同，例如 STOP 帧的第一个 apos 标签是 -100。
FM 只读取该起点之前的主干上下文，GT 回答不会进入 FM 条件。
序列超过 checkpoint `max_seq_len` 或回答不完整时直接报错，不截断后继续训练。

## 从 batch 到 CE / FM loss

读取器提供干净动作 `a = batch.fm_clean_se2_diff`，不预先固定噪声或 flow time。
`BetaModel.forward(batch)` 每次执行时：

1. VLM 以 teacher forcing 读取 prompt 和 GT 回答，按 `labels` 计算 CE；模型内部
   负责一次 next-token shift，读取器不提前移动 labels。
2. 取 `answer_start` 之前、且非 padding 的 hidden states 作为 FM 条件。
   这些位置受因果 attention 约束，看不到后面的 GT 回答。
3. 主干 hidden states 先 `detach()`，再送入可训练的 context projection 和动作专家。
   因此 CE 更新主干/LM head，FM 更新动作专家及投影。
4. 动作按模型保存的 `action_scale` 归一化为 `a_norm`；重新采样高斯噪声 `eps`
   和 `tau ~ Uniform(0,1)`。有效步的专家输入为
   `x_tau = (1-tau)*eps + tau*a_norm`，预测目标为 `a_norm-eps`。
5. 只在 `fm_valid_mask=true` 的动作步上计算三维速度预测的 MSE，最后组合
   `loss = ce_loss + lambda_fm * fm_loss`。STOP 帧的全 false mask 不产生 FM 监督。

所以数据集保存的是最终希望生成的连续动作，FM 训练时的噪声轨迹和速度标签由模型在线构造。
训练中的“速度”指沿 flow time 的变化量，不是按真实秒数计算的机器人速度。

## Batch 与多进程

返回 `TrainLerobotBatch`，它继承 `BetaTrainBatch`：

| 属性 | 形状/说明 |
|---|---|
| `model_inputs.input_ids` | `[B,L]`，左 padding |
| `model_inputs.attention_mask` | `[B,L]`，bool |
| `model_inputs.pixel_values_videos` | 所有样本/视频按顺序拼接的视觉 patch，二维 |
| `model_inputs.video_grid_thw` | `[所有视频 segment 数,3]`，不是 `[B,T,3]` |
| `_original_video_grid_thw` / `_post_vit_pool_factors` | 同一视频顺序的原始 grid 与池化因子 |
| `labels` | `[B,L]`，全局 token ID 或 -100 |
| `answer_start` | `[B]`，在补齐后序列中的回答起点 |
| `fm_clean_se2_diff` | `[B,30,3]`，float32，原始米/米/弧度 |
| `fm_valid_mask` | `[B,30]`，bool |
| `sample_info` | 帧身份、指令、GT token、mask、SlowFast 帧号；只供检查，不输入 VLM |

视觉 patch/grid 在样本维度拼接，不能像普通图片 batch 那样直接 stack。
旧 processor 的 position_ids 不拼接进模型，BetaModel 使用原生 HF mRoPE 重新计算。
每个 worker 只持有 CPU 数据、自己的 processor 和有限缓存。`spawn` 避免继承已初始化的 CUDA；
worker 不加载 VLM，也不携带 ModelBundle。`pin_memory=True` 时自定义 batch 支持内存固定。

构造参数：

| 参数 | 默认值 | 用途 |
|---|---|---|
| `dataset_root` / `model_path` | 必填 | LeRobot 根目录、LightNav checkpoint |
| `batch_size` / `num_workers` | 2 / 0 | 每批帧样本数、CPU worker 数 |
| `shuffle` / `seed` | true / 0 | 帧样本随机顺序及种子 |
| `episode_indices` | None | 选择原始 LeRobot episode_index；None 使用全部，不能传 source episode ID |
| `drop_last` / `pin_memory` | false / false | 最后一批处理、CPU pinned memory |
| `prefetch_factor` | 1 | 每个 worker 预取 batch 数，仅 workers>0 时启用 |
| `cache_row_groups` / `cache_frames` | 4 / 64 | 每个进程的 Parquet 列组缓存数、预处理图像缓存帧数 |
| `rank` / `world_size` | 0 / 1 | 多卡训练时显式指定，内部构建 DistributedSampler |

每轮调用 `loader.set_epoch(epoch)`。分布式 sampler 在样本数不能被 world_size 整除时，
沿用 PyTorch 的补齐/丢弃规则；读取器不会创建分布式进程组。
训练/验证 episode 列表由调用者提供，建议按 scene 或物理路线分组，避免同路线不同指令泄漏。
动作归一化尺度须从训练划分的有效动作确定，不能对类别 ID 做连续量归一化。

单 worker 的 64 帧 FP32 缓存在当前分辨率下约 84 MiB，另有 row group、当前样本和预取 batch。
先用小 batch/少量 worker 验证内存与吞吐，再增加并发。
当前支持每条 episode 位于一个 data Parquet 文件内、可跨多个 row group；
一个文件可以存多条 episode。跨 data 文件的单条 episode 或 video-backed 数据会明确报错。

## 检查命令

```bash
# 两帧实际 GT + 真实 tokenizer/processor + 小型随机 Qwen3-VL 前向/梯度隔离检查。
# episode_indices 指 LeRobot 编号；sample_indices 在选出的 episode 帧列表中编号。
.venv-train/bin/python -m lightnav.dataloader.Train_Lerobot_DataLoader \
  /mnt/cfs/data/beta_nav/lerobot_r2r_train_470x280_20260912 \
  --model-path /mnt/cfs/Models/Light \
  --episode-indices 0 1 --sample-indices 20 35 \
  --tiny-model-check --report reports/betanav/NEW_LOADER_CHECK.json

# 实际迭代 DataLoader 的首批，包含 spawn worker。
.venv-train/bin/python -m lightnav.dataloader.Train_Lerobot_DataLoader \
  /mnt/cfs/data/beta_nav/lerobot_r2r_train_470x280_20260912 \
  --model-path /mnt/cfs/Models/Light --episode-indices 0 1 \
  --batch-size 2 --num-workers 2
```

| CLI 参数 | 默认/说明 |
|---|---|
| `dataset` | 必填位置参数 |
| `--model-path` | 必填 checkpoint 路径，只加载配置/tokenizer/processor |
| `--batch-size` | 2 |
| `--num-workers` | 0；与 `--sample-indices` 不能同时启用 |
| `--episode-indices` | 可选，一个或多个 LeRobot episode_index |
| `--sample-indices` | 可选，指定检查帧；省略时读取 DataLoader 首批 |
| `--tiny-model-check` | 可选，CPU 小型随机 Qwen3-VL + BetaModel 接口检查，不加载预训练权重 |
| `--report` | 可选，新 JSON 报告路径；拒绝覆盖 |

首个真实数据报告：[train_lerobot_loader_smoke_20260913.json](../reports/betanav/train_lerobot_loader_smoke_20260913.json)。
source episode 1802 的 t=20 / 35，FM 有效步为 15 / 0，CE 监督 token 数为 6 / 5。
真实 tokenizer 和视频 processor 产出 `[2,811]` 输入序列与 `[2,30,3]` FM GT；
小型随机模型确认 FM 梯度不会进入主干。此报告证明数据/模型接口，不代表模型已经训练收敛。

真实 tokenizer/processor 的双 worker 读取也已通过：
[train_lerobot_loader_workers_20260913.json](../reports/betanav/train_lerobot_loader_workers_20260913.json)。
首批 t=0 / 1 得到 `[2,219]` 输入序列与 `[2,30,3]` FM GT，有效步数均为 30。

测试位于 [test_train_lerobot_loader.py](../tests/test_train_lerobot_loader.py)。
使用训练读取环境运行 `make test PYTHON=.venv-train/bin/python`，
确保包含 PyArrow 读取与多进程测试；仅使用原推理环境时，缺少 PyArrow 的新测试会跳过。
2026-09-13 全仓库 CPU 测试结果：739 passed、5 skipped。
