# BetaModel：LightNav Token + Flow Matching

模型入口固定为 `src/lightnav/inference/beta_model.py`。
这是一个可训练的 PyTorch/HF 模型接口，保留现有 LightNav VLM 与 token 词表，
新增独立连续动作专家。`lightnav-serve`、原有 `build_engine` 不会自动切换到 Beta。
`load_beta_model` 初始化的新专家没有经过训练，不能把它的输出当作导航能力验证。
分布式训练入口为 [train_betanav.py](../train_betanav.py)，超参数、Accelerate/DDP/ZeRO-1、
训练数据准备和恢复命令见 [TRAIN_BETANAV.md](TRAIN_BETANAV.md)。

## 模型与梯度

`training_mode="ce_fm"` 保持下面的双分支行为。新增 `training_mode="fm_only"`：
从原始 LightNav 初始化并冻结整个 VLM（含视觉编码器、语言主干和 LM head），
VLM 始终保持 `eval()`，前向在 `no_grad()` 中执行。输入只有指令、截至 t 的 RGB
历史及回答起始标记，不含任何 GT 回答；不计算 CE，也不使用 RVQ、apos、opos 标签或码本。
只更新随机初始化的 context projection 和 FM 专家，训练 loss 为 `2.0 × FM`。
BF16 训练时冻结主干以 BF16 加载，专家参数与 Adam 状态保持 FP32。

FM-only 评估输入使用 `inference.samples.build_continuous_traj_sample()`，
保持训练时的连续轨迹提示、SlowFast 分段和空回答，不能复用 RVQ 输出提示。
FM-only 推理接口为 `encode_context()` → `sample_actions()`；`predict()` 的旧 RVQ/STOP
流程会明确拒绝此模式。末帧和未记录到的未来动作仍由原始有效 mask 排除，**没有新增 STOP
监督**，因此该实验没有学到自动停止策略。Habitat 评估使用固定步数协议：
连续动作执行至 `max_steps`，由评估器在最后一步强制 STOP，不调用 RVQ。
偶发零速度会使用后续预测航点或小角度转向恢复，避免被模拟器误判为 STOP。
也可显式指定 `--fm_only_stop_mode rvq`，使用冻结 LightNav 的原始 RVQ 提示
判断 STOP，再以连续轨迹提示独立计算 FM 动作；停止判据沿用 baseline。空监督 rank 通过零梯度参与 DDP；
全局有效 FM 数量为零时，训练器跳过 AdamW，避免对无监督分支应用权重衰减。

```text
指令 + 历史/当前 RGB → Qwen3-VL → LM Head → apos/opos/RVQ → CE
                           └→ 回答开始前的隐藏特征
                               → detach → context_proj → FM 专家 → FM loss
```

- CE 更新 VLM/LM Head 的可训练参数；若要冻结 ViT，可在构建 optimizer 前单独设置。
- FM 更新 context projection 和动作专家，梯度不会进入主干。detach 在 projection 前。
- 专家默认 6 层、宽度 512、8 heads、FFN 2048；每层依次执行动作双向 self-attention、
  对上下文的 cross-attention、FFN。流时间通过 sinusoidal embedding、MLP 和 FiLM 注入。
- 主干上下文维度从 `vlm.config.text_config.hidden_size` 读取，当前 Light checkpoint 为 2560。
- 专家输出流场 `[B,30,3]`。Euler 从 `tau=0` 噪声积分至 `tau=1` 干净动作，默认 10 次。
  30 个未来动作步与 10 次流积分是独立参数；流场不是机器人的物理速度。

## 加载

```python
from lightnav.inference.beta_model import BetaModelConfig, load_beta_model
from lightnav.inference.config import InferenceConfig

# 三维尺度必须显式传入；此处只是与当前采集步长同量级的接口示例。
# 正式实验应从训练集确定尺度及零方差维度的下限，并固定用于验证/推理。
beta_config = BetaModelConfig(action_scale=(0.25, 0.25, 0.2617993878))
model, bundle = load_beta_model(
    InferenceConfig(model_path="/mnt/cfs/Models/Light", backend="hf"),
    beta_config,
)
model.train()
```

归一化为 `actions / scale`，保留零增量，模型不自动拟合、裁剪或改变 GT。
`normalizer.scale` 是 state_dict 中的 buffer。新专家默认 FP32，HF 主干沿用加载器精度；
需要混合精度时，在训练调用外使用 autocast。模型不自动创建 optimizer、分布式进程或启动训练。

HF 加载器复用 checkpoint 的 resize、SlowFast 和 post-ViT pooling 设置。
原图 470×280 与模型预处理尺寸是不同阶段。`vllm_local` bundle 仅暴露 ViT，不能用作此训练主干。
Blackwell 上运行完整 HF 权重仍需仓库安装说明指定的兼容 CUDA/Torch 构建。

## 训练 batch

现有 [Train_Lerobot_DataLoader](../src/lightnav/dataloader/Train_Lerobot_DataLoader.py) 已实现从
本项目 LeRobot 数据到下述 batch 的转换，包括 SlowFast、真实 tokenizer、逐通道 mask
和视觉 patch 拼接。直接使用它迭代出的 batch 调用 `model(batch)`；
安装与训练端示例见 [读取器文档](TRAIN_LEROBOT_DATALOADER.md)。

训练数据读取/采样代码归 `src/lightnav/dataloader/`，数据生产代码归 `data_pipeline/`；
模型接收已经对齐、完成各通道 mask 的 tensor。

| 字段 | 约定 |
|---|---|
| `model_inputs` | HF 输入 dict，token 张量 `[B,L]`；包含完整 prompt 和 teacher-forced answer |
| `labels` | int64 `[B,L]`，各通道无效标签、prompt 和 padding 设为 `-100` |
| `answer_start` | int64 `[B]`，**补齐后的序列**里第一个回答 token 的位置，必须在 loss mask 之前确定 |
| `fm_clean_se2_diff` | float `[B,H,3]`，量化前真实局部增量，单位为米/米/弧度 |
| `fm_valid_mask` | bool `[B,H]`，True 表示实际记录到的未来动作 |

主干内部执行一次 causal forward。CE 只投影有监督的预测位置，按 next-token 规则移位；
FM 只抽取 `answer_start` 之前的有效 hidden states，不含任何 GT 回答位置。
`labels == -100` **不能**充当 FM 的上下文 mask；无效 apos/RVQ 回答同样可能被设为 -100。

现有 `bundle.data_processor.process_sample` 返回单样本 token 向量，调用模型前需要 batch 维。
批量构造时 token/labels 分别 pad，`answer_start` 随左侧 padding 偏移；视觉 patch、grid、
原始 grid 与 per-video pooling factors 按样本及视频顺序拼接，不能把不同视频简单 stack。
模型接收 `_original_video_grid_thw` / `_post_vit_pool_factors` 并传递给已有 pooling patch。
处理器的旧 `position_ids` 不直接传给 HF；模型重建模态 ID，由当前固定版本的 Qwen3-VL 计算 mRoPE。
只支持普通二维 padding mask，保持主干 causal attention，不支持外部缓存或自定义双向 attention mask。

训练数据契约详见 [读取器文档](TRAIN_LEROBOT_DATALOADER.md)：RVQ 10 步、FM 30 步，
每步增量相对前一步局部坐标系。FM 使用量化前 GT，不使用 RVQ 重建轨迹。
历史 semantic opos 和当前 terminal-ground opos 不能在不标明版本的情况下混用。

```python
import torch
from lightnav.inference.beta_model import BetaTrainBatch

batch = BetaTrainBatch(
    model_inputs=processed_batch,
    labels=masked_token_labels,
    answer_start=answer_start_before_loss_mask,
    fm_clean_se2_diff=clean_actions,
    fm_valid_mask=valid_future_steps,
)
groups = model.optimizer_parameters()
optimizer_vlm = torch.optim.AdamW(groups["vlm"], lr=1e-5)
optimizer_fm = torch.optim.AdamW(groups["fm"], lr=1e-4)
# 学习率仅为启动示例，需结合 batch size、微调范围和验证结果调整。
optimizer_vlm.zero_grad(set_to_none=True)
optimizer_fm.zero_grad(set_to_none=True)
out = model(batch, lambda_fm=2.0)  # DDP 也应通过 forward 调用
out.loss.backward()
torch.nn.utils.clip_grad_norm_(groups["vlm"], 1.0)
torch.nn.utils.clip_grad_norm_(groups["fm"], 1.0)
if out.ce_token_count:
    optimizer_vlm.step()
if out.fm_step_count:
    optimizer_fm.step()
```

`out` 提供 `loss/ce_loss/fm_loss`、有效 token/动作步计数、专家 context 和预测流场。
测试可显式传 `noise=` 与 `flow_time=` 固定随机性；训练默认在线采样。
`lambda_fm=0` 禁用专家分支，其 loss 为零、context/velocity 为 None。
使用 DDP 时，某个 rank 可能整批没有 FM 或 CE 监督，需要处理 unused parameters
（例如 `find_unused_parameters=True`）。不同 rank 的有效元素数不同时，训练器还需
按全局计数加权损失；当前模型返回的是本 rank 的有效元素均值。

FM 对有效动作元素取均方误差，分母为 `3 * 有效步数`。未知尾部进入专家时使用纯噪声，
不参与损失；mask 不作为输入特征或专家 attention mask。无效存储值可以为 NaN，
有效 GT 必须有限。没有 FM 监督的整批数据返回与专家参数断开的零 loss；结合
`zero_grad(set_to_none=True)`，即使误调用 AdamW.step 也不会对专家施加权重衰减。
全无 CE 标签时同理，不会通过一个人为零梯度触发主干权重衰减。

## 推理与 STOP

连续动作接口返回物理量，而不是归一化值：

```python
model.eval()
context = model.encode_context(prompt_only_inputs)
se2_deltas, waypoints = model.sample_actions(context, num_steps=10)
```

`prompt_only_inputs` 必须只有视觉/指令/assistant 开始标记；若保留 `labels`，它们必须全部为 -100。
`waypoints` 是 SE(2) 组合后的当前机体坐标路点。该低层接口**不判定 STOP**。
不要从 FM 的短小位移推断任务已经结束，也不要把位移直接当作 m/s。

完整 Token+FM 推理复用原有 RVQ decoder：

```python
from lightnav.tracking import TrackingAgent
from lightnav.traj_vocab import load_rvq_bundle

rvq = load_rvq_bundle("/mnt/cfs/Models/Light/action_tokenizer", horizon=10)
decoder = TrackingAgent(engine=None, rvq_bundle=rvq)  # 仅复用 decode，不创建第二个引擎
prediction = model.predict(
    prompt_only_inputs,
    tokenizer=bundle.tokenizer,
    action_decoder=decoder,
)
# prediction.texts / token_ids / rvq_waypoints 用于检查原有分支。
# prediction.stop / se2_deltas / waypoints 为连续分支的执行接口。
```

`predict` 使用 HF greedy generation，batch 必须左侧 padding；只捕获首个 prefill 的最后层隐藏特征，
后续生成的 answer tokens 不进入专家。FM 复用这些特征，不重复运行 VLM prefill。
输入模型不能被同时并发调用；此实现面向 HF eager，未接入 vLLM 或编译后的生成路径。

现有 `TrackingAgent` 的 RVQ STOP 判定完整保留：显式 stop L0 或近零 RVQ 解码都转为停止。
停止行不运行 FM，输出全零轨迹及 `stop=True`。畸形 token 继续报错，不伪装成 STOP。
新管线的末帧 FM mask 全为 False，不能训练为 30 步零动作；末帧 STOP 由 CE 学习。
若未来改用独立 StopHead，需新增可靠的继续/停止监督，不能只用当前末帧正例字段。

## 保存与恢复

必须保存 **完整 Beta state_dict**，因为 CE 更新了主干。只保存专家会丢失 CE 学到的变化。

```python
from dataclasses import asdict
from lightnav.inference.beta_model import BetaModel

torch.save({
    "beta_config": asdict(model.config),
    "model": model.state_dict(),
    "optimizer_vlm": optimizer_vlm.state_dict(),
    "optimizer_fm": optimizer_fm.state_dict(),
}, "beta_checkpoint.pt")

state = torch.load("beta_checkpoint.pt", map_location="cpu", weights_only=True)
# bundle.model 须按相同 HF 架构、词表、预处理配置加载。
restored = BetaModel(bundle.model, BetaModelConfig(**state["beta_config"]))
restored.load_state_dict(state["model"], strict=True)
```

恢复训练还应保存/恢复训练器的全局 step、scheduler、随机状态及数据划分版本。
`load_beta_model` 本身只初始化新专家，不会自动发现上面的 Beta checkpoint。

## 验证

`tests/test_beta_model.py` 使用随机初始化的小型 **真实 Qwen3-VL** 做 CPU 验证，
覆盖 CE/FM 梯度隔离、GT answer 不泄漏、独立 mask、损失移位、无效尾部、
Euler 积分方向、SE(2) 组合、保存恢复、真实视频池化/mRoPE，以及 RVQ STOP 与 prefill 复用。
运行 `make test PYTHON=.venv/bin/python`。CPU 测试证明接口与数值逻辑；
完整 checkpoint 的 GPU 训练、收敛和导航成功率需要单独验证。
