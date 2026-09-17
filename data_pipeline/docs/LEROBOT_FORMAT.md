# LeRobot 保存约定：按帧计时与四类 action GT

状态：2026-09-12 已实现并实际导出。使用官方 LeRobot 0.4.4 写入器生成 v3.0 数据集，
并通过官方 LeRobotDataset、DataLoader、源数组比较和独立几何复核。
采集阶段的 JSON/NPZ 是中间数据；最终训练交付为自包含 LeRobotDataset。
完整 GT 生成定义见 [GT_CONSTRUCTION.md](GT_CONSTRUCTION.md)。
末帧 STOP 和缺失点位的词表对应关系见 [TOKEN_CONTRACT.md](TOKEN_CONTRACT.md)。

真实数据展示：[episode 2864 第 20 帧完整逻辑记录](../reports/lerobot_episode_2864_frame20_example_20260912.json)。
其中的两组 FM `(30,3)` 数组、RVQ、apos/opos 和位姿均读取自现有数据，
全部 30 个源数组的该帧数值已通过 JSON 往返校验。`record.action` 展示完整监督，
`numeric_feature_schema` 给出对应平铺特征名、形状和源 dtype。
示例的 episode_index=0 表示“将该 episode 作为第一个导出”的假定编号，
原始 episode_id=2864 单独保留。它是格式示例；RGB/深度仍为文件引用，
不构成 LeRobot 数据集或原始数据备份。

用户确认：时间根据帧确定；每一帧的 action 同时包含 opos、apos、RVQ
和 FM 的完整未来 GT 轨迹。GT 的单位、坐标系和各通道 mask 继续沿用当前定义。

## 按帧确定时间

第 t 条记录对应第 t 帧，t 从 0 开始。一帧推进一个逻辑时间单位。
FM 的未来 30 步表示 t+1 到 t+30，RVQ 的未来 10 步表示 t+1 到 t+10。
所有未来窗口都限制在当前 episode 内。

LeRobot 的接口包含 fps 和 timestamp；本项目采用以下导出约定：

| 字段 | 约定 |
|---|---|
| `frame_index` | episode 内的帧序号 0..N-1，是采样和窗口的依据 |
| `fps` | 固定 1，用于 LeRobot 文件和视频对齐 |
| `timestamp` | `float(frame_index)`，即 frame_index / fps，每条 episode 从 0 开始 |
| 自定义 `time_basis` | `frame_index` |
| 自定义 `timestamp_kind` | `synthetic` |
| 自定义 `physical_dt_s` | `null`，未测量物理控制周期 |

这里的 timestamp 是给加载器使用的合成秒值，一个合成秒对应一帧，
不代表机器人执行动作的真实耗时。位移保持米、转角保持弧度，不能据此推导物理速度。
在此约定下，时间窗口偏移 10 对应未来 10 帧。

本项目采用官方支持的 image-backed 形式：RGB 原图无损 PNG 字节嵌入 Parquet，
因此不需要训练用 MP4，也不依赖外部 PNG 路径。人工检查视频单独使用 5 FPS；
可视化播放速度不改变数据索引和 GT。多条 episode 可共用 Parquet 分片，
episode 内 frame_index/timestamp 都从 0 开始。

LeRobot 的字段 schema、fps 和写入接口见
[官方数据集 API](https://huggingface.co/docs/lerobot/main/api/datasets)。

## 一帧 action 的逻辑结构

```text
record[t]
  observation：第 t 帧 RGB、深度、机体/相机位姿
  task：这次 episode 的 Instruction
  frame_index：t
  timestamp：float(t)
  action
    opos：终点的像素、三维点、网格、token 和有效性
    apos：未来十帧地面目标及可见回退点的同类字段
    rvq：未来十帧的三个码本索引、重建轨迹和有效性
    fm：未来三十帧的真实轨迹、逐步运动增量和逐步有效性
    explicit_stop / stop_loss_valid：保留原始显式终止监督
```

这是本项目的结构化监督定义。落盘使用平铺的 `action.*` 特征列，训练适配器
可将它们组织为上述字典。各列分别声明 dtype、shape 和语义，保留源数组精度。
标量在 LeRobot feature schema 中按 `(1,)` 声明。

LeRobot 官方示例使用 `action.cartesian_position` 等分列方式；下面的名称是
本项目的自定义 schema，不能假定内置策略会自动识别四类监督。
见 [官方自定义特征示例](https://huggingface.co/docs/lerobot/porting_datasets_v3#droid-features-schema)。

## FM 和 RVQ 的字段映射

下表形状均为单帧形状；来源是 `targets.npz` 中同名数组的第 t 行。

| 导出字段 | 来源 | 形状 | 语义 |
|---|---|---|---|
| `action.fm.waypoints_ego` | `fm_waypoints_ego` | `(30,3)` | 每个未来点相对当前机体的 `[forward_m,left_m,yaw_rad]` |
| `action.fm.clean_se2_diff` | `fm_clean_se2_diff` | `(30,3)` | 每次运动相对前一帧机体的 `[delta_forward_m,delta_left_m,delta_yaw_rad]` |
| `action.fm.valid_mask` | `fm_valid_mask` | `(30,)` | 该未来步是否实际存在 |
| `action.fm.future_frame_ids` | `future_frame_ids` | `(30,)` | 对应未来帧序号，缺失为 -1 |
| `action.fm.vertical_offset_m` | `future_vertical_offset_m` | `(30,)` | 相对当前帧的世界 Y 高度差 |
| `action.rvq.codes` | `rvq_codes` | `(3,)` | 三个码本索引，共同表示未来十帧轨迹 |
| `action.rvq.reconstruction` | `rvq_reconstruction` | `(10,3)` | RVQ 解码的近似路点，用于解码和误差检查 |
| `action.rvq.valid_mask` | `rvq_valid_mask` | `(1,)` | RVQ 通道能否监督 |
| `action.rvq.ade_m` | `rvq_ade_m` | `(1,)` | 量化位置误差，保留源数据中的无效值 |

两个 FM `(30,3)` 数组均须保存。`action.fm.clean_se2_diff[0]` 是下一步运动，
其余有效行提供之后的运动；一个 `(3,)` 的下一步向量不足以表达完整 action。
FM 路点来自量化前的实际位姿，不使用 `action.rvq.reconstruction` 替代。
RVQ 量化前的十个真实点可以读取 `action.fm.waypoints_ego[:10]`，同时遵守 mask。

RVQ token 字符串可由 codes 按已有规则生成；保存码本版本、哈希和 token
格式说明，保证可重建。显式终止的 `explicit_stop` 和 `stop_loss_valid`
从 `samples.jsonl` 保留为同名的 `action.*` 布尔列。

## apos 和 opos 的字段映射

下表中的 `{p}` 分别展开为 `apos` 和 `opos`，两组均需保存。

| 导出字段 | 来源 | 形状 |
|---|---|---|
| `action.{p}.pixel` | `{p}_pixel` | `(2,)`，原始 470×280 图像上的浮点 `[u,v]` |
| `action.{p}.pixel_rc` | `{p}_pixel_rc` | `(2,)`，整数 `[row,col]` |
| `action.{p}.grid_rc` | `{p}_grid_rc` | `(2,)`，48 列×27 行网格的 `[row,col]` |
| `action.{p}.grid_id` | `{p}_grid_id` | `(1,)`，空间格号 |
| `action.{p}.token_id` | `{p}_token_id` | `(1,)`，模型 token 编号 |
| `action.{p}.world` | `{p}_world` | `(3,)`，可见标记的世界点，米 |
| `action.{p}.target_world` | `{p}_target_world` | `(3,)`，可见性判断前的三维目标，米 |
| `action.{p}.target_frame_id` | `{p}_target_frame_id` | `(1,)`，目标来源帧 |
| `action.{p}.valid_mask` | `{p}_valid_mask` | `(1,)`，该通道是否可监督 |

apos 的 target_frame_id=min(t+10,N-1)，opos 的 target_frame_id=N-1。
回退后 apos.world 可以不同于 apos.target_world。opos 始终指向同一个最终目的地。
当前 opos 默认仅在 frame_index >= floor(N/2) 时允许空间点；更早的帧使用
token_id=0、valid_mask=true、reason=before_opos_visibility_window，坐标缺失。
窗口内继续按投影和深度判断可见性；pointing_config 与起始帧需保留在 episode 元数据。
补标 JSON 的 state/reason 等诊断信息保存在来源记录或附加特征中，便于追溯。

## 尾部、缺失值和训练读取

- 不足 30 个未来帧时，FM 保持 `(30,3)` 和 `(30,)` 的固定形状，沿用已有
  padding 和逐步 mask。比如 52 帧 episode 的 t=45 仅有 t=46..51 六步有效。
- FM、RVQ、apos、opos 分别使用自己的 mask；一个通道无效仍保留整帧其他监督。
- 已确认不可见的 opos 保留 token_id=0、valid_mask=true，其空间坐标缺失；
  unknown 也使用 token_id=0、`<opos_0>`，但 valid_mask=false。
  apos 缺失同样保存 token_id=0、`<apos_0>`、valid_mask=false。
  必须保留各通道 state/mask 的区别，不能仅由 0 推断监督有效性。
- 整数缺失空间位置保持 -1，浮点缺失值和三维目标沿用源数据。
  训练及归一化统计须按字段语义排除无效元素，不能将缺失坐标作为真实点。
- 保留源数据中的显式 STOP 及对应 mask；文件结束本身不足以产生 STOP 标签。
- 输入视觉历史截至 t；action 中的未来轨迹和终点属于监督，不作为当前观测输入。
- RVQ/pointing 索引是类别，不能套用连续运动量的归一化方式。

## 导出边界和验证要求

导出器为 `data_pipeline/export_lerobot.py`，在现有采集/补标及校验之后运行。
输出使用 `/mnt/cfs/data/beta_nav` 下的新目录，保留历史原始数据和已冻结的
`final_gt_20260912` 发布包。

RGB、深度、位姿和相机标定继续来自当前 LightNav 配置的 Habitat 回放。
RGB 保存原始相机画面，以 `observation.images.rgb`、dtype=image 写入；
`observation.depth_m` 为 `(280,470)` float32 camera-Z 米值，
`observation.depth_valid` 为相同形状的 bool 数组。两者是 Parquet 数值特征，
不量化成毫米整数，也不使用彩色深度图替代实际深度。

导出实现至少验证：逐 episode 帧数和 Instruction 对齐；所有 GT 数组值与
mask 对齐；timestamp=frame_index 且视频帧一致；未来窗口不跨 episode；
末帧与部分未来窗口保持源标签语义。保留原始 episode/scene ID、相机标定、
源数据哈希和码本版本，训练端增加对应的四类 action 读取与 loss 适配。

实际导出时：30 组原始 GT 数组的类型和值保留；token/state/reason 额外作为字符串
列保存；RGB、深度和全部数值数组逐项比对来源哈希。`meta/beta_nav.json` 保存
每条 episode 的相机、源 ID、Instruction、opos 窗口、逻辑时间与源文件哈希。
`assets/action_tokenizer/` 保存实际使用的码本副本；源采集目录不参与最终加载。

默认官方统计会把缺失值/padding 当普通数值。`lerobot_stats.py` 在单次导出过程中
为官方写入器提供按 mask 计算统计的适配，保留原 GT 数值：FM 按有效未来步汇总
三维运动尺度；RVQ 解码点按有效 chunk 汇总；pointing 像素和可见世界点仅统计
空间 token>0 的行；深度只统计 valid 像素。无有效观测时 count=0，保存有限的中性
统计值，多 episode 聚合跳过这些空统计。类别索引、token 和 mask 不应用连续量归一化。
这不是现成的 BetaModel 训练器；训练代码仍需按 action.* 和独立 mask 构造 loss。

## 运行与加载

在仓库根目录执行。数据集环境使用 Python 3.11，与 Habitat Python 3.9 分离；
锁定 `lerobot==0.4.4`，格式版本为 `v3.0`。更新的 LeRobot 0.6.x 要求 Python 3.12，
本次不升级现有推理/Habitat 环境。安装脚本只补齐数据集功能依赖，复用推理环境的
torch 2.10 / torchvision 0.25；不安装机器人控制或策略训练 extras。

```bash
bash data_pipeline/setup_lerobot.sh

# 完整主入口：回放 -> 校验 -> LeRobot -> 校验 -> GT 视频 -> 删除本次临时采集
bash data_pipeline/run_lerobot.sh /mnt/cfs/data/beta_nav/NEW_LEROBOT_RUN 1

# 已有完整原始数据时，只做格式导出；原始目录不会删除
data_pipeline/.venv/lerobot/bin/python data_pipeline/export_lerobot.py \
  /mnt/cfs/data/beta_nav/RAW_RUN \
  --output-dir /mnt/cfs/data/beta_nav/NEW_LEROBOT_RUN \
  --repo-id beta-nav/r2r --episode-id 5670

# 完全独立验证；加 --source RAW_RUN 可同时比较源数组
HF_HUB_OFFLINE=1 data_pipeline/.venv/lerobot/bin/python data_pipeline/validate_lerobot.py \
  /mnt/cfs/data/beta_nav/NEW_LEROBOT_RUN --report data_pipeline/reports/NEW_CHECK.json
```

| 入口 / 参数 | 默认值与含义 |
|---|---|
| run_lerobot.sh 的第 1 参数 | 必填，全新的最终 LeRobotDataset 目录 |
| 第 2 参数 | 1，episode 数量；-1 为所有，与采集器一致 |
| 后续参数 | 传给 collect_r2r.py；参数见 R2R_COLLECTION.md |
| LEROBOT_PYTHON | data_pipeline/.venv/lerobot/bin/python，可指定兼容的独立环境 |
| export_lerobot.py 的 dataset | 原始完整采集或引用原始观测仍可用的补标目录 |
| --output-dir | 必填，拒绝覆盖的最终目录 |
| --repo-id | beta-nav/r2r，仅作本地标识，不上传 Hugging Face |
| --episode-id | 默认全部，可重复指定多个源 episode ID |
| validate_lerobot.py 的 dataset | 最终 LeRobotDataset 根目录 |
| --source | 可选，额外与源数据比较；日常加载和验证无需源目录 |
| --report | 可选，新的 JSON 校验报告路径，拒绝覆盖 |

当前实际样例是 `/mnt/cfs/data/beta_nav/lerobot_r2r_5670_opos_last_half_20260912`。
60 帧；opos 窗口从帧 30 开始，30–55 共 26 帧可见。目录结构为：

```text
data/chunk-000/file-000.parquet           RGB、深度、位姿、四类 action 和标准索引
meta/info.json                          官方 v3.0 特征 schema、FPS 和条数
meta/stats.json                         按 mask 计算的统计
meta/tasks.parquet                      Instruction / task_index
meta/episodes/chunk-000/file-000.parquet  episode 索引、长度、分片位置与统计
meta/beta_nav.json                      相机、单位、帧时间、窗口、源 ID 与哈希
meta/validation.json                    官方加载器与全部字段的验证报告
meta/provenance/                        代码和原始元数据快照，不参与训练读取
assets/action_tokenizer/                与 RVQ 标签匹配的码本
review/                                单帧可读 JSON、GT 视频；仅供人工检查
```

使用独立环境运行下面的读取代码：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

ds = LeRobotDataset(
    "beta-nav/r2r_5670_opos_last_half",
    root="/mnt/cfs/data/beta_nav/lerobot_r2r_5670_opos_last_half_20260912",
    video_backend="pyav",
)
x = ds[30]
print(x["task"], x["frame_index"], x["timestamp"])
print(x["observation.images.rgb"].shape)   # (3,280,470)，float32，范围 [0,1]
print(x["observation.depth_m"].shape)      # (280,470)，float32，米
print(x["action.fm.waypoints_ego"].shape)  # (30,3)
print(x["action.fm.clean_se2_diff"].shape) # (30,3)
print(x["action.rvq.codes"], x["action.apos.token"], x["action.opos.token"])
batch = next(iter(DataLoader(ds, batch_size=2, num_workers=0)))
```

Parquet 保留源 dtype；官方 PyTorch 转换可能把标量/一维整数列读成 int64，
或把一维 float64 列读成 float32。高精度人工审计使用
`ds.hf_dataset.with_format(None)`，结合 `meta/info.json` 的 dtype 读取；
本次验证同时检查 Parquet 精度和官方 Tensor 读取精度。

## 历史数据记录

2026-09-12 的 [数据目录清单](../reports/beta_nav_inventory_20260912.json)
记录了目录体积、用途、标签引用和外部采集输入；检查没有删除任何数据。
当时原始数据 `r2r_train_470x280_20260911` 有 10 条 episode、506 帧。
最终标签 `r2r_2864_opos_endpoint_20260912` 引用该原始目录中的 RGB/深度，
尚无已转换的 LeRobot 副本。继续使用/转换当前数据时应同时保留这两个目录。

完整清空 beta_nav 会移除现有采集、标注、模型实际运行记录及视频。
代码、R2R 源路线、场景、RVQ 码本和 Habitat 环境位于该目录之外，所查路径存在；
若明确放弃当前结果并重新采集，可将清空作为全量重建的一步。
本次仅检查输入位置，没有重跑 GPU 采集，也不保证模型运行结果逐次一致。
代码发布包和本页的一帧 JSON 示例不能恢复整个图像/深度数据集。

后续状态：执行 pointing_zero 更新时，以上数据集目录已被清空。
原始数据无法迁移；当前代码和新采集/补标均采用零 token 规则，
既有 reports/ 数值示例与版本快照仍保存当时的数据及编码。

随后已重新生成一条完整的 episode 5670，位于
`/mnt/cfs/data/beta_nav/r2r_one_episode_zero_20260912`。
该目录保存实际 JSON/NPZ 数据与 `review/frame_000020.json`、
`review/frame_000059.json` 两个 observation/action 展开的样例，采用上述帧时间语义；
那次仅生成 JSON/NPZ；该原始目录随后被移除。当前已重新回放同一 episode 并实际
导出上述自包含 LeRobotDataset，旧的三分之一标注和视频保留历史版本。
