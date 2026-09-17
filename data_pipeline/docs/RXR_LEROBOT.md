# RxR → 相同四类 GT → LeRobotDataset v3.0

更新日期：2026-09-16。RxR 使用与 R2R 相同的 Habitat 回放、FM/RVQ/pointing 几何和
LeRobot 导出器，入口为 `data_pipeline/run_rxr.sh`。生成代码不会修改现有 R2R 数据。

## 范围与输入

本机源目录为 `/mnt/cfs/data/habitat/datasets/RxR_VLNCE_v0`。实际检查结果：

| split | guide episode | follower episode | 专家动作 GT |
|---|---:|---:|---|
| train | 60,300 | 58,752 | 两类均有 |
| val_seen | 6,746 | 6,571 | 两类均有 |
| val_unseen | 11,006 | 10,847 | 两类均有 |
| test_challenge | 9,557 | 本机无 follower 文件 | 无，不能按专家 GT 流程生成 |

默认选择 `train + guide + all`。guide/follower 和不同 split 使用不同输出目录。
`train/train_guide.json.gz` 提供指令、起点、场景、目标等，
`train/train_guide_gt.json.gz` 提供 `actions`、`locations` 和 `forward_steps`。
follower 对应文件名为 `train_follower*.json.gz`。

train guide 的语言分布：en-US 6,660、en-IN 13,336、hi-IN 20,221、te-IN 20,083。
只保留两种英语时是 19,996 条；默认不继承评测服务中的 English-only 筛选。
源 Instruction 不翻译、不按文本去重。语音时间标记保留在原始源文件中，
不用于当前按帧确定的逻辑时间轴。

## RxR 动作需要先适配

RxR 原始转向动作是 30°；现有 LightNav/R2R 回放每次转向为 15°，前进一步同为 0.25 m。
直接按 15° 执行 RxR 原始动作会改变路线。首次试跑已被路线/终点检查拒绝，没有发布错误数据。
本机 GT 位置方向检查确认了 30° 约定，也与
[VLN-CE 官方 RxR 配置](https://github.com/jacobkrantz/VLN-CE/blob/master/habitat_extensions/config/rxr_vlnce_english_task.yaml) 一致。

`vlnce_source.py::rxr_replay_actions()` 将每次左/右转展开为两次相同方向的 15° 转向，
前进动作与唯一的末尾 STOP 保持不变。插入的中间帧由 Habitat 真正渲染，位姿也实际记录；
FM、RVQ、apos 的未来窗口都基于这个新帧序列，opos 后半段边界也使用新的帧数。
三维路线检查仍使用未经改写的原始 GT locations。

全语言 train guide 原始动作对应 5,686,821 帧，其中有 1,843,183 次转向，
适配后预计 **7,530,004 帧**。原始/新帧数和动作转换参数保存在 source_selection 及逐 episode source_metadata。
这一规则保持与当前 LightNav 15° 数据采样一致；不能把原始帧序号直接作为新数据的帧序号。

## RxR 导航网格与碰撞设置

相机仍使用 LightNav 参数；导航网格和碰撞采用上述官方 RxR 配置：agent 高度 0.88 m、
半径 0.18 m、`allow_sliding=false`。每个场景在内存中调用 `recompute_navmesh`，
其余网格参数使用记录的 Habitat-Sim 版本默认值，不写入或覆盖源 `.navmesh`。
R2R 仍使用既有导航网格和原配置。

这是回放原始 RxR 路线所必需的适配：episode 58477 在原网格下卡在楼梯，终点距离为
6.87 m；新网格和 RxR 碰撞设置下，15° 展开回放的最大路线偏差约 0.000005 m，
终点距离为 0.435 m。原生 30° 回放也精确复现了源 GT 位置。
诊断结果见 [rxr_replay_probe_navmesh_58477_20260916.json](../reports/rxr_replay_probe_navmesh_58477_20260916.json)。
传感器和位姿仍全部从这次 Habitat 回放采集，没有将旧 GT 位姿写作新观测。

这项修正改变生成定义，修正前 `.work` 只作历史诊断保留，不继续混写。
当前生产目录为 `/mnt/cfs/data/beta_nav/lerobot_rxr_train_guide_470x280_rxrnav_20260916`，
其完整性以 `.work/status.json` 和最终 `meta/full_train_validation.json` 为准。

修正后的独立小样本已导出至 `/mnt/cfs/data/beta_nav/lerobot_rxr_58477_rxrnav_20260916`：
1 条 episode、111 帧、49 个逐帧字段，与现有 R2R schema 完全一致。
已通过官方加载、源数组/图片/深度核对，以及移除临时采集文件后的独立加载检查。
该路线 apos 可见 48 帧；opos 虽在后半段开放，但终点没有通过可见性检查，所以其 token 全部为 0。
样本中也保存 GT 视频 `review/episode_58477_gt.mp4`。
校验及第 55 帧字段见 [rxr_58477_rxrnav_validation_20260916.json](../reports/rxr_58477_rxrnav_validation_20260916.json)。

## 生成流程与字段

```text
按 dataset / split / role / languages 读取原始 episode 与 GT
  → 将 RxR 每次 30° 转向展开为两次 LightNav 15° 转向
  → 检查 ID、指令、语言、GT、场景与 navmesh
  → 在内存中构建 RxR 尺寸的导航网格，使用 RxR 碰撞设置
  → 同一物理输入的路线可复用回放，每条指令单独保留
  → LightNav 配置的 Habitat 新 RGB / 深度 / 位姿
  → FM-30、RVQ-10、apos-10、后半段且真实可见的终点 opos
  → 分批 LeRobot 导出和独立验证
  → 合并全局索引，验证指令/语言/来源覆盖，发布最终目录
```

相机与 GT 定义沿用 [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md)：

- 原生 470×280，水平 FOV 120°，相机高度 0.88 m，俯仰 0°。
- RGB 为 Parquet 内无损 PNG；深度为 float32 camera-Z 米值和 bool mask。
- FM 保存 `(30,3)` 路点、逐步 SE(2) 增量以及独立的逐步 mask。
- RVQ 保存三个 code、token、十步解码轨迹与 mask；末帧单独验证 STOP。
- apos 指向 `min(t+10,N-1)` 地面目标，必要时沿 GT 路线回退找可见点。
- opos 固定为实际末帧地面位置，只在 `t >= floor(N/2)` 且实际可见时保存空间点。
- 缺失 pointing token 为 0；不可见与未知仍由 state/mask 区分。
- `fps=1`、`timestamp=frame_index`、`physical_dt_s=null`；内部动作回放不作为离散训练类别导出。

最终逐帧列与 R2R 相同，因此现有 BetaModel 数据读取器可读取相同 schema。
RxR 的语言与来源记录在每条 episode 的 `source_metadata`，包含 dataset、split、role、language，
以及源文件中存在的 instruction_id、annotator_id、edit_distance、demonstration_id。
这些元数据保存在 `meta/beta_nav.json`、生成计划和 episode 来源快照中；合并时核对。
不同指令共享一次物理回放时，仍写出各自的语言与标注 ID。

LeRobot 标准 `meta/info.json` 的 `splits.train` 表示当前文件集合的索引范围；
原始 RxR split 由上述元数据明确标识。验证集放在单独目录，不能因为这个存储索引名而加入训练。

## 命令

在仓库根目录执行；新版本使用新输出路径。环境沿用 `setup_lerobot.sh` 和独立 Habitat Python 3.9。
GPU 编号按可用设备选择。首次建议先跑一批，验证后用同一目录继续全量。

```bash
# 全量计划和输入检查，不渲染、不创建数据目录
bash data_pipeline/run_rxr.sh /mnt/cfs/data/beta_nav/NEW_RXR_TRAIN_GUIDE --dry-run

# 小批量验证；保留完整计划以便续跑
bash data_pipeline/run_rxr.sh /mnt/cfs/data/beta_nav/NEW_RXR_TRAIN_GUIDE \
  --max-new-shards 1 --workers 1 --gpus 0

# 完整 train guide，保留全部语言；中断后重复相同命令继续
bash data_pipeline/run_rxr.sh /mnt/cfs/data/beta_nav/NEW_RXR_TRAIN_GUIDE \
  --workers 4 --gpus 0 1

# 英语子集，使用另一个目录
bash data_pipeline/run_rxr.sh /mnt/cfs/data/beta_nav/NEW_RXR_TRAIN_GUIDE_EN \
  --languages en-US en-IN --workers 4 --gpus 0 1

# follower 与验证集分别生成，不能合并进同一来源计划
bash data_pipeline/run_rxr.sh /mnt/cfs/data/beta_nav/NEW_RXR_TRAIN_FOLLOWER --role follower
bash data_pipeline/run_rxr.sh /mnt/cfs/data/beta_nav/NEW_RXR_VAL_SEEN --split val_seen
bash data_pipeline/run_rxr.sh /mnt/cfs/data/beta_nav/NEW_RXR_VAL_UNSEEN --split val_unseen
```

| 参数 | 默认值 / 含义 |
|---|---|
| 第一个位置参数 | 最终 LeRobotDataset 新目录；生产中使用同名 `.work/` |
| `--split` | `train`；另支持 `val_seen`、`val_unseen` |
| `--role` | `guide`；可选 `follower` |
| `--languages` | `all`；或 `en-US en-IN hi-IN te-IN` 的非空子集 |
| `--workers` / `--gpus` | `4` / `1 3`；并发批次数和物理 GPU 编号 |
| `--routes-per-shard` | `8`；每批最多同场景物理路线数，属于冻结计划的一部分 |
| `--max-new-shards` | 默认不限；限制本次新增批次数，之后去掉可继续 |
| `--retries` | `1`；每批失败后额外重试次数 |
| `--dry-run` | 只核对源数据、GT、场景/navmesh 并打印计划 |
| `--data-root` | 本机 `RxR_VLNCE_v0` 源目录 |
| `--scenes-dir` | `/mnt/cfs/data/habitat/scene_datasets` |
| `--rvq-bundle` | `/mnt/cfs/Models/Light/action_tokenizer` |
| `--pointing-config` | `data_pipeline/configs/pointing.json`，全量入口要求 `last_half` |
| `--repo-id` | 自动为 `beta-nav/rxr-<split>-<role>`；仅本地名称，不上传 |

共用底层入口 `build_r2r_train.py` 新增 `--dataset r2r|rxr`，默认仍为 R2R。
共用采集器 `collect_r2r.py` 支持相同的 dataset/split/role/languages 选择。
wrapper `run_rxr.sh` 负责传入 `--dataset rxr`，无需复制另一套 GT 计算代码。

改变 split、role、语言、相机、码本或标注定义时必须换新输出目录；
已有工作目录只接受同一生成计划。`--workers`、`--gpus`、重试次数及试跑上限可以调整。

## 检查与代码入口

- `vlnce_source.py`：原始文件选择、语言筛选、来源标识、缺失 GT/重复 ID 检查。
- `build_r2r_train.py`：同路线渲染复用、全量覆盖、指令展开、续跑调度。
- `collect_r2r.py` / `geometry.py` / `pointing.py`：与 R2R 共用的回放和四类 GT。
- `export_lerobot.py` / `merge_lerobot.py`：相同逐帧 schema，保存和核对 RxR 元数据。
- `tests/test_rxr_source.py`：多语言、重复文本、独立来源、同路线不同语言及官方格式往返测试。
- `probe_rxr_replay.py`：对比 15°/30°、滑动碰撞和 RxR 网格；只输出诊断报告，不生成训练标注。

诊断脚本在独立 Habitat Python 环境运行，必填 `--episode-id ID [ID ...]` 与新报告路径
`--output FILE`；`--gpu-id` 默认 0，`--split/--role` 默认 train/guide，
`--data-root/--scenes-dir` 默认本机输入路径。加 `--rxr-navmesh` 测试内存重建的 RxR 网格，
不加则测试已有网格。每次比较 15°/30° 和开启/关闭滑动四个组合。

生产中先查看 `.work/status.json`、`.work/logs/` 和通过校验的 `.work/shards/`。
只有完整覆盖且合并验证通过后才发布最终目录；最终查看 `meta/summary.json`、`meta/info.json`、
`meta/full_train_validation.json`。最后这个报告名是共用流程的兼容名称，真实 split 见 source_selection。
当前文档中的命令是执行方法，完成数量以实际状态和最终验证报告为准。

当前冻结计划摘要见 [rxr_train_guide_rxrnav_plan_20260916.json](../reports/rxr_train_guide_rxrnav_plan_20260916.json)：
60,300 条 episode、8,396 条物理路线、59 个场景、1,078 个生产批次，
物理渲染 1,050,063 帧，按全部指令展开为 7,530,004 帧。
修正前通过校验的样例字段见 [rxr_first_shard_frame_example_20260916.json](../reports/rxr_first_shard_frame_example_20260916.json)。
修正前首批 8 条路线的回放检查见 [rxr_turn15_replay_20260916.json](../reports/rxr_turn15_replay_20260916.json)，
实施与验证状态见 [rxr_implementation_20260916.json](../reports/rxr_implementation_20260916.json)。
