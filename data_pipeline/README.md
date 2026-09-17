# 最终版数据生产管线

当前代码：**后半段 opos + LeRobotDataset v3.0 导出**。本目录是数据管线的统一入口。
完整生成说明见 [当前数据管线总览](docs/PIPELINE_OVERVIEW.md)（2026-09-14 核对）：
从 R2R 回放、四类 GT 计算、尾部与缺失点处理，到全量分批、Parquet/chunk、真实样本和核心代码入口。
已导出的 LeRobot 数据可直接使用 `visual/render_episode_gt.py --episode-id` 生成未来 10 步、
apos/opos 视频，无需保留采集 scratch。读取方式、预览命令和输出约定见
[GT 可视化说明](docs/VISUALIZATION.md#直接读取-lerobot-parquet)。
完整 R2R train 生产使用 [run_r2r_train.sh](run_r2r_train.sh)：保留全部 10,819 条指令、
631,244 帧，支持并行分批、失败重试、断点续跑和标准 LeRobot 合并。
入口、状态文件与训练读取示例见 [FULL_R2R_TRAIN.md](docs/FULL_R2R_TRAIN.md)。
RxR 使用 [run_rxr.sh](run_rxr.sh) 接入同一套回放、四类 GT 和 Parquet 导出，
支持 train/val_seen/val_unseen、guide/follower 及语言筛选；各来源分别保存。
输入范围、命令和元数据约定见 [RXR_LEROBOT.md](docs/RXR_LEROBOT.md)。
BetaModel 的训练读取入口位于 `src/lightnav/dataloader/`：
[Train_Lerobot_DataLoader.py](../src/lightnav/dataloader/Train_Lerobot_DataLoader.py)，
按需读取 Parquet，复用 checkpoint 的 SlowFast 采样，直接产出 `BetaTrainBatch`。
独立训练读取环境、batch 字段、mask 与使用示例见 [TRAIN_LEROBOT_DATALOADER.md](../docs/TRAIN_LEROBOT_DATALOADER.md)。
[prepare_data.py](../src/lightnav/training/prepare_data.py) 生成训练清单和仅从训练有效动作计算的归一化尺度；
当前 R2R 使用全部 episode 训练，不划分 validation。训练环境、manifest 和报告已迁出本目录；
已有 LeRobot 数据可以直接训练，不依赖本目录代码。训练入口及参数见 [TRAIN_BETANAV.md](../docs/TRAIN_BETANAV.md)。
先读 [最终流程与人工检查顺序](docs/FINAL_PIPELINE.md)，再按下表查看代码。
清理时的冻结代码包和逐文件哈希位于 [releases/](releases/README.md)，
这些历史快照不包含后续 pointing_zero 修改。

## 核心代码

| 顺序 | 文件 | 人工检查重点 |
|---|---|---|
| 1 | [run.sh](run.sh)、[configs/pointing.json](configs/pointing.json) | 本机环境、数据路径、相机入口、点位配置 |
| 2 | [collect_r2r.py](collect_r2r.py) | R2R 专家回放、RGB/深度/位姿采集、写出全部 GT |
| 3 | [geometry.py](geometry.py) | FM-30 路点与逐步 SE(2) 增量、RVQ-10 编码、路线偏差 |
| 4 | [pointing.py](pointing.py) | apos 十步目标与回退、opos 固定终点、深度可见性、像素/网格 |
| 5 | [annotate_r2r.py](annotate_r2r.py) | 已有原始轨迹的 CPU 补标，保存独立版本 |
| 6 | [validate_r2r.py](validate_r2r.py)、[validate_pointing.py](validate_pointing.py) | 帧对齐、源哈希、独立 mask、几何和 token 检查 |
| 7 | [visual/render_episode_gt.py](../visual/render_episode_gt.py) | GT 轨迹、同心 apos/opos 标记、视频及像素格子图 |
| 8 | [run_lerobot.sh](run_lerobot.sh)、[export_lerobot.py](export_lerobot.py) | 采集到 LeRobot 的完整入口、标准 Parquet/嵌入图片写入 |
| 9 | [lerobot_stats.py](lerobot_stats.py)、[validate_lerobot.py](validate_lerobot.py) | 排除无效 GT 的统计；官方加载器、全部字段与独立几何校验 |
| 10 | [build_r2r_train.py](build_r2r_train.py)、[merge_lerobot.py](merge_lerobot.py) | 完整 train 指令覆盖、同路线渲染复用、续跑、合并与索引校验 |
| 11 | [run_rxr.sh](run_rxr.sh)、[vlnce_source.py](vlnce_source.py) | RxR split/role/语言、30° 转向展开、来源记录；复用同一 GT/导出流程 |

共享依赖：[RVQ 解码器](../src/lightnav/traj_vocab.py)、
[网格 token 工具](../src/lightnav/vln_utils.py)、
[LightNav 相机/运动配置](../habitat_server/configs/vlnce_r2r.yaml)。

## 固定的生成定义

```text
R2R 场景、起点、指令和专家 GT 路线
  → LightNav 配置的 Habitat-Sim 回放
  → 每步新的 RGB、深度、机体和相机位姿
  → FM：未来 30 步真实路点 + 逐步运动增量
  → RVQ：同一轨迹前 10 步 → 三层码本索引 + 解码轨迹
  → apos：min(t+10,N-1) 地面目标；必要时沿未来 GT 路径回退
  → opos：固定末帧三维地面位置；仅最后二分之一允许空间标记，仍检查深度
  → 每类独立 mask、像素与 48×27 网格、数据与代码来源记录
  → LeRobotDataset v3.0：Parquet 内保存 RGB、深度与 action.* → 验证 / 视频 / 人工检查
```

- RGB/深度原生 470×280，HFOV 120°，相机高 0.88 m、俯仰 0°。
- 运动量采用前向米、左向米、逆时针 yaw 弧度；FM 不使用 RVQ 重建轨迹当 GT。
- 一个 step 是一次实际专家采样转移，包含原地转向；physical_dt_s=null。
- 新样本不输出前进/左转/右转类别。RVQ 的显式 STOP 监督保留。
- apos 与 opos 在不可见时仍保存三维目标；缺失像素不会伪造为图像边缘点。
- opos 默认从帧 `floor(N/2)` 开始允许显示，帧号从 0 开始且包含边界帧。
  此前为 `<opos_0>`、mask=true、`reason=before_opos_visibility_window`；
  进入窗口后仍按真实投影和深度判断。配置为 `opos_visibility_window=last_half`。
- FM、RVQ、apos、opos 分别使用自己的 mask；完整定义见 [GT_CONSTRUCTION.md](docs/GT_CONSTRUCTION.md)。
- apos/opos 缺失 token 统一为 0，始终保存 `<apos_0>` / `<opos_0>`。
  unknown 仍为 mask=false，opos 确认不可见仍为 mask=true；空间坐标保持缺失。
  `pointing_config.unknown_token_id=0` 标识新规则，历史 -1 编码仅用于只读兼容。
- 新采集和补标只生成 terminal_ground + terminal，当前流程无需豆包或语义建模。
- 原始数据和大型生成结果保存在 /mnt/cfs/data/beta_nav，已有数据不覆盖。

LeRobot 导出已实现：以帧序号计时，每帧 action 同时保存 opos、apos、
RVQ 和完整 FM-30 GT，使用各自 mask。使用 fps=1、timestamp=frame_index
对齐合成时间轴，physical_dt_s 仍为 null。字段映射见
[LEROBOT_FORMAT.md](docs/LEROBOT_FORMAT.md)。RGB 无损 PNG 字节嵌入 Parquet，深度是
原精度 float32 米值；不依赖原始图片目录。数据集由官方 LeRobot 0.4.4 写出并读取验证。

## 运行

在仓库根目录运行；输出路径必须是新目录。

```bash
# 首次准备独立的数据集环境（复用已有 torch，不改 Habitat/推理环境）
bash data_pipeline/setup_lerobot.sh

# 推荐：直接得到完整 LeRobotDataset，校验后删除本次临时原始数据
bash data_pipeline/run_lerobot.sh /mnt/cfs/data/beta_nav/NEW_LEROBOT_RUN 1

# 完整本地 R2R train；重复相同命令可以续跑
bash data_pipeline/run_r2r_train.sh /mnt/cfs/data/beta_nav/NEW_FULL_TRAIN --workers 4 --gpus 1 3

# 已有完整原始数据时，可只导出
data_pipeline/.venv/lerobot/bin/python data_pipeline/export_lerobot.py \
  /mnt/cfs/data/beta_nav/RAW_RUN --output-dir /mnt/cfs/data/beta_nav/NEW_LEROBOT_RUN

# LeRobot 文件独立校验（不需要 RAW_RUN）
HF_HUB_OFFLINE=1 data_pipeline/.venv/lerobot/bin/python data_pipeline/validate_lerobot.py \
  /mnt/cfs/data/beta_nav/NEW_LEROBOT_RUN

# 只检查采集配置，不渲染、不生成数据；本机 Habitat 环境由 run.sh 指定
bash data_pipeline/run.sh /mnt/cfs/data/beta_nav/NEW_RUN 1 --dry-run

# 从 R2R 回放并同时生成全部四类 GT
bash data_pipeline/run.sh /mnt/cfs/data/beta_nav/NEW_RUN 10

# 已有完整原始轨迹：CPU 重新生成 apos/opos，复用运动 GT
.venv/bin/python data_pipeline/annotate_r2r.py \
  /mnt/cfs/data/beta_nav/r2r_one_episode_zero_20260912 \
  --output-dir /mnt/cfs/data/beta_nav/NEW_POINTING_RUN \
  --episode-id 5670 --episodes 1

# 只读验证
.venv/bin/python data_pipeline/validate_r2r.py /mnt/cfs/data/beta_nav/NEW_POINTING_RUN

# 人工检查视频
.venv/bin/python visual/render_episode_gt.py \
  /mnt/cfs/data/beta_nav/NEW_POINTING_RUN/episodes/5670 \
  --output /mnt/cfs/data/beta_nav/visual/NEW_GT.mp4 --fps 5 --horizon 10 --scale 2

# CPU 测试
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q data_pipeline/tests
```

采集参数见 [R2R_COLLECTION.md](docs/R2R_COLLECTION.md)，补标参数见
[POINTING.md](docs/POINTING.md)，视频参数见 [VISUALIZATION.md](docs/VISUALIZATION.md)。
Habitat 使用独立 Python 3.9 环境，离线补标和视频可使用仓库 .venv。

## 辅助工具与资料

- [export_pointing_grid.py](export_pointing_grid.py)：旧标注升级像素/网格字段并导出 CSV。
- [rollout_lightnav.py](rollout_lightnav.py)：真实模型闭环预测与视频，使用方法见
  [LIGHTNAV_ROLLOUT.md](docs/LIGHTNAV_ROLLOUT.md)；其输出不作为专家 GT。
- [GT_CONSTRUCTION.md](docs/GT_CONSTRUCTION.md)：数组语义、单位、mask 与完整数值例子。
- [TOKEN_CONTRACT.md](docs/TOKEN_CONTRACT.md)：末帧 RVQ STOP、不可见/未知点位及本地词表核对结果。
- [LEROBOT_FORMAT.md](docs/LEROBOT_FORMAT.md)：按帧计时、四类 action GT 的导出字段约定。
- [LeRobot 第 20 帧真实数据示例](reports/lerobot_episode_2864_frame20_example_20260912.json)：
  完整 FM-30、RVQ、apos/opos 和位姿；格式展示，尚未导出为 LeRobot 数据集。
- [数据清理前目录清单](reports/beta_nav_inventory_20260912.json)：当前目录体积、来源引用及重建输入位置。
- [INTERNNAV_FM_COMPARISON.md](docs/INTERNNAV_FM_COMPARISON.md)：与 InternNav 的运动定义对照。
- [HISTORY.md](docs/HISTORY.md)：历史数据路径和各次生成记录；reports/ 保存小型报告。
- [archive/](archive/README.md)：清理前的代码和文档，不参与当前运行。
- 本机 config.py 仅保留账户配置，当前 GT 流程不读取它；不进入 Git、快照或代码包。

完整 R2R train 已生成：
`/mnt/cfs/data/beta_nav/lerobot_r2r_train_470x280_20260912`，
10,819 条 episode、3,603 条路线、61 个场景、631,244 帧、10,815 个不同 task 文本。
完整覆盖和官方加载检查见该目录 `meta/full_train_validation.json`；
先查看小型 `meta/summary.json` 与标准 schema `meta/info.json`。
`meta/quality.json` 列出 15 帧原始黑图/全无效深度及其索引，这些帧已保留真实数据和 mask。
生产与验证报告见 [full_r2r_train_build_20260912.json](reports/full_r2r_train_build_20260912.json)，
重跑及训练读取见 [FULL_R2R_TRAIN.md](docs/FULL_R2R_TRAIN.md)。

此前的单 episode LeRobotDataset 示例为
`/mnt/cfs/data/beta_nav/lerobot_r2r_5670_opos_last_half_20260912`，含 1 条 episode、60 帧。
opos 从帧 30 开始开放，30–55 共 26 帧可见，apos 可见 53 帧。
先查看 `meta/info.json`（标准 schema）、`meta/beta_nav.json`（相机/帧时间/来源/策略）、
`review/frame_000030.json`（完整 action 示例）与 `review/episode_5670_gt.mp4`。
正式训练数据在 `data/chunk-000/file-000.parquet`；完整 RGB/深度/GT 都在数据集内。
官方加载器及 DataLoader 检查见 `meta/validation.json`。

历史三分之一标注为
`/mnt/cfs/data/beta_nav/r2r_5670_opos_last_third_20260912/episodes/5670`：
60 帧，opos 从帧 40 开始允许显示，实际可见为 40–55 共 16 帧；0–39 因窗口限制
输出 0，56–59 因终点在视野外输出 0。全部 apos、FM、RVQ 数组和数据类型与原数据一致。
标注目录引用的原始 RGB/深度已在本次校验和视频渲染完成后被移除；用户选择只保留
新标注和视频。当前可查看全部 GT 数组、JSON 与视频，图片路径暂不可用。
先看新目录上一级的
`review/frame_000020.json`、`review/frame_000040.json` 和 `review/frame_000059.json`。
新版视频为 `/mnt/cfs/data/beta_nav/visual/episode_5670_opos_last_third_20260912.mp4`。
生成与逐项比较见 [窗口更新报告](reports/episode_5670_opos_last_third_20260912.json)，
几何与标签验证见 [验证报告](reports/episode_5670_opos_last_third_validation_20260912.json)。

对应原始采集样例（当前已移除）的生成记录为
`/mnt/cfs/data/beta_nav/r2r_one_episode_zero_20260912/episodes/5670`：
60 帧、470×280，apos 可见 53 帧；当时未限制 opos 时间窗口，可见 43 帧，末帧 RVQ STOP 已验证。
生成时样例包含原始 RGB/深度、全部四类 GT、代码/配置及 RVQ 码本副本，约 32 MiB。
它使用 JSON/NPZ 格式，未转换为 LeRobotDataset。

原始数据目录下的 `review/frame_000020.json`（完整 FM-30 和当时可见双点）与
`review/frame_000059.json`（终止帧）。它们以 observation/action 展开单帧数据，
timestamp 等于帧序号。GT 视频位于
`/mnt/cfs/data/beta_nav/visual/episode_5670_zero_gt_20260912.mp4`，60 帧、5 FPS。
命令及检查结果见 [本次生成报告](reports/episode_5670_zero_generation_20260912.json)
和 [完整数据校验](reports/episode_5670_zero_validation_20260912.json)。

此前 2864 等原有数据集目录已被清空，reports/ 中的相关数字示例属于历史记录。
零 token 更新历史见 [pointing_zero_update_20260912.json](reports/pointing_zero_update_20260912.json)。
历史兼容入口 scripts/collect_beta_nav.py 保留，其实现转发到本目录。
