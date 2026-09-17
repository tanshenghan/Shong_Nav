# 完整 R2R train → LeRobotDataset

正式入口是 `data_pipeline/run_r2r_train.sh`，调度逻辑在
`build_r2r_train.py`，合并逻辑在 `merge_lerobot.py`。沿用
[GT 定义](GT_CONSTRUCTION.md)与 [LeRobot 字段](LEROBOT_FORMAT.md)。

已完成的数据集：`/mnt/cfs/data/beta_nav/lerobot_r2r_train_470x280_20260912`。
全部 10,819 条 episode / 631,244 帧已生成并校验；4,853 个 Parquet 文件的数值读回、
27,398 个来源文件、全部 source ID/指令/全局帧索引和官方加载器/DataLoader 检查通过。
人工查看先读 `meta/summary.json`、`meta/info.json` 和 `meta/full_train_validation.json`。
`meta/beta_nav.json` 包含全部逐帧哈希，体积较大，建议通过 Python 按需读取。
最终 95 项 pipeline CPU 测试、ruff 和入口 shell 语法检查通过。

## 数据范围

本机 `R2R_VLNCE_v1-3_preprocessed/train/train.json.gz` 和 `train_gt.json.gz`
包含 **10,819 条指令 episode、3,603 条物理路线、61 个场景、631,244 帧**。
这是本地预处理 R2R-VLNCE train 的完整覆盖；不是经典离散 R2R 文件的数量，
不混入 EnvDrop 增广、RxR、val 或 test。

每条路线一般对应三条指令，10 条路线对应四条。训练时每条指令都作为独立
LeRobot episode，保留原始 source episode_id 与逐字指令，不丢弃重复文本。
其中有 10,815 个不同的指令文本：LeRobot 的 task 表按文本共享索引，
但 episode 表仍保留全部 10,819 条样本。
仅当起始位姿、目标和完整 expert GT 都相同时才复用一次渲染。
因此实际渲染 210,194 帧，按指令展开后写入 631,244 帧。

原来的 `run.sh ... -1` 表示“每条路线取一条指令”，适合采样，不代表完整训练集。
全量数据请使用下面的新入口。

## 运行、恢复和状态

```bash
bash data_pipeline/run_r2r_train.sh \
  /mnt/cfs/data/beta_nav/lerobot_r2r_train_470x280_20260912 \
  --workers 16 --gpus 1 3

# 只检查所有源数据、GT、场景/navmesh 与固定生成定义
bash data_pipeline/run_r2r_train.sh /mnt/cfs/data/beta_nav/NEW_TRAIN --dry-run

# 先生成一批并完整校验，保留全量计划；之后去掉该参数继续
bash data_pipeline/run_r2r_train.sh /mnt/cfs/data/beta_nav/NEW_TRAIN \
  --max-new-shards 1 --workers 1 --gpus 1
```

中断后重复同一命令即可恢复；允许改变 workers、gpus、重试次数和试跑上限。
完整计划绑定源数据、相机、pointing、RVQ 和 GT 代码哈希。改变生成定义必须
使用新输出目录，避免在同一训练集中混入不同版本。

本次全量数据暴露出 MP3D 户外扫描缺口：极少数原始朝向没有任何表面，
RGB 全黑、深度全 0，转向后恢复。处理方式是保留真实帧与时间轴、保留全 false
深度 mask；apos 没有可见候选时仍为 unknown/token=0/loss_valid=false。
FM/RVQ 继续由实际回放位姿构造，opos 仍遵守后半段与实际可见性规则。
修复仅移除旧代码对“整帧深度无效”的拒绝，不改变已有可用样本的数值。
续跑器只对这一处精确的两行删除允许兼容续跑，并保存旧/新代码和哈希到
`definition_updates/`，最终一并保存在数据集 provenance；其他定义变化仍拒绝混用。

最终目录只在全部批次与合并校验通过后出现。生产中的路径是输出目录加 `.work`：

| 路径 | 含义 |
|---|---|
| `plan.json` | 全部路线/指令的不可变计划、631,244 帧的覆盖预期、源文件哈希 |
| `status.json` | 已完成并通过校验的批次数、episode 数、帧数、错误列表 |
| `logs/` | 每批独立日志；本次启动另有 `full_train.log` |
| `shards/00000/` | 通过全部校验的独立标准 LeRobotDataset，可单独检查 |
| `scratch/` | 当前批次临时 RGB/深度与导出；成功后删除 |
| `failed/` | 失败尝试的现场，保留便于定位；不会进入正式数据集 |
| `cache/` | 本任务的 Hugging Face/Arrow 缓存；成功批次的缓存会清理 |
| `merge/` | 合并后的每个 Parquet 文件哈希和续跑记录 |
| `assembled/` | 正在合并的最终数据集；全部验证后原子改名到最终目录 |

某批失败时自动重试一次；仍失败会记录并继续处理其他批次，最终退出失败，
不会把缺少 episode 的数据发布为完整 train。再次运行时只重做未完成批次。
同一工作目录有进程锁，阻止两个调度器同时写入。

## 参数

| 参数 | 默认值 / 用途 |
|---|---|
| wrapper 第一个位置参数 / `--output-dir` | 最终 LeRobotDataset 新目录 |
| `--workers` | 4；并发采集/导出/验证批次数，按可用 CPU、RAM、GPU 调整 |
| `--gpus` | `1 3`；物理 GPU 编号列表，循环分配；不使用模型推理 |
| `--routes-per-shard` | 8；同一场景内分批，当前 train 共 480 批 |
| `--retries` | 1；每批失败后的额外重试次数 |
| `--max-new-shards` | 无上限；只用于小规模试跑，不修改完整计划 |
| `--dry-run` | 只检查和打印计划，不创建输出/启动 Habitat |
| `--repo-id` | `beta-nav/r2r-train`；本地标识，不上传 Hugging Face |
| `--data-root` | `/mnt/cfs/data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed` |
| `--scenes-dir` | `/mnt/cfs/data/habitat/scene_datasets` |
| `--rvq-bundle` | `/mnt/cfs/Models/Light/action_tokenizer` |
| `--pointing-config` | `data_pipeline/configs/pointing.json`，全量入口要求 last_half |

底层 `collect_r2r.py --episode-ids-file FILE` 接受 JSON ID 列表，覆盖随机路线采样，
精确采集这些 source episode；调度器使用它采集每批的代表路线。
`--worker-plan`、`--shard-id`、`--gpu-id` 是调度器内部进程参数。

## 监督与存储

- 新 LightNav Habitat RGB/深度/位姿，470×280、HFOV 120°、相机高 0.88 m。
- 合成时间 `fps=1`、`timestamp=frame_index`，`physical_dt_s=null`。
- 每帧含完整 FM-30 真实路点与 SE(2) 增量、RVQ-10、apos/opos 的坐标、token 与独立 mask。
- opos 同时满足 `t >= floor(N/2)` **并且当前帧投影和深度可见**才有空间点。
- 指向缺失 token 为 0，unknown 与确认不可见通过 state/mask 区分。
- 末帧 RVQ 为已验证的显式 STOP；FM 无未来部分 masked；不导出离散转向/前进类别。
- RGB 为 Parquet 内无损 PNG，深度为原始 float32 米值与 bool mask；不依赖 capture。
- 官方 LeRobot 0.4.4 / v3.0 负责写入，标准 features 共 49 列。

每批都在删除临时 capture 后逐帧用官方加载器验证 RGB/深度哈希、30 组 GT 数组、
重新计算的 FM/pointing 几何与末帧 STOP，并运行 DataLoader。
合并只调整全局 `index`、`episode_index` 和 `task_index`，其他值逐文件读回核对。
各 episode 和全局统计同步更新索引；无效/未观察的 GT 不进入运动归一化统计。
最后检查全部 source ID、指令、场景、帧数和全局索引，再用官方加载器抽查合并后的
首部、中部、尾部 episode 和 DataLoader。详细证明在 `meta/full_train_validation.json`。

生产期间发现并修复了原几何校验的 ±π 等价朝向边界问题：仅将角度误差改为周期角差，
保存的 FM 路点和增量未改变。相关测试在 `tests/test_full_train.py`。

## 训练读取

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader

dataset = LeRobotDataset(
    repo_id="beta-nav/r2r-train",
    root="/mnt/cfs/data/beta_nav/lerobot_r2r_train_470x280_20260912",
    video_backend="pyav",
)
batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)))
rgb = batch["observation.images.rgb"]          # B,3,280,470; float32 [0,1]
fm = batch["action.fm.waypoints_ego"]           # B,30,3; 米/弧度
fm_valid = batch["action.fm.valid_mask"]        # B,30
rvq_codes = batch["action.rvq.codes"]           # B,3
apos = batch["action.apos.token_id"]
opos = batch["action.opos.token_id"]
instructions = batch["task"]
```

首次完整读取会生成 Arrow 缓存，请把 `HF_DATASETS_CACHE` 放在容量足够的数据盘，
不要放入 `/tmp`。完整帧数据预计数百 GB，生产批次和最终合并暂时并存，预留约 1 TB
以上空间更合适；精确体积以完成报告为准。数据格式可由官方加载器直接读取，
BetaModel 训练代码仍需按 `action.*` 字段分别构造 CE/FM loss，并遵守独立 mask；
标准单向量 `action` policy 不能直接替代这里的多类监督。
