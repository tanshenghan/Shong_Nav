# R2R FM + StopHead：八卡闭环评估

本次训练已完成 **3 个 epoch / 3,699 个 global step**。评估默认固定使用最后一个完整检查点：

```text
/mnt/cfs/Models/beta_nav/r2r_fm_stop_h30_prefix_kv_all36_beta1_1p5/checkpoints/step-00003699
```

VLM 保持冻结，FM 和 StopHead 都更新了 3,699 次。完成证据、训练损失图与可复现绘图命令见
[训练报告](../reports/betanav/fm_stop_training_20260928/README.md)。训练损失不能代替本页的闭环导航指标。

## 提交平台任务

申请单机 **8 张 GPU** 后，在平台启动命令中填写以下六 worker 配置：

```bash
exec bash /mnt/pfs/tanshenghan/vln4project/LightNav-0/scripts/eval_betanav_fm_stop_8gpu.sh \
  --num-gpus 8 --workers-per-gpu 6 --shards-per-worker 2 \
  --load-parallelism 2 --cpu-threads 2
```

脚本自动定位仓库，复用独立的 `.venv-train-stop` Python 3.11 推理环境；Habitat 使用独立的
`.venv-habitat` Python 3.9 环境。两者不会混装。脚本前台运行，平台停止任务时会清理本任务创建的
worker 和 Habitat 子进程。

默认每卡启动 6 个常驻模型 worker，共 **48 个 worker**；R2R `val_unseen` 全集按 scene 稳定排序后
划成 **96 个连续分片**。空闲 worker 动态领取下一片，以减少长 episode 导致的尾部等待。
每个 worker 的 VLM、FM 和 StopHead 只加载一次，每个分片使用新的 Habitat server。
全局最多同时加载 2 份模型，限制检查点读取与初始化的并发；加载完毕的 worker 可同时推理。
每次导航决策共用一次 VLM 编码；只有 StopHead 判定继续时才生成 FM 动作。

分片数采用每个 worker 两片：当前 Habitat 将切分余数全部放入最后一片，1,839 条 episode
划为 96 片时是 `95 × 19 + 34`；若仍用每个 worker 四片得到 192 片，则是 `191 × 9 + 120`，
最后一片会显著变长。更细分片不能保证更快完成。

每卡 2–6 个 worker 要求 CUDA 报告的总显存 **至少 80 GiB**。这是原有准入下限，
**不是六份模型及 Habitat 场景可同时放入显存的保证**。当前两 worker 运行截图约占 36% 显存；
简单线性外推六 worker 为约 108%，虽然实际占用未必线性，仍存在真实的 OOM 风险。
六 worker 的显存峰值和吞吐尚未实测，脚本不会自动降低并发。显存不足或需要降低并发时，
可明确改成每卡一个 worker：

```bash
exec bash /mnt/pfs/tanshenghan/vln4project/LightNav-0/scripts/eval_betanav_fm_stop_8gpu.sh \
  --workers-per-gpu 1
```

这会使用 8 个 worker、16 个动态分片。若要复现之前每卡两个 worker、64 分片的配置，
需显式传入 `--workers-per-gpu 2 --shards-per-worker 4`。
提高 worker 数不等于同比提高吞吐，不能提前承诺 GPU 利用率或完成时间。
评估结束后会在汇总中记录实际墙钟耗时和每小时完成 episode 数。

正在运行的旧任务不会因修改脚本而热更新。这条命令启动**新的完整评估**，使用新的输出目录，
不会续跑或合并旧任务的部分结果。若复用同一组 GPU，请先在平台结束旧任务，再启动新任务，
避免两轮模型同时占用显存；旧结果文件保留用于查看。

## 数据与 CPU 预检

默认数据位置：

| 资源 | 路径 |
| --- | --- |
| LightNav VLM backbone | `/mnt/cfs/Models/Light` |
| R2R VLN-CE 数据根目录 | `/mnt/cfs/data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed` |
| Episode 文件 | 数据根目录下 `val_unseen/val_unseen.json.gz` |
| NDTW ground truth | 数据根目录下 `val_unseen/val_unseen_gt.json.gz` |
| MP3D 场景根目录 | `/mnt/cfs/data/habitat/scene_datasets` |
| Habitat 配置 | 仓库内 `habitat_server/configs/vlnce_r2r.yaml` |

截至 2026-09-28，当前开发账户读取上述 CFS Habitat 数据目录得到 `Permission denied`；
平台任务账户的权限与开发账户可能不同。**执行平台任务的账户必须具有这些目录
及其父目录的访问权限，并能读取数据、GT、场景与检查点文件**。脚本遇到权限问题会给出具体路径后退出。

可先在拥有数据权限的任务账户下执行 CPU 预检：

```bash
bash /mnt/pfs/tanshenghan/vln4project/LightNav-0/scripts/eval_betanav_fm_stop_8gpu.sh --check-only
```

预检校验检查点元数据、episode 唯一性、GT 覆盖和有限坐标、scene 几何文件可读性，
并在独立 Python 3.9 解释器中验证 Habitat 导入；**不加载模型权重、不创建模拟器、不要求已分配 GPU**。
它不能验证 CUDA、实际渲染、多模型显存或导航性能。通过时打印 JSON 检查摘要后退出。

若平台使用其他挂载路径，参数直接传给脚本：

```bash
exec bash /mnt/pfs/tanshenghan/vln4project/LightNav-0/scripts/eval_betanav_fm_stop_8gpu.sh \
  --dataset-root /readable/path/R2R_VLNCE_v1-3_preprocessed \
  --scenes-dir /readable/path/scene_datasets \
  --habitat-python /readable/habitat-env/bin/python
```

`--scenes-dir` 是包含数据集 `scene_id` 所引用相对路径的根目录，通常包含 `mp3d/`。
已有完整 Python 3.11 推理环境时，可通过环境变量 `BETA_EVAL_PYTHON` 指定其解释器，跳过
`.venv-train-stop` 的准备步骤。全部参数见 [CONFIGURATION.md](CONFIGURATION.md#queued-fm--stophead-r2r-evaluation)。

## 固定评估协议

| 设置 | 默认值与含义 |
| --- | --- |
| Split | 全部 R2R `val_unseen`；episode 总数从真实压缩数据读取 |
| 模型 | `fm_stop` 检查点；HF backend；`action_output=fm` |
| STOP | `learned`，StopHead score 阈值 `0.5` |
| FM | 10 次积分步，seed `42`；按 scene、episode、决策步确定随机噪声 |
| 动作 horizon | 使用检查点保存的配置；本次为 30 |
| 最大决策步数 | `500`；最后预算步执行显式 STOP，单独记为 `max_steps` |
| 成功距离 | `3.0 m`；报告终点范围采用严格 `< 3.0 m` |
| Oracle STOP | 关闭；合并器拒绝含 oracle STOP 的结果 |
| 视频 | 关闭，减少渲染附加工作和磁盘写入 |

每个 episode 的导航指令由数据集自动提供，无需手动输入。StopHead 提前停止记为
`learned_head`；FM 给出使 Habitat 停止的近零控制量记为 `fm_zero_command`；步数耗尽记为
`max_steps`。即使最终都表现为机器人停住，这三类仍分别统计。

阈值是训练所得 score 的判定门槛，不代表已校准的成功概率。如果改变 `--stop-threshold`、
`--fm-steps` 或预算，输出的 `job.json` 会记录新设置，应将其视为不同评估配置。
`val_unseen` 用于报告泛化结果；不应在反复查看它后把挑选出的最佳阈值称为独立测试结果。

固定同一 checkpoint 比较四个阈值与一次/两次确认，使用
[STOP 八组开发集对比](EVAL_BETANAV_STOP_SWEEP.md) 的独立入口。该入口默认官方 `val_seen`，
一组一张 GPU，每卡六个 worker，共八组；本页原入口继续默认 `val_unseen`。
单组入口现在也接受 `--split val_seen --stop-confirm-steps 2`，并记录开发集独立性审核。

## 输出、进度与失败定位

默认输出目录为仓库下 `outputs/betanav_eval/<UTC时间>_fm_stop_r2r_val_unseen/`。
可用 `--output-dir /path/to/new-directory` 指定；目录必须尚不存在，避免向旧结果追加重复 episode。

| 文件 | 内容 |
| --- | --- |
| `job.json` | 固定检查点路径/step/元数据 SHA-256、GPU 映射、资源记录、参数和全部分片 episode 清单 |
| `status.json` | 全局 `running` / `completed` / `failed` 状态；失败原因或最终汇总位置 |
| `progress.json` | 已完成/总分片数、活跃 worker 和待分配分片数；运行中约每 30 秒更新 |
| `workers/worker_<id>/status.json` | worker 的启动、就绪、运行、分片完成、停止或失败状态 |
| `workers/worker_<id>/worker.log` | 模型加载、导航客户端和 worker 异常日志 |
| `shard_<id>/server.log` | 该分片 Habitat 的启动、模拟与服务日志 |
| `shard_<id>/results.jsonl`、`summary.json` | 各分片逐 episode 原始结果和摘要 |
| `coverage.json` | 全集及分片身份核验；`complete=true` 才表示不缺、不重、不混入其他 episode |
| `results.jsonl` | 核验通过后合并的全部逐 episode 结果，保留分片来源 |
| `summary.json` | 全集 SR、OS、SPL、NDTW、NE、并行配置及实际耗时/吞吐 |
| `stop_diagnostics.json` | STOP 来源、条件成功率、终点距离和 OS–SR 差距分析 |

全局 `completed` 同时要求全部分片完成、协议一致、episode 覆盖完整，以及 STOP 统计有效。
任一 worker 异常退出、模型加载超时、模拟器未就绪或分片超时都会使任务失败并清理子进程。
默认单分片硬超时为 6 小时，可以通过 `--shard-timeout` 调整。遇到问题先看全局 `status.json`，
再看对应 worker 的 `worker.log` 和分片 `server.log`。失败留下的部分结果供诊断，不能作为全集成绩。
脚本不会自动续跑已有输出目录；重跑请使用新的目录。

## 如何读指标

所有报告中的 `*_pct` 均以百分数表示。SR 是环境判定成功的 episode 占全部 episode 的比例；
OS 是途中曾达到目标成功条件的比例；SPL 是兼顾成功与路径效率的平均值。
SR、OS、SPL 的百分数通常在 `[0,100]`。NE、最终距离以米表示。
OS−SR 使用**百分点**，不是相对于 SR 的增长百分比。

`stop_diagnostics.json` 的关键分母如下：

| 统计 | 分子 | 分母 |
| --- | --- | --- |
| 某种 STOP 来源占比 | 以该来源结束的 episode 数 | 所有已核验 episode 数 |
| `learned_head.success_rate_pct` | learned head 停止且环境判成功的 episode 数 | 所有 learned head 停止的 episode 数 |
| `learned_head.goal_range_rate_pct` | learned head 停止且终点距离 `<3 m` 的 episode 数 | 所有 learned head 停止的 episode 数 |
| `forced_stop` 占比 | 预算步强制 STOP 的 episode 数 | 所有已核验 episode 数 |
| `oracle_success_without_success` 条件比例 | OS 为真、最终 success 为假的 episode 数 | OS 为真的 episode 数 |

各 STOP 来源的统计位于 `by_stop_source` 下；距离和步数另列 count、mean、median、min、max。
没有相关 episode 时，条件比例为 JSON `null`，不记成 0%。字段缺失、非有限导航指标、未知
STOP 来源拼写或不一致的强制停止标记会被拒绝，避免生成看似完整的指标。

这些是 **episode 级闭环诊断**，不能称为逐帧 STOP 分类 precision/recall。
例如“OS 为真但最终失败”可以提示停止问题，也可能涉及离开目标后的路径控制或其他原因；
它不能单独证明 StopHead 在某一帧漏检。训练 loss 降低同样不能代替这些导航结果。
