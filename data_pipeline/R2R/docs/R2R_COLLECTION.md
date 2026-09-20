# R2R / RxR 原始轨迹采集

`data_pipeline/R2R/collect_r2r.py` 在独立 Habitat-Sim Python 3.9 环境中回放
专家动作，采集 RGB、深度及逐步位姿，并生成 RVQ-10、FM-30、apos 与 opos。
输出定义见 [GT_CONSTRUCTION.md](GT_CONSTRUCTION.md)，代码流程见
[COLLECTOR_WALKTHROUGH.md](COLLECTOR_WALKTHROUGH.md)。
需要自包含训练数据时，使用 [LeRobot 导出流程](LEROBOT_FORMAT.md#运行与加载)；
完整指令覆盖使用 [R2R 全量入口](FULL_R2R_TRAIN.md) 或 [RxR 入口](RXR_LEROBOT.md)。

在仓库根目录运行，输出必须为新目录：

```bash
# 校验输入并显示配置，不渲染、不写数据
bash data_pipeline/R2R/run.sh /mnt/cfs/data/beta_nav/NEW_RAW_RUN 1 --dry-run

# 采集十条去重后的物理路线
bash data_pipeline/R2R/run.sh /mnt/cfs/data/beta_nav/NEW_RAW_RUN 10

# CPU 只读校验，可另存新报告
.venv/bin/python data_pipeline/R2R/validate_r2r.py \
  /mnt/cfs/data/beta_nav/NEW_RAW_RUN \
  --report data_pipeline/R2R/reports/NEW_RAW_CHECK.json
```

## Collection command parameters

以下是直接调用 `collect_r2r.py` 时的参数。`run.sh OUTPUT [EPISODES] [FLAGS...]`
会填入本机路径与相机参数，其后的 `FLAGS` 原样传给采集器。

| 参数 | 默认值 / 含义 |
|---|---|
| `--output-dir` | 必填，新采集目录，不覆盖已有目录 |
| `--data-root` | 必填，R2R 或 RxR VLN-CE 源数据根目录 |
| `--scenes-dir` | 必填，包含 `mp3d/<scene>/<scene>.glb` 的场景根目录 |
| `--rvq-bundle` | 必填，已发布的 `action_tokenizer` 目录 |
| `--config` | `habitat_server/configs/vlnce_r2r.yaml`，相机与运动配置 |
| `--dataset` | `r2r`；可选 `r2r`、`rxr` |
| `--split` | `train`；可选 `train`、`val_seen`、`val_unseen` |
| `--role` | `guide`；RxR 可选 `guide`、`follower` |
| `--languages` | `all`；RxR 语言筛选，具体语言见 [RXR_LEROBOT.md](RXR_LEROBOT.md) |
| `--episodes` | `10`；按场景/物理路线去重后的数量，`-1` 选全部路线 |
| `--episode-ids-file` | 指定源 episode ID 的 JSON 列表；绕过去重抽样，保留指定指令 |
| `--seed` | `20260911`，抽样随机种子 |
| `--gpu-id` | 直接调用默认 `0`；本机 `run.sh` 默认 `3` |
| `--width` / `--height` | `470` / `280`，当前数据与验证使用此分辨率 |
| `--pointing-config` | `data_pipeline/R2R/configs/pointing.json` |
| `--dry-run` | 只校验输入并打印 manifest，不渲染或写出 |

`run.sh` 支持环境变量 `HABITAT_PYTHON`、`BETA_DATA_ROOT`、`BETA_SCENES_DIR`、
`BETA_RVQ_BUNDLE` 和 `BETA_GPU_ID`，分别覆盖本机 Python、源数据、场景、码本与 GPU。
已安装的附加依赖保留在 `data_pipeline/.venv/`。

`validate_r2r.py` 接受位置参数 `dataset` 及可选 `--report`，自动识别原始采集或
pointing 补标格式。补标、像素网格与配置参数见 [POINTING.md](POINTING.md#命令)。
