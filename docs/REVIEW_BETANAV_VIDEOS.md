# 录制四个 R2R 闭环样例视频

`scripts/record_betanav_four_samples.sh` 从已完成的 R2R `val_unseen` 评估中，
按固定随机种子抽取 **2 条成功、2 条失败**，使用原检查点与评估参数重新执行闭环导航并录制视频。
原始全量评估没有保存逐帧图像，因此这些视频是**新录制的复跑结果**，不是原评估帧的回放。

## 单卡平台命令

申请 **1 张 GPU**，使用有权读取原模型、R2R 数据和 MP3D 场景的任务账户，运行：

```bash
exec bash /mnt/pfs/tanshenghan/vln4project/LightNav-0/scripts/record_betanav_four_samples.sh
```

脚本只使用平台 `CUDA_VISIBLE_DEVICES` 中的第一张 GPU，支持 GPU UUID，复用
`.venv-train-stop` 推理环境和独立的 Habitat 环境。模型只加载一次，四条 episode 顺序执行。
已有可用推理环境时可用 `BETA_EVAL_PYTHON` 指定 Python 3.11 解释器。

截至准备文档时，开发账户 uid 1000 无法读取所需模型及场景，尚未生成真实 MP4。
准备清单和 CPU 检查通过不代表视频已生成；真实录制需要上述平台任务访问权限及 GPU。

## 固定抽样

默认来源为已完成、覆盖全部 1,839 条 episode 的目录：

```text
/mnt/pfs/tanshenghan/vln4project/LightNav-0/outputs/betanav_eval/20260928T012647_241235Z_fm_stop_r2r_unseen
```

默认 `--selection-seed 20260928` 对该来源抽得：

| 来源结果 | Habitat episode ID | 场景 |
| --- | --- | --- |
| 成功 1 | `938` | `oLBMNvg9in8` |
| 成功 2 | `1091` | `X7HyMhZNoso` |
| 失败 1 | `1060` | `QUCTc6BB5sX` |
| 失败 2 | `522` | `x8F5xyUWy9e` |

工具要求来源状态为 `completed`，核对 episode 不缺不重，并将来源 `job.json`、
`results.jsonl` 的 SHA-256 和抽样原始记录写入 `selection.json`。换来源或抽样种子会得到不同清单。

只冻结抽样清单、无需访问模型/场景或使用 GPU 的命令为：

```bash
bash /mnt/pfs/tanshenghan/vln4project/LightNav-0/scripts/record_betanav_four_samples.sh \
  --prepare-only
```

该命令仅生成 `selection.json` 和 `status.json`，状态为 `selected`，不会生成视频。
之后正式录制应使用新输出目录；相同来源和种子会重复选中同四条样例。工具不续写已经存在的目录。

## 视频内容与复跑核验

每个导航决策保存一帧，并额外保存终帧。画面包含完整指令横幅、预测轨迹、目标距离（DTG）、
STOP score 和阈值等诊断信息。默认 **5 FPS** 表示录制帧的播放速率，
不代表机器人或 Habitat 的物理实时运动速度；视频帧数应为 `steps + 1`。

复跑保留原检查点、FM 随机种子、STOP 阈值及步数预算，保留原始 scene 路径写法，
以维持按 scene、episode、决策步生成 FM 噪声的身份。工具会核验当前数据顺序、检查点元数据
和 NDTW GT，再比较复跑的 scene/episode 身份、指令以及 success、OS、步数、STOP 来源、
终点距离、最小距离、SPL、NDTW、STOP score。浮点指标采用 `rel_tol=1e-5`、`abs_tol=1e-4`。

若身份或指令不符，任务失败。若数值或结果与来源不同，仍保留视频与差异报告，状态为
`completed_with_differences`，进程退出码为 **2**；不会重抽样例来替换不一致结果。
MP4 文件名中的 `success` / `failure` 反映**这次复跑**的实际结果，来源标签保留在清单及对比文件中。
仅在所有核验一致时，最终状态才为 `completed`，退出码为 0。

## 输出

默认目录为仓库下 `outputs/betanav_videos/<UTC时间>_four_samples/`：

| 路径 | 内容 |
| --- | --- |
| `selection.json` | 固定抽样、来源哈希和四条原始记录 |
| `replay_job.json` | 原评估配置、资源检查、Habitat 环境、端口和视频帧率 |
| `videos/*.mp4` | 四个新录制视频，文件名包含场景、episode ID 和复跑结果 |
| `comparison.json` | 每条来源与复跑指标对比、视频路径、解码帧数及尺寸等 |
| `status.json` | `selected` / `running` / `completed` / `completed_with_differences` / `failed` |
| `episode_*_manifest.json` | 每条复跑的精确 episode 身份 |
| `shard_*/results.jsonl`、`summary.json` | 单条复跑的原始结果与摘要 |
| `shard_*/server.log` | Habitat 日志 |
| `shard_*/records/` | 逐步原始帧和记录，供进一步诊断或渲染 |

工具会完整解码生成的视频，检查帧数和帧率后再复制至 `videos/`。缺帧或无法解码会使任务失败，
不会把缺失的视频当成成功输出。全部参数见
[CONFIGURATION.md](CONFIGURATION.md#four-sample-fm--stophead-videos)。
