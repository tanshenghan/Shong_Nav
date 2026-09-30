# 观测渲染与仿真回放

新 A* 连续曲线样例使用 `render_parallel` 渲染新位姿，再用 `compare_planned`
生成 `planned.mp4` 和 `before_after.mp4`；固定 15 FPS，顶部完整指令，未来 30 点轨迹带。
该入口与保留官方逐步对应关系的 `compare` 分开，详见
[连续曲线的命令与播放约定](../docs/PLANNED_ROUTES.md)。

当前提供 Blender CPU、Isaac 独立相机渲染和原始/平滑视频比较入口。
最新 `kujiale_0172_51_0` 的 23 帧原始与 111 帧平滑底图、共享 LightNav 风格三段视频已验证，
完整解码通过。完成范围限于这些样例的 CPU 视觉预览，Isaac GPU 渲染仍未实测。
下面的机器人控制、实际位姿、接触与深度采集仍待实现。
历史 `kujiale_0018_2_2` 的完整 Instructions 横幅、20° 下俯相机和未来 30 个平滑位姿地面标注版已完成：
23 个原始/111 个平滑视图全部生成，三段视频完整解码通过。新版推荐链接与历史版本均保留于后文。

历史直行样例 `kujiale_0018_2_2` 有 23 个原始位姿、111 个平滑采样位姿，
合成参考时长 30.078125 秒。初选 `kujiale_0195_4_2` 未通过半径 0.30 m 的扫掠检查；
后续筛选及地图校准证据保存在 [几何报告](../reports/preview_geometry_20260919.json)。
最新 `kujiale_0172_51_0` 包含 4 次 ±15° 原地转向；两例都不能展示或验证连续绕角弧线。

## Blender CPU 预览

[blender_render.py](blender_render.py) 在独立的 `data_pipeline/.venv/blender-preview/`
Python 3.11 环境中运行，依赖版本固定于 [requirements-preview-cpu.txt](../requirements-preview-cpu.txt)。
新建环境时从仓库根目录运行：

```bash
python3.11 -m venv data_pipeline/.venv/blender-preview
data_pipeline/.venv/blender-preview/bin/python -m pip install \
  -r data_pipeline/vlnverse/requirements-preview-cpu.txt
```

此环境仅用于静态场景预览，独立于训练、推理和 Isaac 环境。
它预览真实 USD 几何，并通过 [usd_materials.py](usd_materials.py) 近似转换部分 MDL
漫反射颜色、纹理与 UV 变换、粗糙度、金属和玻璃。
初次试跑发现原生 USD 导入丢失父变换，不能把“导入了网格”视为坐标正确。
现已接入 [usd_scene.py](usd_scene.py)，由 pxr 读取静态网格、精确世界矩阵、UV 和全路径材质绑定；
本例世界变换已通过核对，导航场景首帧已验证；其他场景仍须检查世界范围及代表性顶点对齐。
未实现完整 MDL 求值；法线、发光、视角相关颜色和部分参数纹理等不保证一致。
原场景灯光不导入，改用两版共用的环境光与相机补光，因此外观不是官方 Isaac RTX 的精确复现。

**场景入口使用 `start_result_navigation.usd`。** 官方 `evaluator/utils/common.py` 的
`load_kujiale_scene_usd` 优先加载该文件，缺失时才回退到原始场景。本例的原始
`kujiale_0018.usda` 保留了导航版删除的门体，会遮挡首帧视线；不能用它替代已核对的导航资产。
两版必须使用同一入口，实际路径写入各自 manifest。

从仓库根目录运行，分别为两版创建新的输出目录。以下生成带地面路径的审阅版本：

```bash
data_pipeline/.venv/blender-preview/bin/python \
  -m data_pipeline.vlnverse.replay.blender_render \
  --scene-usd /mnt/cfs/jiayilin/vlnverse_eccv_2026/assets/vlnverse_scenes/kujiale_0018/start_result_navigation.usd \
  --poses-npz /PATH/TO/RETIMED/source_poses.npz \
  --output-dir /mnt/cfs/data/beta_nav/vlnverse/NEW_PREVIEW/render_instruction_path/original \
  --camera-pitch-down-deg 20 --future-steps 0

data_pipeline/.venv/blender-preview/bin/python \
  -m data_pipeline.vlnverse.replay.blender_render \
  --scene-usd /mnt/cfs/jiayilin/vlnverse_eccv_2026/assets/vlnverse_scenes/kujiale_0018/start_result_navigation.usd \
  --poses-npz /PATH/TO/RETIMED/trajectory.npz \
  --output-dir /mnt/cfs/data/beta_nav/vlnverse/NEW_PREVIEW/render_instruction_path/smoothed \
  --camera-pitch-down-deg 20 --future-steps 30
```

| 参数 | 默认值 | 约定 |
|---|---|---|
| `--scene-usd PATH` | 必需 | 优先使用官方导航场景 `start_result_navigation.usd`，连同引用资产树保持可读 |
| `--poses-npz PATH` | 必需 | `positions_xyz[N,3]` 与 `yaw_rad[N]` |
| `--output-dir PATH` | 必需 | 新目录，拒绝覆盖 |
| `--camera-height-m` | 1.25 | 相机 Z = 输入 Z + 此高度，米 |
| `--camera-pitch-down-deg` | 0 | 向下俯角，度；上述 `scene` 示例两版为 20，`lightnav` 底图须为 0 |
| `--future-steps` | 0 | 地面标注后续位姿数，0 关闭；上述 `scene` 示例原始 0/平滑 30，`lightnav` 底图两版均须为 0 |
| `--hfov-deg` | 90 | 水平视角，度 |
| `--width` / `--height` | 480 / 360 | 输出 RGB 尺寸 |
| `--samples` | 8 | Cycles 每像素样本数 |
| `--threads` | 16 | Cycles CPU 线程数 |
| `--max-frames` | 不限制 | 可选首帧/短序列诊断数量，记录在 manifest；不是完整采集 |

两版使用完全相同的场景、相机、采样质量和灯光。
相机无前向偏移，世界 +Z 向上；默认俯角 0° 时 yaw=0 沿世界 +X。
`kujiale_0018_2_2` 的场景标注版两侧都下俯 20°，便于观察地面；共享 LightNav 风格使用水平相机。
高度与 HFOV 是选定的预览参数，不是恢复出的官方传感器外参；本次输入 Z 为 0，
所以实际相机高 1.25 m。渲染不加载受控机器人、不执行物理轨迹跟踪，也不输出训练标签。

`--future-steps 30` 标注的是当前平滑位姿之后的 `i+1..i+30`，**不是 30 个原始动作**，
也不是固定秒级 horizon。靠近终点只使用真实剩余位姿，不重复末位姿补足 30 点。
路径用向场景落地的射线确定地面位置；首次命中须靠近源 Z、法线向上，并满足以下条件之一：

- 属于明确的 USD `floor` 组。
- 属于 `wall` 组中的水平封口：整个网格厚度及其相对源 Z 的偏差均不超过 2 mm，
  且单独只查询 `floor` 网格的射线证实其下方 8 cm 以内有地板支撑。

第二条处理导航场景中归入 `wall` 的薄水平门洞封口；家具、竖墙或没有独立地板证据的表面仍拒绝。
青色连线与绿色圆点以 3D 几何渲染并受场景遮挡；视野外、近裁剪面以内
或被场景挡住的部分自然不可见，不把整条路径强行画到图像上。
这些标注用于审阅，带路径的 PNG 和带横幅的视频不能作为干净训练 RGB。

输出为目录根下的 `000000.png` 等及最后写入的 `render_manifest.json`。
manifest 记录输入位姿哈希、相机、渲染参数、材质转换审计、帧数、诊断上限和
`physics_replayed=false`、`isaac_equivalent_render=false`。
下俯角记录于 `camera.pitch_down_deg`；启用路径时，`future_path` 记录实际地面高度、射线命中和逐帧未来位姿索引。
材质审计记录未支持或未成功绑定的材质；不要把近似替代说成恢复全部原始材质。
接入检查拒绝空网格、无效位姿和尺寸错误/近乎空白的图像。完整比较必须渲染全部源位姿，
不能用 `--max-frames 1` 的诊断目录冒充完整结果。

## 同一时间轴的原始/平滑视频

[compare.py](compare.py) 需要 NumPy、Pillow 和 PyAV。当前输入约定是上述 Blender 的
编号 PNG 与 `render_manifest.json`；不能直接把 Isaac 的另一套目录/manifest 作为输入。
默认两侧都是重新渲染的预览。可加 `--official-rgb-npy PATH` 将左侧替换为同一官方 episode 发布的 RGB，
右侧继续使用新位姿的 Blender 预览；官方图像不由本地重渲染冒充。
视频最上方完整显示 `retimed/manifest.json` 中与官方 episode 元数据关联的 `instruction`。
并排版使用全宽 `Instructions` 横幅，两个独立视频也有横幅；按文本换行增加高度，不截断指令。
默认 `--overlay-style scene` 使用上述场景标注，右侧底部用青色标明实际剩余的
`next count / 30 smooth poses`，末帧为 0/30。也可使用下述共享 LightNav 可视化风格。

推荐以下 15 FPS 位姿播放命令：右侧每个新位姿只显示一帧，不补重复帧；左侧保持前一个
对应源位姿，直到下一个源锚点。本例左侧前 22 个源位姿各显示 5 帧，末位姿显示 1 帧。

```bash
python -m data_pipeline.vlnverse.replay.compare \
  --retimed-dir /PATH/TO/RETIMED \
  --original-images /mnt/cfs/data/beta_nav/vlnverse/NEW_PREVIEW/render_instruction_path/original \
  --smooth-images /mnt/cfs/data/beta_nav/vlnverse/NEW_PREVIEW/render_instruction_path/smoothed \
  --output-dir /mnt/cfs/data/beta_nav/vlnverse/NEW_PREVIEW/review_instruction_path_15fps \
  --playback-mode pose_frames --fps 15
```

前四个参数必需，`--output-dir` 必须新建；`--fps` 默认为 20，仅控制视频容器频率。
`--retimed-dir` 应有 `manifest.json`、`source_poses.npz`、`trajectory.npz`，
并使用保留源锚点的 `per_source_step` 模式。

| 比较入口参数 | 默认值 | 约定 |
|---|---|---|
| `--overlay-style` | `scene` | `scene` 保留输入图像中的场景标注；`lightnav` 复用共享蓝色轨迹带与 HUD |
| `--future-steps` | 30 | 仅用于 `lightnav` 轨迹带的未来平滑位姿数，须为正整数；不是 Blender 入口的同名 3D 标注开关 |
| `--official-rgb-npy` | 不启用 | 同一源 Parquet episode 根目录的 `videos/chunk-000/observation.images.rgb/rgb.npy`；替换左侧为官方 RGB |

| `--playback-mode` | 播放约定 |
|---|---|
| `pose_frames` | 每个平滑位姿恰好显示一帧；`--fps 15` 时每个位姿占 1/15 秒，本例 111 帧、7.4 秒；左侧按源锚点同步并保持上一源位姿 |
| `reference_time`（默认） | 按 `timestamp_s` 安排图像，保持到下一位姿的合成时刻；本例 `--fps 20` 输出 603 帧、30.15 秒 |

`pose_frames` 只改视频播放时钟，**不修改 `trajectory.npz` 中非均匀的 `timestamp_s`**。
固定每个位姿 1/15 秒不保留参考运动的 `dt`，不能将播放表现解释为仍满足原速度、加速度或 jerk 约束。
两种模式都不会用光流或图像插帧，不增加训练观测，也不恢复官方真实动作时间。

`reference_time` 中，原始 GT 每个位姿对齐到平滑曲线经过该源锚点的合成时间，
平滑位姿使用 `timestamp_s`；不能把默认每源间隔 6 点直接当成 30 Hz。
两种模式的原始和平滑版分别共享同一总时长和视频帧数。
默认“原始 GT 视频”表示官方原始位姿的重新渲染；启用 `--official-rgb-npy` 后左侧才是官方 RGB。
两种来源的比较视频都没有恢复官方真实动作时间。

输出 `original_gt.mp4`、`smoothed_{K}poses.mp4`、`side_by_side.mp4`、少量预览 JPG，
最后写 `comparison_manifest.json`。`K` 来自 `poses_per_step`，默认文件名为 `smoothed_6poses.mp4`。
manifest 保存 `playback_mode`、`reference_duration_s`、`video_duration_s` 及每视频帧对应的
原始/平滑索引和参考时间，并记录完整指令、横幅布局与平滑路径标注配置。
`pose_frames` 的播放时长由位姿数/FPS 决定；`reference_time` 的
视频时长包含终点显示和容器帧周期量化，与精确参考时长可能略有差异。

### 官方 RGB 对照平滑预览

`--official-rgb-npy` 的路径必须与 retimed manifest 指向的源 Parquet 位于同一 episode 根目录；
例如 Parquet 为 `EPISODE/data/chunk-000/episode_000000.parquet`，RGB 必须为
`EPISODE/videos/chunk-000/observation.images.rgb/rgb.npy`。
读取器核验路径关联、`uint8 [N,H,W,3]` 和 `N=原始源位姿数`，并记录 RGB 文件 SHA256。
匹配帧数不足以证明来自同一任务，因此不能传入另一 episode 的同尺寸数组。

在现有比较命令上增加该参数，例如：

```bash
PYTHONPATH=src data_pipeline/.venv/lerobot/bin/python \
  -m data_pipeline.vlnverse.replay.compare \
  --retimed-dir /PATH/TO/RETIMED \
  --original-images /PATH/TO/RENDER/original \
  --smooth-images /PATH/TO/RENDER/smoothed \
  --official-rgb-npy /PATH/TO/EPISODE/videos/chunk-000/observation.images.rgb/rgb.npy \
  --output-dir /mnt/cfs/data/beta_nav/vlnverse/NEW_PREVIEW/review_official_vs_smoothed \
  --overlay-style lightnav --future-steps 30 \
  --playback-mode pose_frames --fps 15
```

当前仍要求两个 Blender 目录及其 manifest，用于验证源/派生位姿哈希、预览相机和渲染配置；
通过校验后，左侧画面替换为官方 RGB。`lightnav` 仍要求这两套预览输入为水平相机、无 3D 路径底图。
官方 RGB 保持宽高比缩放到显示面板，居中留黑边，不裁切；没有时间插帧。
左侧按源锚点保持官方帧，右侧逐个平滑位姿播放；标题和页脚明确区分 `OFFICIAL RGB` 与 Blender 预览。

此模式输出 `official_rgb.mp4`、`smoothed_{K}poses.mp4`、`official_vs_smoothed.mp4`；
不传该参数时保留原文件名。manifest 记录 `official_rgb` 的来源/哈希/尺寸、`left_view_source` 和
`matched_left_right_camera=false`，其中原有 `camera` 字段描述 Blender 预览配置，不代表官方标定。
**两侧相机与渲染器不等价。** 材质、HDR 环境、视角差异不能被当作轨迹错误，也不能声称两侧 RGB 应逐像素一致。
官方图像保留原始视觉证据；平滑新视图仍只是近似渲染，完整 Instructions 和视觉标注仍仅用于审阅。

### 可选：共享 LightNav 轨迹带与 HUD

成片的 `original_gt.mp4` 使用官方位姿重新渲染，不是官方原始 RGB。
后续 [画面与选样审计](../reports/preview_fidelity_audit_20260919.json) 发现窗外环境、
瓷砖等指令地标在 Blender 近似材质中缺失，两条已生成路线也明显偏短、偏直。
当前样例只验证运动与绘制逻辑，不能作为原始数据难度或视觉语义一致性的证明。

`--overlay-style lightnav` 通过 [viz_style.py](viz_style.py) 直接复用
[lightnav.viz.render](../../../src/lightnav/viz/render.py) 的 `draw_traj_ribbon` 与 `draw_scifi_hud`。
已有 `data_pipeline/.venv/lerobot/` 环境包含此路径所需的 OpenCV、imageio 等依赖；
从仓库根目录设置 `PYTHONPATH=src`，不修改共享渲染器源码。

先为两版渲染新的水平相机干净 RGB：在上述 Blender 命令中均设置
`--camera-pitch-down-deg 0 --future-steps 0`，分别写入新的 `render_lightnav/original`
和 `render_lightnav/smoothed` 目录。使用所选 episode 对应的 `start_result_navigation.usd`。
比较入口拒绝 20° 下俯图像或已绘制 3D 地面路径的底图，避免叠加两套投影。

```bash
PYTHONPATH=src data_pipeline/.venv/lerobot/bin/python \
  -m data_pipeline.vlnverse.replay.compare \
  --retimed-dir /PATH/TO/RETIMED \
  --original-images /mnt/cfs/data/beta_nav/vlnverse/NEW_PREVIEW/render_lightnav/original \
  --smooth-images /mnt/cfs/data/beta_nav/vlnverse/NEW_PREVIEW/render_lightnav/smoothed \
  --output-dir /mnt/cfs/data/beta_nav/vlnverse/NEW_PREVIEW/review_lightnav_15fps \
  --overlay-style lightnav --future-steps 30 \
  --playback-mode pose_frames --fps 15
```

原始侧显示 HUD、不画轨迹带，速度留空；平滑侧显示未来最多 30 个平滑位姿的渐变蓝色宽带。
未来世界位置只转换一次到当前机器人局部坐标，不累计求和；投影输入第三列置零，
避免动作投影器的纯转向启发式改变 GT 位置，真实 yaw 仍保存在轨迹 NPZ。
右侧 HUD 速度来自保存的合成参考解析速度和角速度，转换到当前机体坐标，
不会按默认 0.1 秒或视频 FPS 从位姿差推算。HUD 的 step/FPS 是视频帧号/播放帧率，
不是官方传感器采样率；固定 15 FPS 播放仍不保留参考运动的 `dt`。
完整官方 Instructions 保留在最顶部横幅，HUD 内仅显示左右 GT 标签。

共享轨迹带使用 `forward_offset=None` 的自动视觉前移：当前 480×360、相机高 1.25 m、
HFOV 90° 时约前移 1.667 m，宽带参数为画面宽度的 0.25。
**这是视觉示意，不是精确地面落点，也不进行深度遮挡检查。** 它与 `scene` 模式中实际落地、
受场景遮挡的 3D 标注含义不同，两者都只用于审阅，不作为训练观测。
`comparison_manifest.json` 的 `shared_viz` 记录共享函数、源文件哈希、坐标约定、视觉前移和 HUD 时钟来源。
此风格的新样例为 `kujiale_0172_51_0`：23 个源位姿转为 111 点，包含 4 次 ±15° 原地转向，
合成参考时长 34.43516469448 秒。完整视频已生成并通过验证；
生产图像使用水平相机和无 3D 标注底图，分别保存在新运行目录的 `render/original`、`render/smoothed`。
三个视频均为 111 帧、15 FPS、7.4 秒，推荐：

- [LightNav side_by_side.mp4](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0172_51_0/review_lightnav_15fps/side_by_side.mp4)
- [LightNav smoothed_6poses.mp4](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0172_51_0/review_lightnav_15fps/smoothed_6poses.mp4)
- [LightNav original_gt.mp4](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0172_51_0/review_lightnav_15fps/original_gt.mp4)

三段视频完整解码通过，独立版 480×686、并排版 960×626；尺寸包含完整指令横幅。
23 个原始底图和 111 个平滑底图各自全部唯一，23 个源锚点两版 RGB 差为 0，
未来窗口逐帧验证、完整指令行还原均通过。相关 341 项测试与 Ruff 通过；
记录见 [LightNav 风格预览报告](../reports/preview_lightnav_viz_0172_20260919.json)。

历史 `kujiale_0018_2_2`：完整 Instructions 与未来 30 个平滑位姿地面标注，每段 111 帧、15 FPS、7.4 秒：

- [original_gt.mp4](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review_instruction_path_15fps/original_gt.mp4)
- [smoothed_6poses.mp4](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review_instruction_path_15fps/smoothed_6poses.mp4)
- [side_by_side.mp4](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review_instruction_path_15fps/side_by_side.mp4)

三个视频完整解码通过；独立版 480×710，并排版 960×656，尺寸包含完整指令横幅。
指令与官方元数据逐字核对通过，111 帧的未来位姿索引全部正确；末帧未来点数为 0，
其渲染图像与同位姿未标注原始图像逐像素一致（MAE=0）。111 个落地点全部通过校验，
其中 5 个水平门洞封口有独立地板支撑证据。记录见
[指令与路径预览报告](../reports/preview_instruction_path_20260919.json)。

此前已完成的水平相机、无地面标注 15 FPS 版本，每段 111 帧、7.4 秒：

- [original_gt.mp4](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review_15fps/original_gt.mp4)
- [smoothed_6poses.mp4](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review_15fps/smoothed_6poses.mp4)
- [side_by_side.mp4](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review_15fps/side_by_side.mp4)

三段视频完整解码通过，平滑索引严格为 `0..110`、无重复帧；左侧索引为 `min(j//5,22)`。
生成与解码检查见 [15 FPS 播放报告](../reports/preview_playback_15fps_20260919.json)。
原先按合成参考时间播放的历史版本保留于下列路径，均为 603 帧、20 FPS、30.15 秒：

- [original_gt.mp4](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review/original_gt.mp4)
- [smoothed_6poses.mp4](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review/smoothed_6poses.mp4)
- [side_by_side.mp4](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review/side_by_side.mp4)

上述历史视频完整解码通过；未加路径标注的原始第 `i` 帧与平滑第 `5*i` 帧逐像素一致，
23 个源锚点最大 MAE 为 0。此像素一致性结论不适用于右侧加入地面路径的新版本。
生成记录与检查边界见 [预览生成报告](../reports/preview_render_20260919.json)。

## 独立 Isaac 相机预览

[isaac_render.py](isaac_render.py) 在已有场景中逐点设置独立相机，输出 PNG。
不加载机器人，不推进物理时间；`rep.orchestrator.step_async(delta_time=0)` 只用于静止位姿渲染。
相机写入 USD session layer，不保存或覆盖场景资产。

从仓库根目录运行，使用 Isaac Sim 4.5.0 的 Python 环境：

```bash
/mnt/cfs/jiayilin/vlnverse_eccv_2026/apps/isaac-sim-4.5.0/python.sh \
  -m data_pipeline.vlnverse.replay.isaac_render \
  --scene-usd /mnt/cfs/jiayilin/vlnverse_eccv_2026/assets/vlnverse_scenes/kujiale_0018/start_result_navigation.usd \
  --poses-npz /PATH/TO/PREVIEW/poses.npz \
  --output-dir /mnt/cfs/data/beta_nav/vlnverse/NEW_PREVIEW/rgb \
  --camera-height-m 1.25 --hfov-deg 90 --width 640 --height 480
```

也可在已配置好 Isaac 的 Python 环境中执行
`python -m data_pipeline.vlnverse.replay.isaac_render`。普通 CPU 环境可以运行
`python -m data_pipeline.vlnverse.replay.isaac_render --help`，不会导入或启动 Isaac。
此 Isaac 路径需要可用的 NVIDIA GPU 与 Isaac 渲染环境；它不会自动切换到上述独立 Blender CPU 入口。

| 参数 | 默认值 | 约定 |
|---|---|---|
| `--scene-usd PATH` | 必需 | 官方导航入口 `start_result_navigation.usd`；米制、Z-up，引用资产树保持可读 |
| `--poses-npz PATH` | 必需 | `positions_xyz[N,3]`、`yaw_rad[N]`；可选 `timestamp_s[N]` |
| `--output-dir PATH` | 必需 | 新目录，存在时拒绝覆盖 |
| `--camera-height-m` | 1.25 | 在输入 Z 上增加的预览高度，米 |
| `--hfov-deg` | 90 | 预览相机水平视角，度 |
| `--width` / `--height` | 640 / 480 | PNG 分辨率 |
| `--warmup-frames` | 30 | 首个位姿保存前丢弃的渲染次数，可设为 0 |
| `--subframes` | 2 | 每次捕获的渲染子帧数，正整数 |

**这里使用选定的预览相机，不声称恢复了官方相机外参。**
输入位置应为地面参考轨迹，常见输入 Z 为 0；输出相机位置是
`input_position + [0,0,camera_height_m]`，无前向偏移和俯仰。
例如直接传入 Z=1.05 的机器人位姿会得到 Z=2.30 的相机，不能忽略该区别。
yaw 使用弧度，`camera_axes="world"` 表示 +X 朝前、+Z 向上，四元数为 wxyz。
两段对照视频应使用同一组选定的高度、视角、分辨率、场景和灯光。

输出 `frames/frame_000000.png` 等逐点图像和最后写入的 `manifest.json`。
manifest 记录输入与输出哈希、场景根文件哈希、所选相机、帧号/输入时间戳及
`physics_rollout=false`、`scene_collision_validated=false`。场景引用资产未逐个计算哈希。
PNG 不自带时序；默认稀疏轨迹应按输入 `timestamp_s` 制作视频，不能强行当成 30 Hz。

场景打开和随后引用资产加载分别最多等待 300 秒，每次捕获最多等待 180 秒；
这些是 Kit 事件循环中的超时，不能中断卡死在本地驱动内部的调用。
RGBA 尺寸、uint8 类型、有限数值、透明/全黑空帧严格检查；失败时停止，不插入随机图或复制上一帧。
灯光保持场景原样，因此全黑结果也可能要求检查实际场景灯光。
没有 `manifest.json` 的残留目录不算完成，重试应使用新的输出目录。

## 机器人控制回放（待实现）

输入：通过平面几何检查的连续参考运动及其位姿采样、版本固定的官方场景资产、机器人与相机配置。
输出：按控制周期回放的实际机器人/相机位姿、同步 RGB/深度、接触记录、跟踪误差和来源 manifest。

默认 `per_source_step` 每个源运动间隔保存 6 点，包含两端并去重共享端点。
这些点不保证全局固定 dt，不能按 30 Hz 逐点播放。
回放应使用其合成时间戳和原连续曲线，按模拟器控制周期另行采样；
不能仅将稀疏点之间改成恒速直线插值而宣称仍保留原来的速度、加速度与 jerk 曲线。
保存点数和仿真控制频率分别配置，减少保存点数不会缩短参考运动时长。

先以受控小场景核对坐标与相机外参，再接入批量 episode。参考位姿与实际反馈位姿分开保存；
逐帧瞬移渲染只能验证视觉/几何对齐，不能声称已通过动力学控制回放。

这里负责仿真器适配和采集，不计算 FM/RVQ 标签，不写 LeRobot 数据集。
新位姿应重新渲染观测，不把原始低频图像复制到新时间轴。
验收见 [ROADMAP.md](../docs/ROADMAP.md)。
