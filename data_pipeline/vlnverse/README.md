# VLNVerse 数据管线

本目录独立管理 VLNVerse 的来源读取、场景地图、运动轨迹、回放、标签和导出；
R2R / RxR 继续由相邻的 [R2R/](../R2R/README.md) 管理。

另有独立的 **A* 连续曲线分支**：在已审阅路线走廊内，经过有序检查点重新规划，
将小幅原地转向改为连续行驶转弯，并保留源起终点位置和朝向。
代码在 `planning/`、`motion/planned.py`，使用单独的 `vlnverse_planned_motion_v1` 契约。
它不受“每个官方动作补 6 点”的组织方式约束，按合成时间均匀采样。
入口、证书和运行目录见 [PLANNED_ROUTES.md](docs/PLANNED_ROUTES.md)。

用户偏好的当前预览采用 **200 位姿**：同一连续曲线按 4.5 Hz 重新采样、重新渲染，
15 FPS 播放约 13.33 秒。平均相邻点距离约 7.3 cm；参考运动仍约 44.22 秒，
视频为加速审阅，训练时间戳不跟着视频缩短。配置见 `configs/planned_0055_compact.json`，
比较入口显式使用 `--playback-mode pose_frames`；664 位姿版本保留在下方作为历史。

- [200 帧短版视频](/mnt/cfs/data/beta_nav/vlnverse/planned_20260919_kujiale_0055_10_3_200poses/review/planned.mp4)
- [200 帧旧/新对照](/mnt/cfs/data/beta_nav/vlnverse/planned_20260919_kujiale_0055_10_3_200poses/review/before_after.mp4)
- [200 位姿生成报告](reports/planned_0055_compact_20260919.json)：443 项相关测试、全部视频帧与未来窗口检查通过。

`kujiale_0055_10_3` 已完成这一分支的首条样例：原路线 14.50 m，新连续曲线约 14.55 m，
8 次途中原地转向变为 0 次，最大偏离源路线不超过 10.44 cm。
新轨迹 664 位姿、15 Hz、44.2 合成秒；成片 15 FPS、44.267 秒（含最后一帧显示时长）。
0.30 m 圆包络及运动导数约束通过；434 项相关测试通过，两段视频完整解码通过。

- [旧停转 / 新连续曲线对比](/mnt/cfs/data/beta_nav/vlnverse/planned_20260919_kujiale_0055_10_3/review/before_after.mp4)
- [新连续曲线视频](/mnt/cfs/data/beta_nav/vlnverse/planned_20260919_kujiale_0055_10_3/review/planned.mp4)
- [俯视路线与转弯放大图](/mnt/cfs/data/beta_nav/vlnverse/planned_20260919_kujiale_0055_10_3/review/route_overview.png)
- [生成与独立验证报告](reports/planned_0055_20260919.json)

对比左侧按旧合成时间归一化，8 段旧转向完整显示；右侧保持新参考时间。
该产物仍是 Blender 相机预览与合成运动参考，不代表机器人控制回放或正式训练数据导出。

目标是把官方逐步位姿变成**允许原地转向的移动机器人的连续、平滑参考运动**，
再在场景中回放并生成与新位姿严格对齐的训练图像及标签。
默认每个原始运动间隔保存 6 个位姿点（含起终点），共享端点只保留一次。
生成结果是派生轨迹，不代表官方提供了相应的实测运动。

当前已实现官方逐步位姿读取、受限格式的官方占据图校准、平面扫掠碰撞检查、
受速度/加速度/jerk 约束的时间重定，以及独立相机预览和双视频编码入口。
已完成 Blender CPU 预览与官方 RGB 对照：最新 14.5 m 样例有 67 个原始位姿和 331 个平滑位姿，
两种对照的六段视频完整解码通过；此前 23 → 111 位姿样例保留在历史记录中。
此结果仅覆盖本例 CPU 视觉预览；Isaac 入口尚未通过 GPU 实测。
机器人控制回放、实际位姿/接触/深度采集、FM/RVQ/apos/opos 标签、LeRobot 导出仍待实现。
完整方案见 [DESIGN.md](docs/DESIGN.md)，输入输出定义见
[DATA_CONTRACT.md](docs/DATA_CONTRACT.md)，后续验收顺序见 [ROADMAP.md](docs/ROADMAP.md)。

## 目录职责

```text
data_pipeline/vlnverse/
  source/       官方 Parquet、任务元数据读取与来源追溯
  scene/        官方占据图校准、标准栅格与底盘包络碰撞查询
  planning/     A*、有序检查点走廊、连续空间曲线与避障证书
  motion/       轨迹分段、时间重定、连续运动采样
  validation/   数值、几何及质量检查
  cli/          薄命令入口，不放核心算法
  configs/      可审阅的速度、底盘、采样方式与点数等参数
  replay/       Blender/Isaac 独立相机预览与同时间轴视频；物理回放待实现
  labels/       待实现：FM、RVQ、apos、opos 与独立 mask
  export/       待实现：LeRobot 写入、统计和读取验证
  tests/        不依赖仿真器、GPU或场景权重的 CPU 测试
  docs/         方案、数据契约与阶段验收
  reports/      小型审计、质量统计及生成记录
```

每一层通过明确的数据契约连接。`motion/` 不导入仿真器；`replay/` 不负责标签；
`labels/` 不下载资产；`export/` 不修复轨迹。正式训练读取继续放在
`src/lightnav/dataloader/`，不依赖这个生成目录。

## 第一版如何处理平滑与避障

1. 读取机器人的位置和 yaw，保留原地旋转；运动依据来自位姿，而不是动作类别。
2. 在明确的误差上限内合并同向共线直行和同向原地旋转，避免每个 0.25 m 采样点都停车。
3. 保留原路线，转角处平滑减速到零，再平滑原地转向、平滑起步。
4. 按底盘圆形包络加安全余量检查整段扫掠区域；未知格、地图外按不可通行处理。
5. 用端点条件确定的七次时间曲线控制速度、加速度和 jerk，再按原始运动间隔采样。

默认配置为 `sample_mode="per_source_step"`、`poses_per_step=6`，点数可选 5、6、7、8。
每个非重复的原始相邻运动采 6 个点，含起终点；若有 `N` 个运动，去重后共有 `1+5N` 个点。
同向连续运动先合并后分配时间，所以源动作的接头经过但不停顿；需要原地转向时仍平滑停车。
**输出不保证全局固定时间间隔，必须用 `timestamp_s` 计算每段 `dt`。**
减少保存点数不会缩短运动时长，也不表示机器人自动加速。

需要均匀时间采样时，在配置中显式设置 `sample_mode="uniform_time"`；
此时 `sample_hz` 才生效，默认数值为 30 Hz。默认的每段 6 点方案不是 30 Hz 输出。

这版首先保证“沿原路线可解释地运动”。它不会把拐角直接拉成可能切过墙角的样条。
后续若希望宽阔区域连续带弧转弯，应增加具有碰撞和非完整约束的局部优化，
再验证每段扫掠区域。仅提高采样频率不能提高避障保证。

## 单条轨迹入口

从仓库根目录运行；Python 环境需要 NumPy、PyArrow。
输入必须是**机器人体坐标约定已核对的官方逐步 Parquet**和**已校准的标准占据地图**。
`scene/official_map.py` 已通过坐标表头、PNG 类别和房间顶点对应关系校准受支持的官方地图，
再保守转换为标准 NPZ；缺少校准文件或约定不符时失败，不猜测黑白含义或坐标翻转。

```bash
python -m data_pipeline.vlnverse.cli.retime \
  --episode-parquet /PATH/TO/OFFICIAL/episode_000000.parquet \
  --occupancy-npz /PATH/TO/CALIBRATED/scene_occupancy.npz \
  --config data_pipeline/vlnverse/configs/retime.json \
  --output-dir /mnt/cfs/data/beta_nav/vlnverse/NEW_RUN/retimed/SCENE/EPISODE
```

`--output-dir` 必须是新目录。完整参数见 [DATA_CONTRACT.md](docs/DATA_CONTRACT.md#命令参数)。
输出 `source_poses.npz`、`trajectory.npz` 和 `manifest.json`，分别保存原始位姿、
连续参考运动的位姿采样、来源与质量记录。包含合成秒时间戳、位置、yaw 及其导数，
以及输入和输出哈希、配置和质量范围。它们目前不是可直接训练的 LeRobot 数据集。
当前输出契约为 `schema_version=2`、`format="vlnverse_smooth_motion_v2"`，
默认 `quality.sample_hz=null`；完整的连续曲线定义见设计文档。
整段曲线的解析导数峰值也会检查，不依赖保存点数足够密集。

默认底盘与运动约束参数是可调整的初始配置，不是经过机器人实测的约束。
先校准地图与底盘尺寸，再做小规模可视化和仿真验收，最后开展批量生成。

## 当前几何样例与视觉预览

最新超过 10 m 的样例为 **`kujiale_0055_10_3`，14.5 m、67 个原始位姿 → 331 个平滑位姿**。
路线经过门口、餐桌旁、客厅和临窗区域，再右转前往洗手池与洗衣机。
官方 RGB、原始任务和逐步位姿已核对；相邻 66 段通过半径 0.30 m 的精确欧氏圆扫掠检查。
每步 6 点、共享端点去重；视频 15 FPS、22.0667 秒，参考合成时间仍为 97.9568 秒。

- [官方 RGB 视频](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0055_10_3/review_official_rgb_15fps/official_rgb.mp4)
- [左官方 RGB / 右平滑预览](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0055_10_3/review_official_rgb_15fps/official_vs_smoothed.mp4)
- [同相机 GT 位姿 / 平滑位姿对照](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0055_10_3/review_lightnav_15fps/side_by_side.mp4)

保留顶部完整 Instructions、共享 LightNav HUD 与未来最多 30 个平滑位姿的轨迹带。
官方图像按源锚点保持，不进行时间插帧；它与右侧 Blender 的相机和材质并不等价。
指令中的物体外观以官方 RGB 为准。完整检查与限制见 [长路线生成报告](reports/preview_long_0055_20260919.json)。

历史调试样例 `kujiale_0172_51_0` 已通过半径 0.30 m 平面扫掠检查：
**23 个原始位姿 → 111 个平滑采样位姿，含 4 次 ±15° 原地转向，合成参考时长约 34.435 秒**。
此前 `kujiale_0018_2_2` 是 23 → 111 点、约 30.078 秒的直行样例，保留于历史结果。
这些样例不能代表连续绕角弧线或全场景验证。
初选 `kujiale_0195_4_2` 在 0.30 m 包络下未通过扫掠检查；筛选历史保留于
[几何报告](reports/preview_geometry_20260919.json)，没有为通过检查而缩小机器人。

视觉预览入口集中在 [replay/README.md](replay/README.md)：

- `blender_render.py`：独立 CPU 环境的 Cycles 渲染，核对场景世界变换并近似转换 MDL 材质。
- `isaac_render.py`：需要 GPU 的原生 Isaac 相机入口，当前未完成 GPU 渲染验证。
- `compare.py`：使用新渲染 PNG 生成原始 GT、平滑版及并排视频，支持逐位姿播放和合成参考时间播放。

比较入口还支持 `--overlay-style lightnav`，直接复用 `src/lightnav/viz` 的蓝色轨迹带与 HUD。
该模式需要水平相机、无 3D 标注的干净底图；轨迹带含视觉前移且没有深度遮挡，只是审阅示意。
HUD 速度使用保存的合成参考导数，帧号/FPS 使用播放时钟，不能混为实测运动。
命令与边界见 [共享 LightNav 可视化](replay/README.md#可选共享-lightnav-轨迹带与-hud)；
`kujiale_0172_51_0` 的完整 LightNav 风格视频已生成并通过全部解码验证，调试样例链接见下方。

后续 [指令与画面审计](reports/preview_fidelity_audit_20260919.json) 确认：23 帧来源完整、
指令和终点匹配，但两条预览明显偏向短直路线，不能代表训练分布。
`original_gt.mp4` 是按官方位姿重新渲染的画面，并非官方 RGB；当前 Blender 预览丢失了
窗外环境和部分瓷砖材质等指令地标。解码、位姿和运动测试通过不等于指令语义或视觉一致性通过。
这两条只作为运动与绘图调试样例；用于语义验收前需对照官方 RGB 恢复相机与地标外观。

CPU 预览默认两版都使用输入 Z + 1.25 m、HFOV 90°、480×360 的同一组选定相机。
推荐的 `kujiale_0172_51_0` 两侧使用水平相机干净底图，再叠加共享 LightNav 可视化；
历史 `kujiale_0018_2_2` 场景标注版两侧下俯 20°，用青色连线和绿色圆点表示落地路径。
场景使用对应目录的官方导航入口 `start_result_navigation.usd`；例如原始 `kujiale_0018.usda`
保留了导航版删除的门体，不能代替导航场景。CPU 环境按
[固定依赖文件](requirements-preview-cpu.txt) 单独安装，具体命令见上述预览文档。
当前推荐 `--playback-mode pose_frames --fps 15`：右侧每个平滑位姿显示一帧，无重复帧；
左侧保持上一源位姿直到下一个源锚点。长路线样例为 331 帧、约 22.067 秒，
旧的 23 位姿样例为 111 帧、7.4 秒。
这只改变播放时钟，轨迹 NPZ 的非均匀 `timestamp_s` 保持不变；
每个位姿固定显示 1/15 秒不保留参考 `dt`，不能据此声称播放速度仍满足原运动约束。
默认 `reference_time` 模式仍可按合成参考时间保持视图；两种模式均不恢复官方真实运动时间。
预览图像不是物理回放实测，也不产出训练标签。

运动与绘图调试样例 `kujiale_0172_51_0`：完整 Instructions + LightNav 轨迹带与 HUD，15 FPS（111 帧、7.4 秒）：

- [LightNav 风格并排对比](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0172_51_0/review_lightnav_15fps/side_by_side.mp4)
- [未来 30 个平滑位姿与 HUD](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0172_51_0/review_lightnav_15fps/smoothed_6poses.mp4)
- [原始 GT 视频](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0172_51_0/review_lightnav_15fps/original_gt.mp4)

三个视频完整解码通过；独立版 480×686，并排版 960×626。23 个原始和 111 个平滑底图各自全部唯一，
23 个源锚点的两版底图 RGB 差为 0；完整指令与未来窗口全部核对通过。
相关 341 项测试及 Ruff 通过，记录见 [LightNav 风格预览报告](reports/preview_lightnav_viz_0172_20260919.json)。

历史 `kujiale_0018_2_2`：完整 Instructions + 未来 30 个平滑位姿的 3D 地面标注，15 FPS（111 帧、7.4 秒）：

- [原始 GT 视频](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review_instruction_path_15fps/original_gt.mp4)
- [带未来路径的平滑视频](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review_instruction_path_15fps/smoothed_6poses.mp4)
- [带指令和路径的并排对比](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review_instruction_path_15fps/side_by_side.mp4)

新 23 个原始/111 个平滑视图全部完成，三个视频完整解码通过；独立版 480×710，并排版 960×656。
指令与官方元数据逐字一致，111 帧的未来窗口逐帧核对通过。30 表示之后的平滑位姿数，
不是 30 个原始动作；终点前自然减少且不补点，视野外或被场景遮挡的路径自然不可见。
标注用于审阅，不能作为干净训练 RGB。记录见 [指令与路径预览报告](reports/preview_instruction_path_20260919.json)。

历史版本：水平相机、无地面标注，15 FPS（111 帧、7.4 秒）：

- [原始 GT 视频](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review_15fps/original_gt.mp4)
- [每源间隔 6 点的平滑视频](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review_15fps/smoothed_6poses.mp4)
- [并排对比视频](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review_15fps/side_by_side.mp4)

三段视频完整解码通过，右侧全部 111 个位姿各一帧；播放规则与检查记录见
[15 FPS 播放报告](reports/preview_playback_15fps_20260919.json)。
原先按合成参考时间播放的历史版本保留，均为 603 帧、20 FPS、30.15 秒（容器时长）：

- [原始 GT 视频](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review/original_gt.mp4)
- [每源间隔 6 点的平滑视频](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review/smoothed_6poses.mp4)
- [并排对比视频](/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0018_2_2/review/side_by_side.mp4)

历史未标注版的 23 个源锚点与对应平滑帧逐像素一致，首中尾视图已核对；验证范围与记录见
[预览生成报告](reports/preview_render_20260919.json)。

## 数据与运行文件放哪里

代码、配置、文档、CPU 测试、小型报告在本目录。已有官方资产保持只读；
大型新结果放在 `/mnt/cfs/data/beta_nav/vlnverse/<run_id>/`，建议使用：

```text
<run_id>/
  provenance/       输入清单、版本、哈希、冻结配置
  maps/             校准后的场景占据图与变换信息
  retimed/          连续参考运动及带时间戳的位姿采样
  replay/           仿真实际位姿、图像、深度及接触记录
  labels/           连续运动与指向标签
  lerobot/          最终自包含训练数据
  review/           轨迹叠图、视频与人工抽查
  state/            批量作业状态、失败原因和续跑信息
```

这是一套后续批量管线的目录约定，不表示所有阶段已经生成。
每次改变地图、运动约束、相机或标签定义都生成新的运行版本；不覆盖官方数据或历史快照。
现有 `data_pipeline/.venv/` 环境保留原位，仿真器依赖与 CPU 转换依赖分开管理。
