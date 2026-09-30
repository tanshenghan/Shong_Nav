# 同任务的 A* 连续曲线分支

这个分支生成合成规划参考，输出格式是 `vlnverse_planned_motion_v1`。
它与保留官方每一步的 `vlnverse_smooth_motion_v2` 分开：允许在已审阅路线走廊内改变
位置、把直行与原地转向改为连续转弯，保留起终点位置、朝向和有序检查点。
不为新曲线编造逐官方动作对应关系，不使用旧 `retime_episode` 或其验证器。

## 分层

- `planning/route.py`：有序检查点、带净空代价的 8 邻域 A*、安全视线简化、C4 五次样条和空间证书。
- `motion/planned.py`：曲线切向朝向、全局时间曲线、连续导数区间上界和固定时间采样。
- `cli/plan.py`：来源/配置核对、阶段连接、哈希与结果写入。
- `replay/render_parallel.py`：多进程渲染全新相机位姿，逐片核对并合并。
- `replay/compare_planned.py`：旧平滑轨迹与新规划的视频对照，复用共享 LightNav 轨迹带和 HUD。
- `replay/plot_route.py`：标准占据图上的完整路线与末端拐弯放大图。

大型运行结果继续放在 `/mnt/cfs/data/beta_nav/vlnverse/<run>/`。
建议本分支使用 `planned/`、`render/planned/`、`review/`、`provenance/`。

## 规划与验证

检查点索引必须严格递增并包含源首尾。每一段 A* 被限制在对应源路线的走廊内，
默认半宽 0.50 m；使用包括安全余量的 0.30 m 圆形包络。
对角网格边和起终点连接都检查完整线段扫掠，未知格和地图外不可通行。
软代价偏好远离障碍；后续视线简化保证可通行与走廊约束，不保证仍是该软代价的最优路径。

从少量导引点求解五次插值样条，内部位置和前四阶参数导数连续，
首尾一阶导数为指定朝向的单位向量，二阶为零。导引点的累计弦长是**曲线参数**，
不是精确弧长，也不是时间。所有检查点在曲线上按顺序经过。

每段五次曲线存为六个 Bezier 控制点。递归细分后，求控制点到端点弦段的最大距离，
将此包围误差加到机器人半径，再检查整条弦的圆扫掠。Bezier 凸包性质保证叶区间中的
整条曲线都被包住；单纯检查曲线采样点或相邻弦段没有这个保证。
另用凸包落在源线段胶囊中的条件证明走廊约束，对每个检查点区间分别检查。
无法证明可通行或非零切线时，细化导引点/证书，达到上限则失败，不退回未验证曲线。

前进方向是曲线切向，运动中没有横向滑动。默认时间曲线为加速、巡航、减速：
首尾速度用五次 smootherstep 衔接，中间恒定曲线参数速度；整个行程内部不停车。
还保留 `global_septic` 作为可选时间曲线。时间曲线按线速度、向量加速度、jerk、
角速度、角加速度及角 jerk 的区间上界整体缩放，最后对齐采样时间网格。
证书采用向外取整的 Bernstein 区间运算，包含曲线接头和时间曲线接头；
保存点的最大值单独报告，不能代替整段上界。浮点接头误差及容差也会记录。
这是带约束的全局调时，不是局部最短时间轨迹优化。

## 数据契约

### 200 位姿的稀疏采样预览

用户审阅后希望这条任务约 200 帧即可。当前推荐示例配置为
`configs/planned_0055_compact.json`，`motion.sample_hz=4.5`，对同一条已验证的连续曲线
重新等时间采样，得到 200 个位姿；平均相邻距离约 7.31 cm、最大约 8.85 cm。
曲线和空间证书不变，速度上限不放宽。合成参考时长约 44.22 秒，时间步长 1/4.5 秒。
为这些新位姿重新渲染 RGB，不把原 664 帧中最近的一张当作新视点的观测。

配合 `compare_planned --playback-mode pose_frames`，每个位姿播放一帧、15 FPS，
成片 200 帧、13.333 秒，是约 3.33 倍速的预览。保存的轨迹时间戳与参考速度仍按原有约束定义；
不能把视频的 1/15 秒写回训练样本当作运动间隔。
未来 30 点此时覆盖约 6.667 合成秒，视频上为两秒；逐帧记录实际剩余 horizon，末尾不补点。
`reference_time` 保持原来严格 15 Hz 输入的默认行为；需要更少帧时必须显式选择 `pose_frames`。

稀疏版本输出目录为 `/mnt/cfs/data/beta_nav/vlnverse/planned_20260919_kujiale_0055_10_3_200poses/`。
这只是减少记录与观测数量，避免把视频 FPS 直接当成必须采集的位姿频率。

### 轨迹文件

`planned/` 在所有计算和检查通过后才创建，拒绝覆盖既有目录。
`manifest.json` 最后写入，作为完成标志，保存来源、地图、配置及代码哈希。

- `source_poses.npz`：完整原始位置、yaw、可用的源时间戳；源输入只读。
- `planned_curve.npz`：`bezier_xy[S,6,2]`、`parameter_knots[S+1]`、`astar_xy`、`guide_xy`。
- `trajectory.npz`：`timestamp_s`、`positions_xyz`、`yaw_rad`、`velocity_xyz`、
  `acceleration_xyz`、`jerk_xyz`、`yaw_rate_rad_s`、`yaw_accel_rad_s2`、`yaw_jerk_rad_s3`、`curve_parameter`。
- `manifest.json`：`planning` 保存空间证书；`quality` 保存时间定义、连续上界与区间明细；
  `route_alignment` 保存检查点及其选择依据。没有 `source_progress` 或每源动作补点的语义。

此前 664 位姿版本选用 15 Hz，所有采样间隔为 1/15 合成秒。视频每个位姿显示一帧、15 FPS，
因此右侧播放与参考时间一致；最后一帧仍占一个视频帧时长，容器时长为 `N/15`，
最后一个位姿时间为 `(N-1)/15`。未来 30 点表示之后两秒的参考位姿，末尾缩短、不补点。

左侧旧轨迹按其合成时间归一化到新行程时长，显示已渲染的旧图像；
左右不保证相同空间进度，左侧播放时间被改变。视频记录各旧帧的出现次数及遗漏帧，
不使用只按弧长匹配而跳过全部原地旋转的方式。顶部完整指令不变。

## 0055 示例命令

在仓库根目录执行，以下 `OLD` 和 `NEW` 是 shell 示例变量，不是程序自动搜索结果。

```bash
OLD=/mnt/cfs/data/beta_nav/vlnverse/preview_20260919_kujiale_0055_10_3
NEW=/mnt/cfs/data/beta_nav/vlnverse/planned_20260919_kujiale_0055_10_3
data_pipeline/.venv/lerobot/bin/python -m data_pipeline.vlnverse.cli.plan \
  --episode-parquet "$OLD/official/kujiale_0055/10_3/data/chunk-000/episode_000000.parquet" \
  --occupancy-npz "$OLD/provenance/scene_occupancy_v2.npz" \
  --config data_pipeline/vlnverse/configs/planned_0055.json --output-dir "$NEW/planned"
data_pipeline/.venv/blender-preview/bin/python -m data_pipeline.vlnverse.replay.render_parallel \
  --scene-usd /mnt/cfs/jiayilin/vlnverse_eccv_2026/assets/vlnverse_scenes/kujiale_0055/start_result_navigation.usd \
  --poses-npz "$NEW/planned/trajectory.npz" --output-dir "$NEW/render/planned" \
  --workers 4 --threads 12 --samples 8
PYTHONPATH=src data_pipeline/.venv/lerobot/bin/python -m data_pipeline.vlnverse.replay.compare_planned \
  --previous-retimed-dir "$OLD/retimed" --previous-images "$OLD/render/smoothed" \
  --planned-dir "$NEW/planned" --planned-images "$NEW/render/planned" --output-dir "$NEW/review"
data_pipeline/.venv/lerobot/bin/python -m data_pipeline.vlnverse.replay.plot_route \
  --planned-dir "$NEW/planned" --output "$NEW/review/route_overview.png"
```

## 验收范围

检查点依据官方 RGB 与逐步轨迹人工选择，约束经过顺序和路线邻域；这不自动证明
Instruction 中每个地标、左右关系都正确。新 RGB 由新位姿重新渲染，不能挪用旧 RGB 当新观测。
Blender 仍为近似材质/灯光和选定相机的预览，不等价于官方 Isaac RGB。
共享轨迹带有视觉前移且没有深度遮挡，仅用于审阅，不进入干净图像。

输出是已做二维几何和参考运动检查的合成轨迹，不是物理机器人执行结果，也不是已经完成
训练导出的 LeRobot 数据。机器人控制跟踪、实际状态/接触/深度、最终标签与训练导出继续分阶段实现。
