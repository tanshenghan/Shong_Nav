# 数据契约

新增的 A* 连续曲线分支使用独立 `vlnverse_planned_motion_v1` 格式，详见
[PLANNED_ROUTES.md](PLANNED_ROUTES.md#数据契约)。其输出为固定时间采样、无官方逐步对应关系；
下文 `vlnverse_smooth_motion_v2` 的源锚点和 `source_progress` 要求只适用于原时间重定分支。

## 来源：官方逐步轨迹

`source/` 读取官方 `traj_data/` 中的 episode Parquet。
`raw_data/` 中的任务描述与 `reference_path` 用于指令和路径核对，
不能替代逐步机器人体朝向。官方逐步位置与 yaw 是本阶段的运动输入。

机器人位置、相机位置是不同字段，不能互换。原始 `timestamp`、步号及动作类别不代表
已经确认过的物理采样周期。接入时必须检查帧顺序、有限数值、位置维度、yaw 单位及 episode 边界。
所有输入都保持只读；派生输出记录输入文件哈希和解析配置。

当前读取器要求 `observation.robot_position` 和 `observation.robot_yaw`，
以及 `frame_index` / `observation.step` 至少一个连续整数索引。
若有 `observation.robot_orientation`，校验其为单位 `wxyz` 四元数且与 Z-up yaw 一致。
支持数值列表和官方文件中 JSON 字符串形式的数组。
若包含 `episode_index`，它必须单一且与 `episode_XXXXXX.parquet` 文件名一致。
不会从动作类别补造缺失位姿，也不静默重新排序不连续的记录。

未显式给出元数据时，会尝试读取对应 episode 根目录的 `meta/episodes.jsonl`；
关联到的指令和 `finish_status` 保存在来源元数据中。
官方文件名中的 `episode_000000` 可能在不同目录重复，episode 身份必须包含场景/轨迹目录。

## 场景：标准占据图 NPZ

| 字段 | 约定 |
|---|---|
| `occupied` | 二维 bool 数组；`True` 表示障碍或未知，`False` 表示确认空闲 |
| `resolution_m` | 正数，单位米 / 栅格 |
| `origin_xy_m` | 两个数，第 0 行、第 0 列栅格外角点在世界 XY 中的位置，非像素中心 |
| `grid_yaw_rad` | 可选，栅格坐标相对世界 XY 的逆时针旋转，默认 0 |

栅格行沿局部 +Y，列沿局部 +X。单元 `(row, col)` 覆盖局部
`[col, col+1] × [row, row+1] × resolution_m`，先按 `grid_yaw_rad` 旋转，再加原点得到世界 XY。
地图外视为障碍；接触障碍边界或地图边界也算碰撞。
已知未知区域必须在写 NPZ 时转成 `occupied=True`，不能标为空闲。
地图生成还应保存资产版本、生成高度范围和坐标校准证据；单个二维数组不能表达这些来源信息。

输入地图必须与机器人**世界 XY** 对齐，不能直接拿相机图像或未经核实翻转的 PNG 充当占据图。
机器人半径和安全余量在 motion 配置中给出。该地图只支持当前平面运动检查，
多楼层、坡道和三维完整包络需要后续场景接口扩展。

### 已支持的官方地图适配

`scene.official_map.load_official_map(freemap_path, scene_dir, resolution_m=0.05)`
返回标准 `OccupancyGrid` 和校准证据字典；目前是 Python 接口，没有另造命令入口。
`scene_dir` 必须包含 `occupancy.json`、`occupancy.png`、`rooms.json`、`room_in_images.json`。
只接受带世界坐标表头的 `freemap.npy`：X 随列递减，Y 随行递增；
类别 0/1/2 对应占据/空闲/未知，与 PNG 的 0/255/127 逐像素一致。
坐标边界和房间顶点对应关系也必须通过交叉检查；缺文件或不同格式不会被猜测接受。

转换按原始坐标采样定义像素边界，并保守重采样成方格；未知和图外仍为障碍。
证据字典记录原始轴向、间距、类别、地图高度区间、房间顶点核对结果及各输入哈希。
当前 `kujiale_0018` 实例的地图仅覆盖 0.1–1.0 m 高度范围，不能用它宣称完整 3D 底盘/机身验证。
校准和候选筛选记录见 [preview_geometry_20260919.json](../reports/preview_geometry_20260919.json)。

## 连续参考运动与位姿采样

当前输出契约为 `schema_version=2`、`format="vlnverse_smooth_motion_v2"`，
这两个标识记录在 `manifest.json`。`trajectory.npz` 保存连续参考运动的离散采样；
所有数组的同一行对应同一采样时刻，但默认模式不保证相邻行具有全局固定时间间隔。

| 字段 | 形状 / 单位 | 含义 |
|---|---|---|
| `timestamp_s` | `[N]`，秒 | 从零开始的合成时间轴；实际间隔取相邻值之差 |
| `positions_xyz` | `[N,3]`，米 | 世界坐标中的参考位置 |
| `yaw_rad` | `[N]`，弧度 | 连续展开的世界 yaw，可超出 `[-π,π]` |
| `velocity_xyz` | `[N,3]`，米 / 秒 | 世界坐标速度 |
| `acceleration_xyz` | `[N,3]`，米 / 秒² | 世界坐标加速度 |
| `jerk_xyz` | `[N,3]`，米 / 秒³ | 世界坐标加加速度 |
| `yaw_rate_rad_s` | `[N]`，弧度 / 秒 | 角速度 |
| `yaw_accel_rad_s2` | `[N]`，弧度 / 秒² | 角加速度 |
| `yaw_jerk_rad_s3` | `[N]`，弧度 / 秒³ | 角加加速度 |
| `source_progress` | `[N]` | 对源记录的连续进度映射；不是源物理时间 |

`source_poses.npz` 保存原始 `positions_xyz`、`yaw_rad`，以及来源存在时的
`source_timestamps`；源时间戳保留原值，不承诺单位为秒。它与 `source_progress`
共同支持将新轨迹追溯到原始逐步位姿。

`manifest.json` 保存生成配置、输入与输出哈希、来源关联和质量说明，最后写入，
作为该条轨迹成功完成的标记；没有 manifest 的目录不能当作成功产物使用。
时间语义为 `synthetic_retimed_seconds`，不能写成 measured / real-time expert rollout。
平面碰撞检查与实际仿真检查必须分别标识；首版没有仿真实测碰撞通过的结论。
manifest 同时记录采样方式和点数含义；默认 `quality.sample_hz=null`。
任何读取器都必须先核对采样语义，用 `timestamp_s` 求 `dt`，不能把默认输出按 30 Hz 播放。
`manifest.sampling.uniform_timestamps` 根据实际时间戳判断是否等间隔；
每源间隔采样也可能恰好等间隔，此时 `quality.sample_hz` 仍为 null，表示没有配置固定采样频率。
`quality.continuous_maxima` 是整个运动段的解析导数峰值，用于检查完整曲线的约束；
`quality.maxima` 是保存点上的导数最大值，不能替代前者。
`primitives` 中的 `source_anchor_indices`、`source_motion_count` 记录源锚点及非重复运动间隔数，
结合源位姿、段时长和七次进度定义支持重建连续曲线。

机器人原地转向时允许连续多帧位置相同而 yaw 不同；不能因此删除这些帧。
当前位置、速度和角速度都是参考值，不代表已经实际执行的控制器反馈。

### 两种采样方式

| 模式 | 采样规则 | 时间约定 |
|---|---|---|
| `per_source_step`（默认） | 每个非重复原始运动间隔取 `poses_per_step` 个点，含起终点，共享端点去重 | 每个原始间隔内等时间采样，不保证全局固定 dt；`quality.sample_hz=null` |
| `uniform_time` | 按 `sample_hz` 均匀采样 | 相邻样本时间差为 `1/sample_hz`；使用该模式时 `sample_hz` 才生效 |

`poses_per_step=6` 时，`M` 个非重复源运动间隔产生 `1+5M` 个采样点；
可选值为 5、6、7、8，一般点数是 `1+(poses_per_step−1)M`。
同向运动仍先合并；各源锚点的几何进度通过反解七次进度曲线映射到时间，
再在两个锚点时间之间采样。采样保留源锚点（在共线合并容差内），不会让同向源动作接头反复停车。
改变每段保存点数不改变由运动约束求出的段时长。

七次曲线由起终点进度及前三阶导数的端点条件确定，详见 [DESIGN.md](DESIGN.md#当前实现保留路线重新安排时间)。
这些采样点来自已确定的曲线，不是先取 5 个点再学习一个轨迹模型；当前仍未实现绕角弧线。

## 命令参数

入口：`python -m data_pipeline.vlnverse.cli.retime`。

| 参数 | 必需 | 作用 |
|---|---|---|
| `--episode-parquet PATH` | 是 | 单个官方 episode 的逐步位姿 Parquet |
| `--occupancy-npz PATH` | 是 | 已校准的标准占据图 |
| `--output-dir PATH` | 是 | 新建输出目录，拒绝覆盖既有运行 |
| `--config PATH` | 否 | 时间重定配置 JSON，默认本目录的 `configs/retime.json` |
| `--episode-id ID` | 否 | 显式指定来源 episode ID |
| `--metadata-jsonl PATH` | 否 | 官方任务元数据关联 |

JSON 参数的名称、单位和默认值以 [configs/retime.json](../configs/retime.json) 和命令帮助为准。
当前默认如下；它们是初始配置，不代表机器人实测性能。

| 配置键 | 默认值 | 含义 |
|---|---:|---|
| `sample_mode` | `per_source_step` | 默认每源运动间隔采样；另支持 `uniform_time` |
| `poses_per_step` | 6 | 每个非重复源运动间隔的点数，含两端；仅可取 5、6、7、8 |
| `sample_hz` | 30 | 仅 `uniform_time` 模式生效的均匀采样频率，Hz |
| `robot_radius_m` | 0.25 | 圆形底盘包络半径，米 |
| `safety_margin_m` | 0.05 | 额外安全余量，米；检查半径为两者之和 |
| `max_speed_m_s` | 0.4 | 最大线速度，米 / 秒 |
| `max_accel_m_s2` | 0.5 | 最大线加速度，米 / 秒² |
| `max_jerk_m_s3` | 1.0 | 最大线 jerk，米 / 秒³ |
| `max_yaw_rate_rad_s` | 0.8 | 最大角速度，弧度 / 秒 |
| `max_yaw_accel_rad_s2` | 1.0 | 最大角加速度，弧度 / 秒² |
| `max_yaw_jerk_rad_s3` | 2.0 | 最大角 jerk，弧度 / 秒³ |
| `max_heading_error_rad` | 约 0.01745（1°） | 直行位移方向与 yaw 之间的容差 |
| `planar_tolerance_m` | 0.00001 | 源轨迹 Z 变化上限，米 |
| `max_merge_deviation_m` | 0.00001 | 共线合并对源锚点的最大偏差，米 |

任何参数变化都必须进入输出 manifest；速度参数是设计输入，不能当作已测机器人性能。

## 视觉预览与比较视频

默认预览使用同一场景、同一渲染器和选定相机配置，分别渲染 `source_poses.npz` 与 `trajectory.npz`。
高度默认在输入 Z 上加 1.25 m，HFOV 为 90°；这是预览选择，不是恢复出的官方相机标定。
场景选择遵循官方导航入口 `start_result_navigation.usd`；原始场景可能含已被导航版删除的门体，
不能静默替换。渲染 manifest 必须记录实际使用的场景路径，两版须一致。

| 入口 | 输出格式 | 边界 |
|---|---|---|
| `replay.blender_render` | `000000.png` 等，最后写 `render_manifest.json` | Cycles CPU；近似 MDL/纹理与统一灯光，不等价于 Isaac；本例 23 个原始/111 个平滑位姿全部渲染验证通过 |
| `replay.isaac_render` | `frames/frame_000000.png` 等，最后写 `manifest.json` | 独立 Isaac 相机；当前尚无 GPU 渲染验证 |
| `replay.compare` | 两个独立 MP4、一个并排 MP4、`comparison_manifest.json` | 读取 Blender 编号图像与 render manifest；可选同 episode 的官方 RGB 替换左侧 |

比较入口要求 `per_source_step` 的源锚点对应关系。默认 `reference_time` 使用 `timestamp_s` 与 `source_progress`
为源位姿安排合成时刻。每张图保留到下一位姿，不用光流，不把稀疏位姿当成 30 Hz。
`--fps` 默认 20，只是视频容器频率；两版视频帧数和时长相同。
`reference_duration_s` 与包含终点显示的 `video_duration_s` 分别记录，不能用后者反推官方动作时长。
当前文件名为 `original_gt.mp4`、`smoothed_{K}poses.mp4`、`side_by_side.mp4`；
`K` 来自 `poses_per_step` 配置，默认输出 `smoothed_6poses.mp4`。
本例三段视频各有 603 帧，20 FPS、30.15 秒，均完整解码通过；23 个源锚点与平滑对应帧 RGB
逐像素一致。记录见 [预览生成报告](../reports/preview_render_20260919.json)。
显式 `pose_frames --fps 15` 则使每个平滑位姿只显示一帧，源位姿按源锚点同步并保持到下一锚点。
本例为 111 帧、7.4 秒，完整解码通过，见 [15 FPS 播放报告](../reports/preview_playback_15fps_20260919.json)。
`playback_mode`、播放帧对应的位姿索引与参考时间写入 manifest；NPZ 的 `timestamp_s` 保持不变。
此模式只定义视频时钟，不保留参考 `dt`，不能据此断言播放运动满足原速度、加速度或 jerk 限制。

默认“原始 GT”是原始官方位姿的重新渲染。显式 `--official-rgb-npy PATH` 才使用官方 RGB 作为左侧画面：

- 源 Parquet 必须为 `EPISODE/data/chunk-000/episode_000000.parquet`，RGB 必须来自同根目录的
  `EPISODE/videos/chunk-000/observation.images.rgb/rgb.npy`，不只核对帧数。
- 数组必须为 `uint8 [N,H,W,3]`，`N` 等于源位姿数。记录文件哈希、原始尺寸和来源关系。
- 官方帧保持宽高比缩放、居中留黑边，不裁切；帧按源锚点安排，不做时间插帧。
- 当前仍需要 `--original-images` 与 `--smooth-images` 的完整 Blender 图像及 manifest 校验；
  `official_rgb` 在通过校验后替换左侧，右侧仍是新位姿的 Blender 预览。
- 此模式文件名为 `official_rgb.mp4`、`smoothed_{K}poses.mp4`、`official_vs_smoothed.mp4`；
  默认模式原文件名不变。记录 `left_view_source`、`official_rgb`、`matched_left_right_camera=false`。
  manifest 原有 `camera` 字段仅描述 Blender 预览；官方相机外参尚未验证。

两侧相机、渲染器、材质和环境不等价，视频标题/页脚必须明确来源，不能把 Blender 图像称为官方 RGB。
即使启用官方图像，比较视频也没有恢复真实动作时间或实际控制器反馈。
渲染 manifest 和 comparison manifest 都不构成物理避障验证或训练标签。
完整参数见 [replay/README.md](../replay/README.md)。

## 后续物理观测、标签和导出契约

- `replay/` 必须分别保存 reference 与 actual pose，观测明确绑定实际相机位姿和时间戳。
  连续参考可另按仿真控制周期采样，控制频率与默认每源间隔的保存点数分别设置。
- `labels/` 用实际位姿生成机器人局部 `[forward_m, lateral_m, yaw_rad]`，
  左为正、逆时针为正；标签需写出时间间隔和 horizon 秒数。
- 默认稀疏位姿的“后 10 个点”没有固定秒数，不能由点号构造恒定 dt。
  标签必须按秒级 horizon 从实际带时间戳的回放中选取；不能继承 R2R 的 10/30 个离散动作含义。
- FM、RVQ、apos、opos 使用独立 mask。连续运动 GT 不以 RVQ 重建替代；
  码本适配新速度尺度与采样间隔必须单独检查。
- 暂无回放观测的 `trajectory.npz` 不得被标成完整训练数据；
  新位姿和旧 RGB 也不能按帧号拼接成“对齐”样本。
- `export/` 明确 fps、时间语义、相机参数、标签参数与来源版本；
  导出后独立训练读取不得依赖生成代码或临时采集目录。
