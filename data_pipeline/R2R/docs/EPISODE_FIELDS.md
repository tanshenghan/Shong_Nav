# Pointing 补标 episode.json 字段说明

本文解释 `beta_nav_pointing_v1` 补标目录的 `episodes/<id>/episode.json`，
以 `/mnt/cfs/data/beta_nav/r2r_train_470x280_apos10_20260911/episodes/2864/episode.json`
为实际例子。它是轨迹级元数据和统计，逐帧标签在 `samples.jsonl` 和 `targets.npz`。
这些配置是生成时的快照；编辑输出 JSON 不会重新生成标签。

注意：本文 2864 实例是历史 apos-only 版本，尾部缺失 t+10 时被 mask。
当前新版本的 `pointing_config.tail_policy=terminal`，最后十帧使用末帧三维目标，
逐帧新增 `apos.target_world_point`，数组新增 `apos_target_world` 和
`apos_target_frame_id`。历史示例的统计不代表重新生成后的结果。
2026-09-12 起新输出还包含 `pointing_grid`，说明像素与 48×27 网格的尺寸、
行列顺序、从 0 开始的索引和缺失值；逐帧记录新增 `pixel_rc` 与 `grid`。
具体见 [二维像素与具体网格](POINTING.md#二维像素与具体网格)。

当前默认 `pointing_config.opos_mode=terminal_ground`，opos 是实际专家末帧地面坐标，
不依赖豆包识别。新补标 `opos_bound=true`、`opos_target_type=stop_location`，
`object_bound=false` 表示固定终点已确定、没有绑定物体，并非标注失败。
`opos_search=null`、`min_semantic_depth_agreement=null` 表示没有执行 API 搜索或语义回放。
所有逐帧 opos 均有 `target_frame_id`、`target_world_point`，对应数组
`opos_target_frame_id`、`opos_target_world`；像素/网格只在可见时有值。
当前 2864 的固定终点版本有 `opos_valid=52`、`opos_visible=36`：
36 个可见点和 16 个确认不可见标签都可监督。详细定义见 [POINTING.md](POINTING.md)。

## 基本信息

| 字段 | 含义 |
|---|---|
| `complete` | 本条轨迹本次选定的补标流程已完成。true 不表示所有通道都已标注；apos-only 也可完成 |
| `source_episode` | 补标所依据的原始轨迹目录。该例的 RGB、深度、实测位姿、RVQ/FM 来自这个目录 |
| `episode_id` | 源 R2R-VLNCE episode 标识，2864 不是帧号或场景号 |
| `scene_id` | 场景资源的相对路径；该例在 MP3D 场景 29hnd4uzFmX 中 |
| `instruction` | 此 episode 的导航指令；本例为下台阶右转、经过吉他、进入左侧第二个门并在门内停止 |
| `frames` | 已记录的观测帧数，52 表示 frame_id=0..51，包含末端观测，不表示 52 秒 |

## camera：相机及深度定义

| 字段 | 本例数值及含义 |
|---|---|
| `width` | 470，图像宽度，像素 |
| `height` | 280，图像高度，像素 |
| `hfov_deg` | 120，水平视场角，度 |
| `position_body_native` | [0,0.88,0]，相机相对机体的安装平移，米；Habitat 本地 +X 向右、+Y 向上、+Z 向后 |
| `orientation_euler` | [0,0,0]，传感器局部欧拉旋转参数，弧度；全零表示没有额外安装旋转，仍随机器人转动 |
| `K` | 3×3 针孔内参矩阵，fx=fy≈135.6773 像素，cx=235、cy=140 像素 |
| `min_depth_m` | 0，保存深度的下限，米 |
| `max_depth_m` | 10，保存深度的上限，米 |
| `depth_unit` | metre，深度使用米 |
| `depth_type` | camera_z，沿相机光轴的 Z 深度，区别于到相机中心的欧氏距离 |
| `depth_normalized` | false，深度未归一化到 [0,1]，值 2 表示 2 米 |
| `depth_clipped_to_range` | true，保存值裁剪在 0..10 米范围内 |
| `invalid_depth` | 必须读取每个深度 NPZ 的 valid mask；当前仅 0<depth_m<10 的像素有效 |

K 的排列为 `[[fx,0,cx],[0,fy,cy],[0,0,1]]`。它与 `world_from_camera_cv`
配合做投影，后者每帧变化，在源轨迹的 `frames.jsonl` 和 `targets.npz` 中。
camera 中的安装位置是固定的相对位置，不是机器人的世界位置。

## pointing_config：生成标签使用的参数

| 字段 | 本例数值及含义 |
|---|---|
| `horizon` | 10；apos 首选第 t+10 个记录步的地面点，不是 FM 的 30 步长度或 10 秒 |
| `path_spacing_m` | 0.02 米；目标不可见时，沿未来轨迹反向搜索的插值点间距上限，不改变原始采集步距 |
| `depth_tolerance_m` | 0.05 米；深度可见性判断的绝对容差 |
| `depth_relative_tolerance` | 0.01；随观测深度增加的相对容差，即 1% |
| `min_object_pixels` | 12；物体实例绑定至少需要 12 个框内实例像素，选取 opos 时也要求足够的可靠实例像素 |
| `min_box_coverage` | 0.6；实例绑定时，豆包框内的该实例像素数 / 全图可见的该实例像素数，至少为 60%；不是 IoU 或框内前景占比 |
| `min_detection_confidence` | 0.6；豆包返回的自报置信度至少达到此值才尝试实例绑定，不是经校准的准确率 |

例如观测深度 D=2 m 时，容差是 0.05+0.01×2=0.07 m；apos 候选点的 Z 深度
需满足 Z≤2.07 m，并同时满足在相机前方、图像内且深度有效。
后三个物体参数在本次 apos-only 运行中仅记录默认配置，没有执行物体识别/绑定。

## 标注与验证统计

| 字段 | 本例数值及含义 |
|---|---|
| `object_bound` | false；尚未绑定目标物体实例。本例 object_binding.json 原因是 apos_only_requested |
| `apos_valid` | 33；33 帧具有可用于训练的 apos 标签 |
| `opos_valid` | 0；没有可监督的 opos 帧。完整标注中，有效 opos 包括可见点以及确认不可见的状态 |
| `opos_visible` | 0；没有输出可见目标点的帧。这次是未标注，而不是确认 52 帧都不可见 |
| `min_semantic_depth_agreement` | null；本次没有语义回放，因此未计算此指标 |

逐帧读取本例 `samples.jsonl` 得到：33 帧为精确 t+10 点、9 帧无可见地面候选、
10 帧缺失 t+10，52 帧 opos 均为 unknown。没有 apos 回退成功的样本出现在该例中。

完整 opos 流程的 semantic-depth agreement 对每帧计算：重渲染深度与原始深度
之差不超过 0.02 m + 原始深度 1% 的像素，占比较像素的比例。比较像素为原始深度
valid 且重渲染深度有限值的像素。每帧要求至少 0.995，此字段取所有检查帧的最小值。
它采用的 0.02 m 是重渲染对齐容差，区别于 pointing 遮挡配置中的 0.05 m。

## source_hashes：源文件身份

此对象记录 `source_episode` 中四个文件的 SHA-256，不是补标输出同名文件的哈希。
64 位十六进制字符串用于检查源数据有无变动，不是训练特征。

| 子字段 | 对应源文件的内容 |
|---|---|
| `episode.json` | 源轨迹指令、相机、统计和采集完成信息 |
| `frames.jsonl` | 源逐帧路径、实测位姿及历史版本的回放元数据 |
| `targets.npz` | 源 RVQ/FM 轨迹数组、位姿及 mask |
| `samples.jsonl` | 源逐帧训练索引、动作 token 及有效性信息 |

实现定义见 [annotate_r2r.py](../annotate_r2r.py)、[pointing.py](../pointing.py)，
完整运行说明见 [POINTING.md](POINTING.md)。
