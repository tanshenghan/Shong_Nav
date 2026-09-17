# 新版 apos / opos 数据管线

本方案按用户最新定义实现。采样仍沿用内部 R2R-VLNCE 专家动作回放；新采集和
补标样本不输出前进/左转/右转标签。实际位姿仍包含转向，RVQ/FM 的 yaw 不删除。
原有 RVQ 终止监督保留；apos 不输出转向或 STOP token。

## apos：第 10 步地面点

实现：[pointing.py](../pointing.py) 的 `ground_point()`。

1. 以新回放机体原点（脚底/navmesh 地面位置）`position[min(t+10, N-1)]` 为目标，保留实际
   世界高度，不把楼梯统一压到 y=0。相机高度不加到目标点上。
2. 用当前 `world_from_camera_cv` 的逆变换和 K 投影。
3. 要求 Z>0、在画面内、对应深度有效，且目标的 camera-Z 不大于观测深度加容差。
   默认容差为 0.05 m + 观测深度的 1%。
4. 若第 10 步不可见，沿 t+10 到 t 的未来轨迹反向搜索，每段插值间距不大于
   0.02 m。选择遇到的第一个可见点，即沿路径距离最接近原第 10 步目标的可见点。
   不选择任意最近的图像像素，也不把越界点裁剪到图像边缘。
5. 最后十帧不存在 t+10 时，统一使用末帧三维地面坐标作为目标，仍执行上述投影和
   遮挡检查。整段没有可见候选时，像素标签为 unknown、loss mask=false，但保留
   `target_world_point`，不会丢失末帧三维目标，也不会伪造图像内的像素。
   此时 token_id=0，保存 `<apos_0>`，loss mask 保持 false。

每帧保留 `target_frame_id`、`target_world_point`；可见点还保留
`selected_frame_fraction`、`backtrack_m`。`terminal_ground_point` 表示直接使用末帧，
`t_plus_10` 表示直接使用第十步，`visible_path_backtrack` 表示遮挡回退。
`world_point` 是实际可见标记点，回退时可能位于两帧之间，与三维目标不同。
默认 `tail_policy=terminal`；旧数据未记录此字段时，验证器按旧版 `mask` 解释，
不更改历史数据的含义。此规则只改变 apos，RVQ/FM 的尾部有效 mask 不变。

这里的近处回退候选只来自当前到目标之间的 **GT 路线**。画面中存在其他可见地面，
不代表这段未来路线上的地面点也可见。第 2～10 帧的逐点/插值检查，以及整段线段
与相机视野的相交检查，见 [2864 近点报告](../reports/apos_2864_nearby_visibility_20260912.json)。
该例这九帧的候选均在视野外，因此没有可写入有效图像格子的 apos。

## opos：整条轨迹最终要到达的地方（当前默认）

`opos_mode=terminal_ground`，实现为 [pointing.py](../pointing.py) 的
`terminal_ground_point()`。设一条轨迹有 N 帧，第 t 帧使用的目标始终是：

```text
opos_target_world(t) = world_from_body[N-1][:3, 3]
opos_target_frame_id(t) = N-1
```

这是用 LightNav 相机配置重新回放后记录的专家末帧地面坐标，保留其真实世界高度。
它不同于 R2R 元数据中的原始目标坐标，也不是相机中心、物体表面或识别框中心。
apos 表示随时间推进的局部目标；opos 表示固定的最终目的地。

默认 `opos_visibility_window=last_half`：只有整条 episode 的最后二分之一
允许出现空间 opos 标记。帧号从 0 开始，起始帧为 `floor(N/2)`（包含该帧），
即允许区间共有 `ceil(N/2)` 帧；只按采样帧数计算，不依赖实际时间。

1. 起始帧之前统一输出 `not_visible`、`<opos_0>`、`loss_valid=true`，
   `reason=before_opos_visibility_window`，像素和网格为空。这是人为限制的标注窗口，
   不表示终点在真实画面中一定被遮挡。
2. 进入窗口后，用保存的 `world_from_camera_cv` 和 K 投影同一个三维终点。
3. 投影在画面内、camera-Z>0、深度有效且无遮挡时，保存精确 `pixel`、
   `pixel_rc`、48×27 `grid`、`world_point` 和空间 token，`loss_valid=true`。
4. 在视野外、相机后方或被有效深度遮挡时，标为 `not_visible`、`<opos_0>`，
   `loss_valid=true`，没有二维坐标或空间格子。
5. 投影处深度无效时为 `unknown`、`token_id=0`、`<opos_0>`、`loss_valid=false`。
6. **所有帧**均保存 `target_world_point` 和 `target_frame_id`，包括上述不可见和未知帧。
   opos 不沿路径回退到别处；只有 apos 使用可见路径回退。

采集器与离线补标默认都执行此规则。无需 API、识别框、语义资产或语义重渲染；
补标可直接使用 CPU。末帧终点在相机脚下时可能不在画面内，三维目标依然存在。
`object_binding.json` 保留兼容文件名，保存 `target_type=stop_location`、
`model_kind=recorded_terminal_ground_point`、固定坐标和末帧编号。
`opos_bound=true`、`object_bound=false` 表示终点已确定但没有绑定物体实例。
`episode.json` 中 `opos_search=null`，不生成 API 识别/搜索文件；
`manifest.json` 中 `api_requested=false`。

旧配置缺少 opos_mode 时仍按历史语义验证。旧版豆包识别、实例建模与末八帧搜索
已从当前生成代码移除，代码与说明保存在 [清理前归档](../archive/README.md)。
新的采集和补标只允许 terminal_ground + terminal + unknown_token_id=0。
旧数据未记录 unknown_token_id 时，验证器保留其 -1/null 规则。
旧数据未记录 opos_visibility_window 时，按 `all` 保留原先整条轨迹的可见性判断。
新输出的 `episode.json` 同时记录 `opos_visibility_start_frame`，便于核对边界。

60 帧 episode 5670 的当前标注从帧 30 开始开放；帧 0–29 为窗口限制的 0，
帧 30–55 实际可见，帧 56–59 终点在视野外。apos、FM、RVQ 保持原定义。
旧 `last_third` 仍可只读解释已有标注；它的边界为 `floor(2*N/3)`。

## 输出格式

轨迹级 `episode.json` 的逐字段解释和 2864 实例见 [EPISODE_FIELDS.md](EPISODE_FIELDS.md)。

新采集为 `beta_nav_v2`，默认采集结束即可得到 apos 和固定终点 opos。
已有 v1 数据也可直接补标，不需要重新采集 RGB/深度。

```text
NEW_POINTING_RUN/
  manifest.json                    源数据/代码哈希、当前参数、api_requested=false
  summary.json                     全部选定轨迹处理完才生成
  failure.json                     失败时生成；不能据此当作整批完成
  provenance/                      管线代码和 pointing 配置快照；不包含 config.py
  episodes/<episode_id>/
    episode.json                   指令、相机、源路径/哈希、有效点数
    samples.jsonl                  干净的逐帧训练索引、RVQ 与 apos/opos token
    targets.npz                    RVQ/FM 数组及新增 pointing 数组
    object_binding.json            固定终点及来源，保留兼容文件名
    preview_000000.png             首/中/末帧的点位预览，绿=apos、粉=opos
```

RGB/深度通过样本中的绝对路径引用原始采集，不重复复制。`targets.npz` 的 RVQ/FM
数组来自源数据，新增 `apos_pixel/opos_pixel` (N,2)、`apos_world/opos_world` (N,3)、
`apos_token_id/opos_token_id` (N,) 和 `apos_valid_mask/opos_valid_mask` (N,)。
新增 `apos_target_world` (N,3，float64，Habitat 世界坐标米) 和
`apos_target_frame_id` (N,)，保存投影/可见性筛选前的目标；最后十行都指向末帧。
即使 `apos_valid_mask=false`，这两个目标数组仍有真实值。
默认终点模式还保存 `opos_target_world` (N,3，float64) 和
`opos_target_frame_id` (N,，int32)，每行分别为相同的末帧坐标与 N-1。
这两个字段不受二维可见性影响；`opos_world` 仅在可见时有值。

有效点编码为 48×27 网格的 1..1296；opos 确认不可见为 0（有效监督）。
apos/opos 未知也使用 0，并保存 `<apos_0>` / `<opos_0>`，但 mask=false。
因此须结合 state/mask 区分未知与确认不可见，不能只由 token 0 推断监督有效性。
缺失坐标在数组中保存 NaN，在 JSON 中保存 null；缺失整数空间索引仍为 -1。
训练视觉历史只能读当前及过去 RGB；末尾识别结果、物体模型和未来位姿是监督侧信息。
目前不构建训练 prompt，也不会把完整轨迹最后一帧放进早期时刻的视觉输入。

## 二维像素与具体网格

从 2026-09-12 起，新采集/补标自动显式保存像素和网格索引。所有坐标以原始
470×280 RGB 为准，不包含视频的放大、顶部文字栏或侧边俯视图。

每帧的 `apos` / `opos` 中保存：

| 字段 | 含义 |
|---|---|
| `pixel` | 精确浮点像素 `[u,v]`，即 `[x,y]`，保持原值 |
| `pixel_rc` | 原图像素 `[row,col] = [floor(v),floor(u)]`，从 0 开始 |
| `grid.row` | 模型 48×27 网格中的行，范围 0～26 |
| `grid.col` | 网格中的列，范围 0～47 |
| `grid.cell_id` | 空间格子编号，`1 + row*48 + col`，范围 1～1296，与有效点 token_id 相同 |

网格换算为 `col=floor(u/width*48)`、`row=floor(v/height*27)`，边缘处理与
现有模型编码器保持一致。图像像素行列和模型网格行列是两种不同分辨率的索引。
`episode.json` 的 `pointing_grid` 保存原图尺寸、网格尺寸、索引顺序和缺失值约定。

真实例子：2864 的第 0 帧 apos 的像素为 `[250.2726976,257.7736847]`，
`pixel_rc=[257,250]`，网格 `row=24,col=25,cell_id=1178`，对应 `<apos_1178>`。
这里行列从 0 开始；按口语从 1 数，是第 25 行、第 26 列。

`targets.npz` 同时新增以下整型数组，apos/opos 各一组：

- `apos_pixel_rc` / `opos_pixel_rc`：`[N,2]`，原图整数像素行列。
- `apos_grid_rc` / `opos_grid_rc`：`[N,2]`，模型网格行列。
- `apos_grid_id` / `opos_grid_id`：`[N]`，空间格子编号。

没有有效空间点时，JSON 中 `pixel_rc` 和 `grid` 为 null，新增数组用 -1。
确认不可见的 opos 仍保留 token_id=0 及其有效分类监督，但 **没有空间格子**，
所以 grid_id=-1。缺失标签不会被塞进左上角或图像边缘的格子。

已保存的 pointing 数据可以离线补上这些字段，不调用 API、不重采集：

```bash
.venv/bin/python data_pipeline/export_pointing_grid.py \
  /mnt/cfs/data/beta_nav/r2r_2864_gt_20260911_v2 \
  --output-dir /mnt/cfs/data/beta_nav/r2r_2864_gt_grid_NEW \
  --episode-id 2864
```

参数 `dataset` 是已有 pointing 数据集根目录；`--output-dir` 必须是新目录；
`--episode-id` 可选，省略时处理输入中的全部 episode。
该入口还生成每条 episode 的 `pointing_pixels.csv`，便于直接查看每帧的像素和格子。
原始 RVQ/FM、像素、三维坐标、token、有效 mask 和豆包记录不变；
旧 provenance 保存在新输出的 `provenance/source_run`，源目录不会改写。

本次结果为 `/mnt/cfs/data/beta_nav/r2r_2864_gt_grid_20260912/episodes/2864`。

## 命令

当前补标只使用 NumPy、Pillow 和保存的几何数据，直接使用 CPU；不读取本机 API 配置。

```bash
.venv/bin/python data_pipeline/annotate_r2r.py \
  /mnt/cfs/data/beta_nav/r2r_train_470x280_20260911 \
  --output-dir /mnt/cfs/data/beta_nav/r2r_pointing_NEW_RUN \
  --episode-id 2864 --episodes 1

.venv/bin/python data_pipeline/validate_r2r.py /mnt/cfs/data/beta_nav/r2r_pointing_NEW_RUN
```

| 参数 | 默认 | 含义 |
|---|---|---|
| dataset | 必填 | 原始完整采集目录，须有 frames.jsonl |
| --output-dir | 必填 | 新的补标目录，拒绝覆盖 |
| --config | data_pipeline/configs/pointing.json | 当前点位几何配置 |
| --episodes | 10 | 处理数量，-1 为全部 |
| --episode-id | 无 | 仅处理指定 episode |
| --dry-run | 关闭 | 检查输入、打印配置，不写数据 |

配置固定 horizon=10、tail_policy=terminal、opos_mode=terminal_ground；
path_spacing_m 控制 apos 路径插值，depth_tolerance_m 和 depth_relative_tolerance
控制遮挡容差。旧 API、语义场景、GPU 和 apos-only 参数已移除。
opos_visibility_window 默认 last_half；last_third 为旧的末三分之一窗口，
all 表示逐帧按几何判断，不限制时间窗口。
原始采集使用同一配置，参数为 --pointing-config。

历史数据仍可通过 validate_r2r.py 只读验证；旧 API/语义实现与当时文档留在归档中。
