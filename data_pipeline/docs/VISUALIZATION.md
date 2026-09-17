# 单条 episode 的 GT 视频

按用户要求，渲染逻辑集中在仓库根目录的
[`visual/render_episode_gt.py`](../../visual/render_episode_gt.py) 一个 Python 文件中。
它独立读取磁盘数据，只需要 NumPy、Pillow、imageio 和 imageio-ffmpeg，
不依赖 Habitat、LightNav 模型或豆包 API。文档、测试和小型验证记录仍在 data_pipeline。

## 视频内容

- 顶部完整显示本条轨迹的 Instruction，按宽度自动换行。
- 将当前帧起点和真实未来 t+1 到 t+10 的机体地面位置连成线，使用当前帧
  `world_from_camera_cv` 与 K 投影到 RGB。位置来自保存的 `world_from_body`，
  保留楼梯的世界高度，不使用 RVQ 解码轨迹代替 GT。
- 蓝色实线表示深度检查可见的轨迹，橙色虚线表示被遮挡或深度无效的部分。
  先裁剪相机近面，再裁剪图像边界，不将相机后方的点强行画入图片。
- 绿色大圆环标 apos，粉色小圆环标 opos；两者重合时同心显示，文字分别放在
  左上与右下，中心保持真实像素，不移动标签。只画已保存且 `state=point`、
  `loss_valid=true` 的像素。不重新计算或伪造标签。
- 底部显示帧号、真实剩余未来步数和两个 pointing 通道的状态。
  unknown、not_visible、unavailable 没有像素标记，状态仍会显示。
- 右侧俯视图显示全部未来路点，当前机体朝前的方向为图中向上，白色三角形为
  当前机体、蓝线为未来路线。按机体 XZ 平面绘制，整段视频保持相同米制比例；
  RGB 视野外的路线也能在这里查看。侧栏不覆盖原始 RGB 或 pointing 标记。
- 原地旋转没有地面位移；最后一帧没有未来位置。此时不造轨迹线。
  尾部只画实际存在的步数，不重复填充末帧以凑满 10 步。

默认以 5 FPS 播放、将 470×280 RGB 等比例放大到 940×560，再增加顶部指令栏和
底部说明栏以及右侧俯视图。放大用于便于查看，不增加原图细节。FPS 只控制播放速度，不能解释为
R2R 专家动作的物理时间。输出采用 H.264 / yuv420p / faststart，便于播放器打开。

## 运行

在仓库根目录使用已有 `.venv`：

```bash
# 直接读取带 apos/opos 的新版 episode；它会引用原始 RGB、深度和位姿
.venv/bin/python visual/render_episode_gt.py \
  /mnt/cfs/data/beta_nav/r2r_2864_gt_20260911_v2/episodes/2864 \
  --output /mnt/cfs/data/beta_nav/visual/episode_2864_NEW.mp4

# 也可以从原始 episode 出发，明确指定对应标签目录
.venv/bin/python visual/render_episode_gt.py \
  /mnt/cfs/data/beta_nav/r2r_train_470x280_20260911/episodes/2864 \
  --annotations /mnt/cfs/data/beta_nav/r2r_2864_gt_20260911_v2/episodes/2864 \
  --output /mnt/cfs/data/beta_nav/visual/episode_2864_NEW.mp4 \
  --fps 5 --horizon 10 --scale 2
```

原始 episode 未指定 annotations 时，只绘制原始数据已有的标签；没有 apos/opos
就显示 unavailable。脚本不会猜测应该选哪个补标版本。
输出文件存在时拒绝覆盖。对原始数据和标签只做读取。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `episode` | 必填 | 原始/补标的单条 episode 目录；指定 `--episode-id` 后为 LeRobot 数据集根目录 |
| `--episode-id` | 无 | 从 image-backed BetaNav LeRobot v3 中读取原始 episode ID，不是内部 episode_index |
| `--scene` | 无 | 配合 `--episode-id` 指定 Scan；不能用于原始 episode 输入 |
| `--annotations` | 无 | 为原始输入指定同一条轨迹的标注目录 |
| `--output` | 必填 | `.mp4` 输出视频及预览，`.png` 输出网格示意图；两种模式均保存 JSON |
| `--fps` | 5 | 视频播放帧率，0 < FPS ≤ 60 |
| `--horizon` | 10 | 未来轨迹步数，范围 1～30；不改变已有 apos 的定义 |
| `--scale` | 2 | RGB 整数显示倍率，范围 1～4 |
| `--rgb-only` | 关闭 | MP4 只输出 RGB 叠加画面；配合 `--scale 1` 保持原生 470×280，不添加指令栏或俯视图侧栏 |
| `--font` | DejaVuSans | 可选 TrueType 字体文件；其他语言需使用支持相应字符的字体 |
| `--frame-id` | PNG 默认为 0 | 网格图的帧号；视频模式使用全部帧，不接受该参数 |
| `--channel` | apos | PNG 示意图选择 apos 或 opos |
| `--hide-occluded` | 关闭 | RGB 上只画深度检查可见的 GT 线；右侧仍显示全部实际未来路径 |
| `--number-steps` | 关闭 | 对 RGB 中可见的未来地面位置编号，重合位置合并显示步号范围 |
| `--preview-frame` | 0 | MP4 附带 PNG 海报使用哪一帧，视频仍编码所有源帧 |

## 直接读取 LeRobot Parquet

新增读取方式支持 BetaNav 的 image-backed LeRobotDataset v3.0。
直接读取 Parquet 内嵌 PNG 字节、float32 米制深度、深度有效 mask、相机与机体矩阵以及 apos/opos。
不读取已经清理的采集 scratch，不安装或调用 Habitat、模型、LeRobot/Torch 运行时。
需要 numpy、Pillow、imageio、imageio-ffmpeg、pyarrow、ijson；当前独立训练环境已具备这些读取依赖。

```bash
/mnt/pfs/tanshenghan/vln4project/LightNav-0/.venv-train/bin/python \
  /mnt/pfs/tanshenghan/vln4project/LightNav-0/visual/render_episode_gt.py \
  /mnt/cfs/data/beta_nav/lerobot_r2r_train_470x280_20260912 \
  --episode-id 515 --scene 17DRP5sb8fy \
  --output /mnt/pfs/tanshenghan/vln4project/data_web/runtime/lerobot-preview-20260917/r2r/train/17DRP5sb8fy/episode_515/gt10_apos_opos_470x280.mp4 \
  --fps 5 --horizon 10 --scale 1 --rgb-only \
  --hide-occluded --number-steps --preview-frame 24
```

这个输出路径是 2026-09-17 的实际预览，重跑时请使用新的输出文件名。
该预览为 41 帧、8.2 秒、严格 470×280；每帧在原图尺寸上叠加标记。
指令、图例及状态由外部播放器展示；完整未来坐标保存在 JSON 中。
不传 `--rgb-only` 时，仍可使用原有带指令栏和俯视图的布局。
`--episode-id` 与 `--annotations` 不能同时使用，LeRobot 标签已经与原始观测保存在同一行。
也可使用 `.png` 输出和 `--frame-id` 检查 LeRobot 单帧的原始像素/网格。

读取时先用 ijson 流式定位大型 `meta/beta_nav.json` 的 episode 条目，
再查 `meta/episodes/chunk-*/file-*.parquet` 的帧范围与首个数据文件。
按 `episode_index` 筛选对应的数据 Parquet；episode 跨文件时继续读取下一个文件，
只在当前 episode 内按需解码 RGB/深度。源 episode ID、Scan、指令、帧号、全局索引范围均需一致。
网站的全量入口 `data_web/render_vln_video_collection.py` 复用压缩的元数据索引，调用
`load_lerobot_episode(..., metadata=...)`，避免逐条重扫大型 manifest；该参数仍检查 episode ID/Scan。
命令行不传缓存时仍使用流式定位。深度和 mask 从 Arrow 数组直接读取，保持 float32/bool，
每个 ffmpeg 固定一个编码线程，便于有界并行；不降低逐帧哈希与几何校验要求。
本文件的单条 CLI 本身不批处理、不上传。

渲染前核验 clean FM 的有效 mask、future_frame_ids、局部前/左坐标及垂直偏移与真实未来机体位置一致；
有效 apos/opos 的世界坐标重投影需与保存的像素一致。渲染每帧时将解码 RGB、深度和 mask 的
SHA-256 与采集元数据比对。MP4 编码完成后检查实际帧数。
所有未来点都来自实际记录，尾部不足 10 步时按实际步数显示。

opos 为固定末帧地面位置，保留数据集的 last_half 与深度可见性规则。
`state=not_visible, valid_mask=true, token=0` 是有效的不可见类别，不是可绘制的空间点。
`state=unknown` 与它保持区分；两者都不会伪造像素位置。
JSON 同时保存每帧真实未来帧号、世界坐标、pointing 状态/原因、相机内参与 Parquet 来源。
float32 像素在图像边缘可能舍入到 width/height；一个 float32 ULP 内的边界点不钳入画面，
绘制状态记录为 `boundary_not_drawn`，原始标签保持不变。明显越界的有效像素仍拒绝渲染。

早期 5 FPS 预览保存在工作区 runtime。确认效果后，主网站已改用 8 FPS、470×280 的成片。
当前全量 R2R/RxR 成片使用以下独立衍生目录，不修改训练 Parquet 或 `meta/info.json`：

```text
/mnt/cfs/data/beta_nav/visual/vln_gt_videos_8fps_v1/
└── r2r/train/guide/17DRP5sb8fy/episode_515/
    ├── gt10_apos_opos.mp4
    ├── gt10_apos_opos.png
    └── gt10_apos_opos.json
```

其他 episode 按 dataset/split/role/Scan/原始 ID 分目录。
网站通过 `data_web/runtime/vln-videos-all.sqlite` 匹配 71,071 条观测：R2R train 10,819 条、
RxR train guide 60,252 条（1,077 个验证完成的 shard）。后台任务使用 8 个 CPU worker 全量渲染，
已完成数量以网页进度为准；不能据此声称 val/test/follower 已有视频。
完全相同的观测、位姿和标签哈希可复用 MP4/PNG；各指令仍保留独立来源 JSON 并明确记录复用关系。
早期源目录下的 R2R review 视频保留作历史记录，主站使用新的统一目录。
页面显示“LeRobot GT 视频”，放在俯视图上方；旧模型回放不再展示。
详细的网站存储与发布约定见
[LeRobot 视频说明](/mnt/pfs/tanshenghan/vln4project/data_web/docs/lerobot_video.md)。

## 单帧像素与网格示意图

同一个代码文件支持 PNG 输出：叠加 48×27 网格和从 0 开始的行列编号，
高亮所选 apos/opos 的空间格子，以白点显示精确像素，并提供局部放大图。
右侧列出原图像素、整数像素行列、网格行列、格子编号及 token。
未知标签会显示没有有效空间格子；不会为其补造位置。

```bash
.venv/bin/python visual/render_episode_gt.py \
  /mnt/cfs/data/beta_nav/r2r_2864_gt_grid_20260912/episodes/2864 \
  --output /mnt/cfs/data/beta_nav/visual/episode_2864_grid_NEW.png \
  --frame-id 0 --channel apos --scale 2
```

PNG 模式不使用 `--fps` 和 `--horizon`。网格位置以原始 RGB 为准，
显示放大与图片外边距不改变标签坐标。已有标签记录网格或 token 时，
绘图前检查它们与精确像素一致。
本次图片位于 `/mnt/cfs/data/beta_nav/visual/episode_2864_frame_000000_apos_grid_20260912.png`。

## 输出和检查

- `.mp4`：每个源观测对应一帧视频。
- `.png`：`--preview-frame` 指定帧的无损预览，默认第一帧。
- `.json`：输入路径、完整 Instruction、帧数、播放帧率、画面大小、代码/数据
  哈希以及每帧轨迹和 pointing 状态。

程序先核对源 episode、标签、相机和帧索引，检查补标记录的源文件哈希；
编码后再用 ffmpeg 核对总帧数。CPU 测试覆盖坐标变换、近面裁剪、遮挡、
尾部不足步数、有效/无效标记、错位标签拒绝及实际 MP4 编码。

历史 episode 2864 的视频（使用当时的豆包标签）：
`/mnt/cfs/data/beta_nav/visual/episode_2864_gt10_topview_20260911.mp4`。
原始数据为 52 帧；当时的标签有 38 个有效 apos，opos 全部 unknown，因此视频中
不会出现粉色物体点。这是本条数据的真实标注状态。

当前默认固定终点 opos 的视频为
`/mnt/cfs/data/beta_nav/visual/episode_2864_gt_opos_endpoint_20260912.mp4`。
使用 `r2r_2864_opos_endpoint_20260912/episodes/2864` 的新标签，38 帧有 apos，
第 11–46 帧共 36 帧有粉色 opos。opos 在所有帧对应同一个最终目的地。
第 44 帧网格图为
`/mnt/cfs/data/beta_nav/visual/episode_2864_frame_000044_opos_endpoint_grid_20260912.png`。

重合标记修正版视频为
`/mnt/cfs/data/beta_nav/visual/episode_2864_gt_opos_endpoint_rings_20260912.mp4`。
第 41–46 帧的 apos/opos 像素完全相同，旧视频后画的粉色圆点覆盖了绿色圆点；
现在以绿色外环、粉色内环同时显示。第 2–10、47–51 帧则是未来地面路线确实在
画面外，没有有效 apos 像素。逐帧标签保持不变，检查记录见
[缺失与重合诊断](../reports/episode_2864_apos_missing_and_overlap_20260912.json)。
