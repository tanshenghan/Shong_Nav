# 历史数据与生成记录

这些记录对应各次生成时的代码和定义。当前入口见 [README](../README.md)。


2026-09-12 按“opos 是最终要去的地方”重新生成 2864，当前版本为
`/mnt/cfs/data/beta_nav/r2r_2864_opos_endpoint_20260912/episodes/2864`。
52 帧均保留末帧 51 的固定三维地面目标；第 11–46 帧有可见 opos，共 36 帧，
其余 16 帧为 `<opos_0>`。apos 仍有 38 个有效点，21 个非 opos 数组与上一版完全一致。
本次不调用 API；几何验证及 70 项管线测试通过，详见
[固定终点报告](../reports/episode_2864_opos_endpoint_20260912.json)。
视频：`/mnt/cfs/data/beta_nav/visual/episode_2864_gt_opos_endpoint_20260912.mp4`，
52 帧、5 FPS、10.4 秒；第 44 帧像素格子示意图为
`/mnt/cfs/data/beta_nav/visual/episode_2864_frame_000044_opos_endpoint_grid_20260912.png`。

以下倒数八帧结果使用历史豆包语义，不是当前默认的终点 opos。
2026-09-12 按倒数八帧顺序完成 2864 的 opos 重标，结果位于
`/mnt/cfs/data/beta_nav/r2r_2864_opos_tail8_20260912_v2`。已实际检查 44→51 帧：
第 44 帧识别区域不包含 GT 终点；47、49、50 帧的真实终点在视野外；
其余帧未可靠识别。最终未绑定目标，52 帧 opos 仍为 unknown/mask=false。
全部候选 API 响应、拒绝原因、像素 CSV 和本地对照图已保存；apos、RVQ/FM
及位姿数组保持原值。详见 [末八帧重标报告](../reports/episode_2864_opos_tail8_20260912.json)。
不带 `_v2` 的目录保留首次坐标格式校验失败的记录，不作为完整数据。

真实模型导航另用 [LIGHTNAV_ROLLOUT.md](LIGHTNAV_ROLLOUT.md) 的入口：
LightNav 根据每步实际观察预测 RVQ 轨迹并驱动 Habitat 连续速度控制，保存新的
RGB/深度/位姿、模型原始 token、预测 apos/opos 及可视化视频。该输出是模型
rollout，不作为专家运动 GT；已有训练标签规则不改变发布权重的原始预测语义。

2026-09-12 已实际运行 episode 2864：
`/mnt/cfs/data/beta_nav/lightnav_2864_closed_loop_20260912`。
61 次模型预测、62 个实际观察，模型主动 STOP，终点距 R2R 目标约 0.344 m。
模型给出 60 个空间 apos 和 1 个 apos STOP；opos 为 29 个空间点、32 次不可见。
预测视频为目录中的 `lightnav_navigation.mp4`（5 FPS，12.4 秒）。
全部 RVQ 解码、控制旋转、相机位姿、RGB/深度和视频帧检查通过，详见
[真实模型运行报告](../reports/episode_2864_lightnav_rollout_20260912.json)。

2026-09-12 为 2864 补充了明确的二维像素和网格地址，新目录为
`/mnt/cfs/data/beta_nav/r2r_2864_gt_grid_20260912/episodes/2864`。
每个有效 apos/opos 保存 `pixel`、`pixel_rc` 和 `grid`（行、列、格子编号）；
JSON、NPZ 和 `pointing_pixels.csv` 均可读取。52 帧中仍有 38 帧有效 apos，
几何标签保持原值，未再次请求豆包。字段与离线升级命令见
[二维像素与具体网格](POINTING.md#二维像素与具体网格)。

2026-09-11 单条新版 GT 位于
`/mnt/cfs/data/beta_nav/r2r_2864_gt_20260911_v2/episodes/2864`：52 帧，38 个有效 apos，
最后十帧三维目标全部指向 frame 51。保留 42 个完整 RVQ-10 chunk、1 个显式 STOP
及 22 个完整 FM-30 chunk，其余未来步按 mask 使用。
真实豆包返回 `found=false`：指令指定的左侧第二个门在末帧不可见，因此 52 帧
opos 均为 unknown，未生成物体网格。该结果不能用于正/负 opos 监督。
详见 [本次生成报告](../reports/episode_2864_gt_20260911.json) 和
[逐帧验证](../reports/episode_2864_gt_validation_20260911.json)。
不带 `_v2` 的同名单条目录保留首次请求超时的失败记录，不作为训练数据。

以下为历史 v1 采集。新版 apos 补标位于
`/mnt/cfs/data/beta_nav/r2r_train_470x280_apos10_20260911`，506 帧中有 364 个有效 apos；
其 opos 因该次明确使用 `--apos-only` 而保持 unknown，尾部沿用当时的 mask 规则。
当前新数据采用末帧三维目标补齐 apos，opos 直接使用固定专家终点。
仅旧版豆包模式从本机 `config.py` 读取 API 配置。

数据目录：`/mnt/cfs/data/beta_nav/r2r_train_470x280_20260911`。

- 10 条轨迹、506 帧 RGB/深度/位姿。
- 406 个完整 RVQ motion chunk，另有 10 个经验证的 STOP 标签。
- 206 个完整 FM 30 步 chunk，尾部仍保留部分有效监督。
- 50 对转向帧的深度重投影误差中位数约 2.95 mm。
- 新位姿与 GT 路线最大偏差约 5.8e-8 m。
- 本采样定义下 RVQ 重建平均 ADE 约 9.84 cm，不能作为官方编码器精度报告。

验证细节见 [首批报告](../reports/pilot_20260911.json)。后续变更在本目录更新范式、
代码与说明，生成的新数据写入独立版本目录。
