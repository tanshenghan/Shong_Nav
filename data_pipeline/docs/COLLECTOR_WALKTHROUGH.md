# 当前采集与补标流程

`collect_r2r.py` 采集并生成运动、apos 与最终目的地 opos 标签；
`annotate_r2r.py` 为已有数据生成符合当前规则的 apos/opos 补标版本。
具体标注参数和格式见 [POINTING.md](POINTING.md)。

## collect_r2r.py

1. `main()` 读取 R2R-VLNCE episode、专家动作、参考路径、LightNav 相机配置及 RVQ 码本。
2. `select_episodes()` 按场景/物理路线去重，每条路线选一条指令。
3. `simulator_config()` 读取相机和运动参数，覆盖分辨率为 470×280，关闭深度归一化。
4. `Renderer` 加载已有场景和 navmesh，创建 RGB/深度传感器。
5. `collect_episode()` 从 R2R 起点开始，内部执行专家动作。每次执行动作之前，
   `capture()` 保存当前 RGB、米制深度及仿真器实测的机体/相机位姿。
   新版 frames/samples 不再包含前进/左转/右转类别标签。
6. 回放后检查与 GT 路线的最大偏差不超过 0.5 m，最终到目标的测地距离小于 3 m，
   并要求源专家序列有且仅有末端 STOP。
7. `export_targets()` 用实际未来位姿生成 FM-30 连续运动增量和 RVQ-10 码本索引。
   第 t 帧的未来从 t+1 开始；尾部未知未来按各自的 mask 屏蔽。
8. 同时调用 `ground_point()` 投影 min(t+10, N-1) 的地面点，必要时沿未来轨迹
   反向寻找可见点。尾部目标固定为末帧三维坐标；没有可见点时像素 mask=false，
   仍保存 target_world_point。不生成转向/STOP apos token。
9. 默认 `opos_mode=terminal_ground`：`terminal_ground_point()` 把实际末帧的
   地面坐标作为固定最终目的地，用每帧相机和深度检查投影与遮挡。opos 不回退，
   可见时保存像素/格子，不可见时输出 `<opos_0>`，深度无效时 mask=false；
   所有帧均保存同一个 `target_world_point` 和末帧 `target_frame_id`。
10. 保存全部标签数组、样本索引、相机和采样约定；apos/opos 在采集结束时一起完成。

LightNav 提供仿真相机和运动配置，没有调用 LightNav 神经模型生成运动 GT。
深度用于记录、验证和 pointing 遮挡检查；RVQ/FM 从仿真器位姿直接计算。
内部仍使用离散专家动作回放，采样步没有标定物理秒数；取消输出标签不会自动把
回放改成连续路径重采样。

## annotate_r2r.py

默认 CPU 补标不调用 API，也不启动 Habitat：

1. 读取原始完整轨迹的 RGB/深度路径、实际相机和机体位姿、RVQ/FM 数组。
2. 以最后一帧机体地面坐标固定整条轨迹的 opos；不从物体框选目标。
3. `ground_point()` 计算 apos，`terminal_ground_point()` 计算 opos，
   再保存原图像素和 48×27 网格索引。
4. 写入新的版本目录，保存源哈希、参数、目标绑定和逐帧记录；RGB/深度引用源文件。
   `opos_target_world` 和 `opos_target_frame_id` 在所有帧都保存最终目标，
   即使该帧不能得到二维空间标记。

补标入口只支持当前最终定义；旧 API/物体建模代码已归档。

## 如何检查

`validate_r2r.py` 自动识别普通采集和 pointing 补标格式。它会检查源文件哈希、
帧索引、未来步、相机几何、pointing mask、像素与世界点投影，以及物体可见区域。
历史物体标签仍可读取原来保存的实例 mask 进行验证，无需 API 或语义重渲染。

[原始 LightNav 论文](../../docs/LightNav-0.pdf) 提供双通道 pointing 的总体思路。
当前第 10 步 apos 和固定最终目的地 opos 是按用户定义实现的数据范式，
不声称逐项复现未公开的官方标注程序。
