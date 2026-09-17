# 单条 LightNav 实际导航与预测视频

`data_pipeline/rollout_lightnav.py` 将本地 `lightnav-serve --task vln` 与独立
Python 3.9 Habitat-Sim 连接起来，运行一条闭环导航。它与专家 GT 采集是两种入口。

```text
R2R 起点和指令
  → Habitat 当前 RGB
  → LightNav 当前及历史 RGB 推理
  → 原始 pointing / RVQ token 和解码的 10 个轨迹点
  → 选取首个可执行轨迹点，转换为连续速度
  → Habitat 碰撞过滤、移动并获取下一帧
  → 重复至停止或达到步数上限
```

R2R 目标位置仅供测量距离，专家动作、未来图像、深度、位姿、GT pointing 和
豆包结果均不进入模型输入。输入图像通过 WebSocket 发送无损 PNG；服务端沿用
checkpoint 的 resize、历史压缩、导航提示词及 RVQ 解码。此入口不安装 Habitat-Lab，
也不修改共享 Habitat 环境；原生 Habitat-Sim 使用与 Habitat `VelocityAction`
相同的刚体速度积分、navmesh `try_step` 碰撞过滤和停止阈值。

## 运行

先在一个空闲 GPU 上启动本地模型服务，等待 ready 文件出现：

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 .venv/bin/lightnav-serve \
  --task vln --model_path /mnt/cfs/Models/Light --backend vllm_local \
  --host 127.0.0.1 --port 8067 --gpu_memory_utilization 0.35 \
  --max_batch_size 1 --max_new_tokens 64 --ready_file /tmp/lightnav.ready
```

然后在另一终端运行，输出目录必须是新的：

```bash
.venv/bin/python data_pipeline/rollout_lightnav.py \
  --source-episode /mnt/cfs/data/beta_nav/r2r_train_470x280_20260911/episodes/2864 \
  --output /mnt/cfs/data/beta_nav/lightnav_2864_closed_loop_NEW \
  --server ws://127.0.0.1:8067 --model-path /mnt/cfs/Models/Light \
  --gpu-id 3 --max-steps 300 --fps 5
```

`--dry-run` 只验证和打印初始化配置，不创建数据目录、不连接模型、不启动仿真。
服务端必须实际使用 `--model-path` 指定的权重；客户端此参数仅用于记录模型配置。
完成单次任务后可以停止为此次任务启动的模型服务。

| 参数 | 默认值或用途 |
|---|---|
| `--source-episode` | 必填；提供同一条 R2R 的指令、起始观察位姿和相机参数 |
| `--output` | 必填；新目录，保存真实闭环记录和视频 |
| `--server` | `ws://127.0.0.1:8067`，须为 `--task vln` 服务 |
| `--model-path` | `/mnt/cfs/Models/Light`，与服务端权重一致 |
| `--habitat-python` | 当前可用的独立 `habitat-image2sim/bin/python` |
| `--config` | `habitat_server/configs/vlnce_r2r.yaml`；读取相机和连续速度控制配置 |
| `--r2r-data` | `/mnt/cfs/data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz` |
| `--scenes-dir` | `/mnt/cfs/data/habitat/scene_datasets` |
| `--gpu-id` | 3，只选择 Habitat 渲染 GPU；模型 GPU 在服务端选择 |
| `--seed` | 20260912，仿真随机种子 |
| `--max-steps` | 300，最多执行的预测次数；达到上限记录 `max_steps`，不伪造模型 STOP |
| `--fps` | 5，视频播放帧率，与控制周期独立 |
| `--dry-run` | 只检查配置和路径 |

控制周期读取 YAML，当前为 0.1 秒；线速度范围 0～2.5 m/s，角速度范围
±300°/s。模型输出 `[forward_m,left_m,yaw_rad]`，控制器沿用仓库
`select_action_waypoint` 和 `first_waypoint_to_velocity_cmd`。该二自由度控制器
不执行侧向速度；停止由实际线、角速度阈值决定，与模型全部轨迹为零的 STOP
信号分别记录。

## 文件与坐标

- `episode.json`：指令、场景、相机、初始位姿、控制配置、模型配置、结束原因、
  距离、pointing 统计、视频信息及关键源文件哈希。只有全部采集和视频完成才设置
  `complete=true`；不将单条训练集示例作为完整 benchmark 结果。
- `predictions.jsonl`：一行一次真实推理，保存 `raw_text`、`rvq_codes`、
  `waypoints_ego`、实际选择的路点、速度命令、完整服务返回值以及 `apos/opos`。
- `frames.jsonl`、`rgb/*.png`、`depth/*.npz`：每个实际观察的 RGB、米制深度、
  相机/机体位姿、碰撞状态和目标距离；最后一个观察位于最后一次动作之后，
  没有新的预测。因此观察数为推理次数加一。
- `lightnav_navigation.mp4`、`preview.png`：预测视频和第一帧预览。
- `habitat.log`、`rollout_lightnav.snapshot.py`：仿真日志和运行代码快照。

`apos/opos.pixel` 使用原始 470×280 图像坐标；模型输出网格 token 时，它表示
相应格子的中心，不能当成 GT 的精确连续像素。`pixel_rc` 为零起始像素行列，
`grid` 保存零起始网格行列和 `1+row*48+col` 的编号。

模型权重可能输出 `rot_left`、`rot_right`、`stop` 或 `not_visible` 等已有指令。
这些原始预测完整保留、没有空间像素，不强制改造成训练管线定义的地面 apos。
用户的新 GT 标签规则不会自动改变已经训练好的模型行为。

## 视频解读

顶部完整显示 Instruction；RGB 上的蓝色轨迹是当前预测的 10 个路点，绿色
和粉色分别标记模型 apos/opos，侧栏显示预测路径俯视图，底部展示原始 token、
通道状态和目标距离。边界格子标记 `edge`，表示模型可能指向画面边缘或之外。

RGB 中的预测轨迹采用当前相机参数和水平地面假设，`forward_offset=0`，不将
短轨迹人为向远处平移；预测 SE(2) 没有楼梯高度，也没有物体三维几何。俯视图
可以显示 RGB 视野外的短轨迹。末尾仅显示执行后的观察，不沿用上一步预测在
新视角上画线。视频 FPS 是播放速度；推理耗时不代表模拟运动的时间步长。
