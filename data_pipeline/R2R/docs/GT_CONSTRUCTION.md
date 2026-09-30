# 当前四类 GT 的构造与读取

数值示例来自历史数据 `/mnt/cfs/data/beta_nav/r2r_2864_opos_endpoint_20260912/episodes/2864`，
52 帧，默认 `opos_mode=terminal_ground`。这里解释专家 GT；真实 LightNav 模型
闭环预测另见 [LIGHTNAV_ROLLOUT.md](LIGHTNAV_ROLLOUT.md)，两类数据不混用。
该原始数据集目录目前已被清空，保留的 reports 仅供数字示例查看。
下述字段定义已更新为 pointing_zero；旧报告的 -1/null 缺失 token 属于历史编码。
当前 opos 还限制为最后二分之一才允许空间标记；下方历史帧 20 的点位和统计
保留原始含义，不代表这个新窗口策略。新版 5670 示例见 [README](../README.md)。

## 从原始路线到一条训练样本

R2R 提供场景、起点、指令及专家路线/动作。用 LightNav 的相机与运动配置在
Habitat-Sim 中重新回放，保存每步新 RGB、深度、机体与相机位姿。当前 RGB 为
470×280，水平 FOV 120°，相机高 0.88 m、俯仰 0°，深度是米制 camera-Z。
运动 GT 来自这次回放的实际位姿；投影和可见性使用这次回放的相机与深度。

第 t 行对应当前第 t 帧，未来从 t+1 开始，形成四类标签：

```text
实际未来位姿 t+1 ... t+30
  ├─ 全部最多 30 步 → FM 真值路点、逐步运动增量与逐步 mask
  ├─ 前 10 步 → RVQ 三层码本索引、token、解码近似轨迹与整段 mask
  └─ min(t+10,N-1) 地面点 → 相机投影、深度检查、沿路径回退 → apos
实际末帧 N-1 地面位置 → 最后二分之一窗口 → 固定终点投影、深度检查 → opos
```

训练输入可使用指令和截至 t 的视觉历史；未来位姿和终点属于监督信息。
`history_start` / `history_end_inclusive` 记录允许的历史范围，目前未构建训练
prompt、历史图像采样器或 FM 训练器。不要把未来 RGB 放进当前样本的视觉输入。

## 数组和含义

所有数组保存在 `targets.npz`；`samples.jsonl` 每行包含帧号、图像路径、数组行号、
RVQ token、apos/opos 字典及各通道 mask。`episode.json` 保存指令、相机和轨迹级说明。
当前补标目录通过 `source_episode` 和样本绝对路径引用原始 RGB/深度/frames.jsonl。

| 训练内容 | 数组 | 形状 | 含义 |
|---|---|---|---|
| FM 路点 | `fm_waypoints_ego` | N×30×3 | 未来每帧相对当前机体的 `[forward_m,left_m,yaw_rad]` |
| FM 运动增量 | `fm_clean_se2_diff` | N×30×3 | 每次运动相对前一帧机体的 `[delta_forward_m,delta_left_m,delta_yaw_rad]` |
| FM 有效性 | `fm_valid_mask` | N×30 | 每个未来步是否实际存在 |
| 未来索引 | `future_frame_ids` | N×30 | 每个未来点对应的帧号；缺失为 -1 |
| 高度 | `future_vertical_offset_m` | N×30 | 相对当前帧的世界 Y 高度差，米 |
| RVQ 编码 | `rvq_codes` | N×3 | 三层码本各一个索引，联合表示整段未来十步 |
| RVQ 解码点 | `rvq_reconstruction` | N×10×3 | 三个索引解码后的近似路点，坐标与 FM 路点相同 |
| RVQ 有效性 | `rvq_valid_mask` | N | 完整十步运动或经验证的终止 token 是否可监督 |
| apos/opos 像素 | `apos_pixel` / `opos_pixel` | N×2 | 当前原始图像的浮点 `[u,v]` |
| 可见标记的世界点 | `apos_world` / `opos_world` | N×3 | 实际可见标记的世界 `[X,Y,Z]`，米 |
| 原始三维目标 | `apos_target_world` / `opos_target_world` | N×3 | 在可见性筛选前的目标；不可见时仍保存 |
| 目标帧号 | `apos_target_frame_id` / `opos_target_frame_id` | N | apos 为 min(t+10,N-1)，opos 固定为 N-1 |
| 像素行列 | `apos_pixel_rc` / `opos_pixel_rc` | N×2 | `[floor(v),floor(u)]`，从 0 开始 |
| 网格行列 | `apos_grid_rc` / `opos_grid_rc` | N×2 | 27 行×48 列模型网格的 `[row,col]`，从 0 开始 |
| 空间格号 | `apos_grid_id` / `opos_grid_id` | N | `1+row*48+col`；无空间点为 -1 |
| pointing token | `apos_token_id` / `opos_token_id` | N | 可见空间点 1..1296；确认不可见或未知均为 0，通过各自 mask 区分 |
| pointing 有效性 | `apos_valid_mask` / `opos_valid_mask` | N | 每个通道独立决定是否监督 |

FM/RVQ 的第三个分量是 yaw，正方向为逆时针；第二个分量为左向位移。
它们是平面 SE(2) 轨迹，不是世界 `[X,Y,Z]`。完整世界位置、姿态保存在
`world_from_body` (N×4×4)，世界 +Y 向上；楼梯高度另存，没有混到 yaw 里。

## FM 与 RVQ 的关系

`fm_clean_se2_diff[t,0]` 就是 t→t+1 的下一步实际运动；接着是 t+1→t+2，
一直到最多 t+29→t+30。积分这些增量可还原 `fm_waypoints_ego[t]`。
FM 的 clean GT 使用量化前的实际轨迹，未用 RVQ 重建值替代。
当前存储为米/弧度，尚未归一化；噪声、流时间、训练目标及 loss 在 FM 训练阶段构造。
这里的 30 步是未来环境采样步，与 flow 求解器的积分迭代次数分开。

RVQ 量化前的十个真实路点可直接读取 `fm_waypoints_ego[t,:10]`。
使用发布的三层、每层 256 项码本，将这十个路点编码成三个索引，输出三个
`<act_l0_*><act_l1_*><act_l2_*>` token。三个 token 联合表示一段十步轨迹，
没有“一枚 token 对应一个路点”的关系。
`rvq_reconstruction[t]` 是量化后的十个近似路点，用于解码/误差检查；
`rvq_ade_m[t]` 保存与原始十点的平均平面位置误差。
RVQ 的编码前路点和解码后路点均使用米/弧度；索引及 token 本身没有物理单位。
“米制”不是整数米的量化精度，0.25 仍表示 25 cm。码本原始向量处于
`weighted_diff` 空间，不能直接按米解读；解码先将各层码字相加，再除以
`jacobian_weights`，最后积分 SE(2) 增量得到米制路点。当前 bundle 的
`alpha_mode=none`、`global_scale=1.0`，本管线没有套用 InternNav 的平移乘 4。
本地编码器使用逐层 ADE 加末端朝向误差搜索，不声称逐 bit 复现未公开的官方编码器。

新训练样本不输出前进/左转/右转类别，实际转向保留为 yaw 数值。
内部回放仍沿专家动作采样；十个 step 可能包含原地转向，不能解释为十段等距离运动。
`physical_dt_s=null`，这些运动量不是速度，视频 FPS 也不是控制周期。

## apos 与 opos

apos 的原始三维目标是第 min(t+10,N-1) 帧的地面位置。目标可见时直接标记；
不可见时沿未来 GT 路线反向搜索，每段按不超过 0.02 m 间距插值，选取沿路径
最接近原目标的可见点。因此 `apos_world` 在回退后可能不同于 `apos_target_world`。
整段没有可见候选时像素为 unknown/masked，但 `apos_target_world` 仍存在，
保存 `apos_token_id=0` 和 `<apos_0>`。

opos 的三维目标固定为实际末帧地面坐标，不回退。默认 t < floor(N/2) 时
输出 `<opos_0>`、mask=true、reason=before_opos_visibility_window，无二维点。
从这个边界帧开始才投影终点并按真实几何决定是否显示。
可见时有像素及网格，确认不可见时是 `<opos_0>` 的有效分类监督；此时无空间格子。
投影处深度无效时也保存 `<opos_0>`，但为 unknown/masked。
两种情况仍保留 `opos_target_world`，通过 state 和 mask 区分。
当前默认终点模式不需要豆包 API。投影公式和遮挡阈值见 [POINTING.md](POINTING.md)。

## 2864 第 20 帧的实际例子

完整数值可查看 [第 20 帧四类 GT JSON](../reports/episode_2864_frame20_gt_20260912.json)，
其中含未来 30 步所有点和增量、RVQ 的十个真实点及十个解码点、两个 pointing 点，
并记录源数据哈希。以下数值四舍五入：

```text
FM 下一步 t20→t21： [0.250000, 0.000000, 0.000000]
FM 第三步 t22→t23： [0.000000, 0.000000,-0.261799]
FM 未来第 10 点（t30，相对 t20）： [1.965926,-0.258819,0.000000]

RVQ 索引： [32,150,207]
RVQ token：<act_l0_32><act_l1_150><act_l2_207>
RVQ 解码第 10 点： [1.614427,-0.298981,0.008531]
这一段量化平均位置误差：约 0.116891 m
```

| 点 | 对应帧 | 世界位置（米） | 像素 `[u,v]` | 网格 `[row,col]` | token |
|---|---|---|---|---|---|
| apos | 30 | [-3.768055,-0.208958,-1.424674] | [252.862260,200.732727] | [19,25] | `<apos_938>` |
| opos | 51 | [-2.065123,-0.208958,-5.280107] | [255.047416,159.474981] | [15,26] | `<opos_747>` |

本帧 apos 未发生回退，三维目标与可见世界点相同。

## 尾部与训练 mask

2864 共 52 帧（0..51）：

- FM：0..21 共 22 行有完整未来 30 步；22..50 共 29 行只有部分未来步；51 无未来。
- RVQ：0..41 共 42 行有完整未来十步；42..50 的运动 chunk 不完整，整体 mask；
  51 有经显式专家 STOP 与终点距离检查确认的终止标签，三个索引为 `[6,122,174]`。
- apos：38 行有有效像素，14 行缺少可见点；所有 52 行均保留三维目标。
- opos：36 行可见、16 行确认不可见，全部 52 行可做各自对应的分类监督。

FM 的未知尾部有数组补齐值，只可使用 `fm_valid_mask=true` 的步。
一个通道不可监督不要求删除整帧；FM、RVQ、apos、opos 使用各自 mask。
像素监督缺失不等于该帧没有运动 GT。

```python
from pathlib import Path
import json
import numpy as np

episode = Path('/mnt/cfs/data/beta_nav/NEW_POINTING_RUN/episodes/2864')  # 换成重新生成的数据目录
samples = [json.loads(s) for s in (episode / 'samples.jsonl').read_text().splitlines()]
t = 20
with np.load(episode / 'targets.npz', allow_pickle=False) as gt:
    fm = gt['fm_clean_se2_diff'][t]        # (30,3)，逐步运动量
    fm_mask = gt['fm_valid_mask'][t]       # (30,)，仅监督实际未来步
    rvq_true_points = gt['fm_waypoints_ego'][t, :10]
    rvq_codes = gt['rvq_codes'][t]         # 仅在 rvq_valid_mask[t] 为真时使用
    rvq_decoded_points = gt['rvq_reconstruction'][t]
    apos = samples[t]['apos']             # 三维目标、可见像素、格子、loss_valid
    opos = samples[t]['opos']
```
