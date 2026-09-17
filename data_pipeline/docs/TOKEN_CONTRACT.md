# RVQ STOP、缺失点位与模型词表

当前规则：新数据的 apos/opos 缺失编号统一为 0，保存 `<apos_0>` / `<opos_0>`，
保留原有 state 和 loss mask。`pointing_config.unknown_token_id=0` 标识这个版本。
结果见 [零 token 更新报告](../reports/pointing_zero_update_20260912.json)。

更新后已实际重新采集 episode 5670 的 60 帧，全部 pointing token 非负，
每个 token 字符串均通过本地 tokenizer 检查，末帧使用 `<apos_0><opos_0>`
和预留 RVQ STOP。见 [新数据生成验证](../reports/episode_5670_zero_generation_20260912.json)。

此前已核对模型 `/mnt/cfs/Models/Light` 的 tokenizer 和 RVQ bundle，
并逐帧检查当时的 `r2r_2864_opos_endpoint_20260912/episodes/2864`。
[历史核对报告](../reports/token_contract_validation_20260912.json) 保留更新前的 -1/null 编码，
当前原始数据集目录已被清空；这些历史报告不能替代重新生成的数据。

## 原论文与官方公开代码的依据

论文 [III-C：Dual-Channel Pointing](https://arxiv.org/html/2608.30935v1#S3.SS3)
明确说明：对应 token family 中预留索引处理没有有效图像网格位置的情况，
包括原地转向、停止和目标不可见。该段没有列出具体预留编号。
同节公式 (4) 使用两个 pointing token 加三个 RVQ token；
[III-E](https://arxiv.org/html/2608.30935v1#S3.SS5) 说明统一的自回归监督目标。

2026-09-12 核对官方仓库 main 的
[vln_utils.py](https://github.com/lightorigins/LightNav-0/blob/main/src/lightnav/vln_utils.py)
和 [serving/protocol.py](https://github.com/lightorigins/LightNav-0/blob/main/src/lightnav/serving/protocol.py)，
grid token 的具体定义为：

| 官方 token | 注释和解码语义 |
|---|---|
| `<opos_0>` | 目标不可见，解码状态 `not_visible` |
| `<apos_0>` | 没有 apos 标签，解码状态 `none` |
| `<apos_1297>` | 原地左转，解码状态 `rot_left` |
| `<apos_1298>` | 原地右转，解码状态 `rot_right` |
| `<apos_1299>` | 停止，解码状态 `stop` |
| `<apos_1>`..`<apos_1296>` / `<opos_1>`..`<opos_1296>` | 有效图像网格位置 |

因此官方对 apos 无投影情况提供无标签及转向/停止等不同表达，
不能将所有 apos 不可见情况解释为同一种转向或停止指令。
`<apos_0>` 的含义也不能直接等同于 `<opos_0>` 的目标不可见分类。

当前管线遵循用户“不再输出离散转向/apos STOP”的要求，未生成 apos 的
1297..1299 指令；缺失 apos 现保存为 0/`<apos_0>`/false，是本管线的实现约定。
上述论文段落与公开解码定义不足以确定官方 `<apos_0>` 的逐 token loss mask
处理，不能据此宣称本管线的 loss mask 做法就是官方完整训练标注规则。

代码还支持 posxy 形式的 `<opos><novis>`，但本地已核对的 checkpoint
和当前数据采用 grid `<opos_k>` / `<apos_k>` 格式，不能混用两套拼写。

## 最后一帧的 RVQ STOP

当前 bundle 的 `action_tokenizer/manifest.json` 定义：

```json
{"stop": {"l0": 6, "codes": [6, 122, 174]}}
```

采集器要求专家动作恰好以一次 STOP 结束，并验证回放路线和终点可达距离。
通过这些检查后，末帧写出预留 STOP 三元组并启用 RVQ/STOP 监督。
普通运动编码排除第 0 层 code=6，避免把普通静止/短运动误编码为预留 STOP。

episode 2864 的第 51 帧按当前规则表示如下；STOP 数值来自已保存的历史标签，
apos 缺失编号和字符串按零 token 规则转换，原始图像数据已不可用：

```json
{
  "frame_id": 51,
  "rvq_codes": [6, 122, 174],
  "rvq_tokens": "<act_l0_6><act_l1_122><act_l2_174>",
  "rvq_loss_valid": true,
  "explicit_stop": true,
  "stop_loss_valid": true,
  "fm_valid_steps": 0,
  "apos_tokens": "<apos_0>",
  "opos_tokens": "<opos_0>"
}
```

第 42..50 帧不足十个未来帧，RVQ 运动标签 masked；第 51 帧独立具有显式 STOP
监督。末帧 FM 的 30 个 mask 全为 false，不能把补齐值当作三十步真实静止轨迹。

`RVQBundle.is_stop` 使用第 0 层 code=6 判断停止，采集器输出上述完整三元组。
STOP 的码本直接解码值存在约 8.3e-6 的残差；运行端识别预留码后输出全零路点。
因此终止语义由预留码确认，不依赖重建轨迹数值恰好为零。
代码见 [collect_r2r.py](../collect_r2r.py)、[geometry.py](../geometry.py)、
[traj_vocab.py](../../src/lightnav/traj_vocab.py) 和 [tracking.py](../../src/lightnav/tracking.py)。

## 不可见与未知的标签

| 通道与状态 | 数据内 token_id | token 字符串 | loss_valid |
|---|---|---|---|
| apos 可见（包括沿路径回退） | 1..1296 | `<apos_k>` | true |
| apos 搜索后没有可见候选 | 0 | `<apos_0>` | false |
| opos 可见 | 1..1296 | `<opos_k>` | true |
| opos 在最后二分之一窗口之前 | 0 | `<opos_0>` | true |
| opos 确认在视野外、相机后方或被遮挡 | 0 | `<opos_0>` | true |
| opos 投影处深度无效，无法判断 | 0 | `<opos_0>` | false |

窗口限制使用 `reason=before_opos_visibility_window`，起始帧为 `floor(N/2)`。
进入窗口后仍检查真实投影与深度。这里新增的是训练标注策略，词表编号不变。
opos=0 且 mask=true 是有效的“不显示目标”分类标签，没有空间像素/格子；
其三维目标仍然保存。深度未知的 opos=0 仍为 mask=false。
apos 没有候选时同样保留三维目标，但当前流程跳过该帧的 apos 监督。
实现见 [pointing.py](../pointing.py)。

本地词表存在 `<apos_0>`，共享解码约定为 no label，当前采集器在缺失 apos
时输出该 token，且保留 loss_valid=false。词表中的 `<apos_1297>`、`<apos_1298>`、`<apos_1299>`
分别是历史左转、右转、停止指令；当前新 GT 不使用这些指令。

## 通道编号与 tokenizer ID

数据内的 code/token_id 表示该通道的编号；模型 tokenizer 的整数 ID 是整个词表的索引。
例如 opos 的通道编号 0 对应 `<opos_0>`，再映射到 tokenizer ID 153743。
RVQ 的 code=6 必须结合 level=0 才能得到 `<act_l0_6>`。

| token | 当前模型 tokenizer ID |
|---|---|
| `<act_l0_6>` | 151681 |
| `<act_l1_122>` | 152053 |
| `<act_l2_174>` | 152361 |
| `<apos_0>` | 152443 |
| `<opos_0>` | 153743 |
| `<apos_938>` | 153381 |
| `<opos_747>` | 154490 |

这些全局 ID 只针对核对报告中具有指定哈希的 tokenizer；更换模型时必须重新映射。
`<apos_-1>` 和 `<opos_-1>` 均不在词表中，当前 pointing token 不再生成 -1。
历史 -1/null 只用于只读兼容。缺失像素行列、网格索引和未来帧号仍可使用 -1
作为存储哨兵；这些索引不是 pointing token，不能改成有效坐标 0。
RVQ 缺失运动 chunk 的存储占位也不在本次 pointing token 修改范围内。

实际检查了 RVQ 768 个、apos 1300 个、opos 1297 个，共 3365 个 token：
全部存在、各自只编码为一个 ID、可解码回原字符串、ID 互不冲突且在模型词表范围内。
另在原数据删除前检查了示例 episode 52 帧实际输出的 219 个有效 token，均匹配源标签。

## 训练适配的边界

目前已完成数据标签和 tokenizer 对应关系的检查，尚未实现训练 prompt/target
拼接、逐 token loss mask 或 LeRobot 导出器。不能把上述检查理解为已运行训练。

后续训练保持固定的 `apos, opos, RVQ l0, l1, l2` 五个输出位置时，可直接使用
当前输出的 `<apos_0>` 填充缺失 apos 的位置，并根据 loss_valid=false
将该位置的交叉熵标签设为忽略值（例如 PyTorch 常用的 -100）。
未知 opos 的 `<opos_0>` 同样跳过监督，确认不可见 opos 的 `<opos_0>` 则参与监督。
忽略值只能进入 loss labels，不能作为 tokenizer 的输入 ID。
若需要学习“apos 无可见点”的显式类别，则需单独确定新的监督语义和 mask。

FM 保持连续轨迹及逐步 mask，由 FM 的训练分支读取，不把米/弧度数组直接当成
语言模型 token ID。完整 action 保存设计见 [LEROBOT_FORMAT.md](LEROBOT_FORMAT.md)。
