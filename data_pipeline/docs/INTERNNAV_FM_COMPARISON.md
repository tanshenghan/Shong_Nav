# 当前运动 GT 与 InternNav FM 的区别

核对日期：2026-09-12。InternNav 官方代码版本：
`7a5c62400ac45b313d9b709c740b64191556a242`。
比较对象为 InternVLA-N1 的 `nextdit` / `nextdit_async` FM 分支；
该仓库还有 NavDP 的 DDPM 分支，不能将它们统一视为一种训练实现。
本次只保存对照说明，没有变更 GT。

两边都能用未来轨迹作为生成模型的监督，但动作张量含义不同：

| 项目 | 当前管线 | InternNav 所核对的 FM 分支 |
|---|---|---|
| 行的含义 | 下一次实际专家采样运动 | 平滑路径的空间重采样段 |
| 长度 | 未来最多 30 个记录步 | 官方训练脚本配置 32 段 |
| 平移坐标 | 每一步相对前一步机体，坐标随转向旋转 | 当前参考帧固定坐标系中的相邻点差 |
| 朝向 | 实测机体朝向差 | 由重采样路径切线推导，再计算方向差 |
| 原地转向 | 保留真实零位移与非零 yaw | 零位移点被路径过滤，不保留独立原地转向标签 |
| 数值尺度 | 米、弧度，未归一化 | 平移差乘 4，方向差未乘 4 |
| 未来不足 | 对不存在的实际未来步设置 mask | 路径不足采样距离时重复终点，未见等价的逐轨迹步 mask |

采样、坐标、方向和缩放依据
[官方数据处理代码](https://github.com/InternRobotics/InternNav/blob/7a5c62400ac45b313d9b709c740b64191556a242/internnav/dataset/internvla_n1_lerobot_dataset.py#L571-L740)；
32 段依据 [官方训练脚本](https://github.com/InternRobotics/InternNav/blob/7a5c62400ac45b313d9b709c740b64191556a242/scripts/train/qwenvl_train/train_dual_system.sh#L35-L59)。
尾部 loss 依据 [FM 训练代码](https://github.com/InternRobotics/InternNav/blob/7a5c62400ac45b313d9b709c740b64191556a242/internnav/model/basemodel/internvla_n1/internvla_n1.py#L229-L286)。

实现细节：过滤条件是平移长度的平方大于 0.05，不是“距离大于 0.05 m”。
平滑函数默认 interval=0.1，但使用 33 个点覆盖 3.3 m，32 段的弧长间隔约
0.103125 m；短路线会重复终点。因此不能把其 32 段称为原始未来 32 帧或固定时长。
路径切线的首个角作为首段方向增量，之后才是相邻切线角的差。
这些结论限定于上述版本的具体函数，不推广到 InternNav 的全部数据集。

仅比较坐标/尺度的示例：机器人已经左转 90°，继续沿自身前方走 0.1 m 且不转向。
当前表示为 `[0.1,0,0]`；在起始参考轴中这段平移是 `[0,0.1]`，经过乘 4 的
InternNav 表示为 `[0,0.4,0]`。示例未加入路径平滑、重采样的额外变化。

FM 方法本身不要求只能用某一种动作定义。官方 nextdit 训练在动作与高斯噪声之间
插值，监督噪声与动作之差；它是生成过程的流场目标，不是机器人实际速度。
当前管线保存 clean 运动 GT，尚未构造 FM 噪声、流时间及训练 loss。

若训练自己的 FM，可以使用当前实测 SE(2) 增量，并保持训练与解码一致。
若复用 InternNav 的 FM 权重，需要对齐重采样、固定参考坐标增量、切线朝向、
尺度和尾部规则；仅把 `[30,3]` 改成 `[32,3]` 不够。转换可从已保存的完整
`world_from_body` 构造，无需为动作标签重采集 RGB，但模型视觉输入还需单独对齐。
当前本地定义与数组读取见 [GT_CONSTRUCTION.md](GT_CONSTRUCTION.md)。
