# BetaNav checkpoint 保存耗时诊断（2026-09-13）

本次八卡训练的 step 3000 checkpoint 共 47.028 GiB：model.pt 16.708 GiB，
optimizer-vlm.pt 30.048 GiB，optimizer-fm.pt 0.272 GiB。BF16 是计算精度；
当前模型主参数、Adam 一阶与二阶状态使用 FP32，完整恢复需要保存这些状态。

从最早的 rank RNG 文件写完到 COMPLETE 创建，耗时 1,386.66 秒（23 分 6.66 秒）。
到 model.pt 写完已过去 1,276.65 秒；这段包含优化器汇总、模型复制和模型写入，
仅靠文件时间戳无法进一步分离。

## 现场证据

对正在保存 step 3047 的原训练进程进行非阻塞 py-spy 栈采样：

- rank 0 在 `ZeroRedundancyOptimizer.consolidate_state_dict` 调用的 `_broadcast_object` 中等待。
- rank 1 持有 GIL，停在 `_broadcast_object` 的 `torch.ByteTensor(data).to(device)`。
- 已安装 PyTorch 的实现先 `torch.save(obj, BytesIO)`，转成 `bytearray`，再构造 ByteTensor，
  按 rank 顺序广播完整序列化对象；其他接收 rank 也会反序列化对象。

CPU 小规模转换测量：32 MiB 的 `torch.ByteTensor(bytearray)` 用时 1.443 秒，
约 22.17 MiB/s。对多 GiB 分片反复执行此转换可累积为很大的串行开销。
同一缓冲区使用 `torch.frombuffer` 构造视图约 45 微秒；两者内存所有权与复制语义不同，
此对比仅用于定位转换成本，不能作为整个 checkpoint 的加速比。

## 本次实际改动

R2R 配置设为 `eval_every: 0`、`save_best: false`。禁用验证时不构建验证 loader，
从旧 checkpoint 恢复允许更改验证选项，仍严格检查数据、模型、batch、优化器与训练 schedule。
保留每 1000 步以及结束/优雅停止时保存完整状态。
原先验证每 500 步改善都会通过 save_best 触发额外保存，关闭后减少了保存次数。

本次没有替换正在运行的 PyTorch 或改变 checkpoint 文件格式，单次保存瓶颈尚未修复。
后续优化可采用每 rank 保存本地 optimizer 分片，恢复时核对 world size 和分片参数映射，
以避免跨卡汇总巨大 Python 对象；或在严格测试格式兼容后优化字节传输实现。
无论哪种方案，都应保留模型、两套 Adam moments、step、数据游标和每 rank RNG，
并实测完整保存/恢复耗时，不能只减少保存内容后声称性能问题已解决。

数据证据见同目录 checkpoint_slow_diagnosis_20260913.json 和
checkpoint_conversion_benchmark_20260913.json。
