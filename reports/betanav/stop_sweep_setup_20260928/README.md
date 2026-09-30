# STOP 八组开发集实验准备记录

入口：`scripts/eval_betanav_stop_sweep_8gpu.sh`。
完整说明：`docs/EVAL_BETANAV_STOP_SWEEP.md`。

固定 `step-00003699` 的同一 VLM/FM/StopHead，运行阈值 `0.5/0.6/0.7/0.8`
与连续确认 `1/2` 次的八组交叉实验。默认官方 R2R `val_seen`，每组一张已分配 GPU、六个常驻 worker。
所有组同一开发集、FM seed 42、10 次积分、500 步预算；仅阈值和确认次数不同。

## 已验证

- 通过真实 shell 入口的 `--plan-only`，生成本目录的 `sweep_plan.json`：8 组、48 worker、48 个不重叠端口。
  该计划使用占位 GPU token，尚未验证实际节点 GPU 分配，不应当作已运行结果。
- 已读取真实 checkpoint metadata：step=3699、training_mode=fm_stop；训练 manifest 的 SHA-256 与
  checkpoint `contract.data_sha256` 一致，10,819 条训练 episode 的 source 审计完整，dataset_root 契约一致。
- `make test PYTHON=.venv-train-stop/bin/python`：**1,590 passed, 9 skipped, 2 deselected**，73.31 秒。
- 本次修改文件 Ruff 通过；`make check` 在当前 PATH 无 Ruff 可执行程序时使用其语法编译 fallback，并通过。
- Shell 语法检查通过；CPU 测试覆盖连续确认的单次 VLM 编码、低分/episode 清零、待确认 FM 动作、零动作恢复、
  预算 STOP 不伪造确认、八组 GPU/端口隔离、模型加载限流、失败清理、开发集泄漏和混合协议拒绝。

## 实际资源预检限制

已执行 `--check-only`，在读取以下文件时遇到当前账户的操作系统权限错误：

```text
/mnt/cfs/data/habitat/datasets/R2R_VLNCE_v1-3_preprocessed/val_seen/val_seen.json.gz
```

完整错误见 `resource_preflight.log`。因此尚未验证 val_seen 的实际 episode 数量、GT/场景可读性和与训练集的独立性，
也未运行任何八组 GPU rollout。排队任务必须先通过这些检查，缺数据/权限不足/发现样本或路线重合时直接退出，
不回退到训练集或 val_unseen。

独立性范围为当前 FM + STOP 的训练数据，允许 val_seen 与训练场景重合；不审计发布的基础 VLM 预训练数据。
CPU 测试不能证明六 worker 显存峰值或任何阈值下的 SR 改善。
