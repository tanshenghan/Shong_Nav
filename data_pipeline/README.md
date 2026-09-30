# 数据管线

按数据来源组织数据采集、转换、GT 标注、验证与相关文档。

- [R2R / RxR 数据处理](R2R/README.md)：代码、配置、测试、文档、小型报告及历史归档统一位于 `data_pipeline/R2R/`。
- [R2R 全量生成](R2R/docs/FULL_R2R_TRAIN.md) 与 [RxR 生成](R2R/docs/RXR_LEROBOT.md)：各自的输入、运行命令与检查方式。
- [VLNVerse 数据处理](vlnverse/README.md)：官方逐步位姿、场景地图、平滑运动、验证独立分层；
  新增独立的 [A* 连续曲线分支](vlnverse/docs/PLANNED_ROUTES.md)，按有序检查点规划并验证连续转弯；
  已实现受限格式地图校准、连续参考运动、每源运动间隔的位姿采样，以及独立相机预览和比较视频入口；
  已完成并验证一个导航场景的 CPU 原始/平滑视频预览，Isaac GPU 未实测，物理回放、标签与导出仍待接入。

所有命令从仓库根目录运行，例如：

```bash
bash data_pipeline/R2R/run_r2r_train.sh /mnt/cfs/data/beta_nav/NEW_R2R_TRAIN --dry-run
bash data_pipeline/R2R/run_rxr.sh /mnt/cfs/data/beta_nav/NEW_RXR_TRAIN_GUIDE --dry-run
```

已安装的环境保留在 `data_pipeline/.venv/`，本机未使用的账户配置保留在被 Git 忽略的
`data_pipeline/config.py`。大型数据继续保存在 `/mnt/cfs/data/beta_nav`；本次目录整理不修改
已有数据、provenance 或历史代码包内容。

独立 GT 视频入口仍为 [visual/render_episode_gt.py](../visual/render_episode_gt.py)。
训练读取与训练运行仍分别位于 [src/lightnav/dataloader/](../src/lightnav/dataloader/)
和 [src/lightnav/training/](../src/lightnav/training/)，详见 [训练说明](../docs/TRAIN_BETANAV.md)。
