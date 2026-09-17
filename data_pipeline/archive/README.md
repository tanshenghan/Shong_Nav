# 清理前归档

`before_final_cleanup_20260912.tar.gz` 保存清理前的代码、测试、说明及小型报告，
供需要时人工查阅或恢复。旁边的 JSON 记录压缩包和逐文件 SHA-256。
本机 config.py、环境依赖和缓存均未打包。当前程序不导入或执行归档内代码。

本次从活动目录移除：

- doubao.py：API 调用、提示词、检测框解析与重试。
- destination.py：末八帧候选搜索和旧目标绑定。
- semantic_scene.py：语义重渲染与物体实例网格提取。
- annotate.sh：旧语义补标专用 Habitat 启动脚本。
- pointing.py 内旧物体绑定、停止区域框选和实例点生成函数。
- 对应 API/实例生成测试；当前有效的几何、终点、补标和验证测试保留。

旧 annotate_r2r.py、pointing.py 和全部原测试也包含在归档内。
历史数据与生成时 provenance 没有改动。归档恢复应解压到新的查看目录，
不要直接覆盖当前最终版源码。旧模块依赖同版本代码，不能只恢复单个文件混用。
