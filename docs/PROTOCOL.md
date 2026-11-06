# 匹配单任务与联合训练协议

## 三组对照

`both_cod_only`、`both_iml_only`、`both_joint` 均使用相同结构：EVP、共享 rank8 Q/KV、私有 rank4 stage exits 和 task_adamw_sum。两套私有模块均被构建，但单任务组只接受所属任务的 loss，另一任务私有参数不更新。

Joint 在同一份更新前参数上计算两个任务的梯度，保存独立 Adam 统计，再统一 step 一次。不存在前一任务先更新、后一任务看到新权重的交替优化。

## 数据量与预算

- Quick：COD684 / IML1152；6 轮，每轮352更新，每任务 batch4。
- 扩大：COD1361 / IML4610；6 轮，每轮1153更新，每任务 batch4。
- 全量：COD2723 / IML4610；30轮，每轮1153更新，每任务 batch4。

全量每组共 34590 次 optimizer.step，每个活跃任务 138360 次图像抽取。固定更新数匹配三个 arm 的本任务训练暴露；Joint 每步额外计算另一个任务，因此算力消耗不等于单任务组。

Quick、扩大、全量都从相同通用 ImageNet MiT-B4 重新初始化，并非接着上一阶段训练权重扩展。seed42。通用主干冻结，提示/decoder/低秩模块新初始化。

## 学习率与损失

- 共享参数及 COD 私有参数初始 LR=2e-4；IML 私有参数 LR=5e-4。
- betas=(0.9,0.999)，eps=1e-8；矩阵参数 weight decay=0.01，偏置及一维归一化参数默认不衰减。
- 全局梯度裁剪上限5；两任务 loss 权重均1。
- warmup 起始因子0.1，warmup_epochs=105/352，Quick105步、扩大和全量343步。
- warmup 后 cosine 到1e-6，每次 optimizer 更新后推进 scheduler。
- COD 为有效区域逐图 BCE+IoU，IML 为有效区域逐图 BCE；无附加 Dice 或边界损失。
- CUDA 训练及评估 BF16；checkpoint 保存 FP32 master 参数与完整优化器状态。

`size` 为保留的父实现配置字段；实际几何由 `input_geometry` 决定。COD fit512，IML native 到最大长边512。

## 验证与保留测试

每轮验证 COD312、IML256。COD 用 S-measure 选模，IML 用固定0.5的 tampered F1 选模。另记录 COD wF/MAE，以及 IML AUROC/precision/recall/FPR/oracle F1。

IML AUC 是4096 bins 的逐图像素 AUROC 宏平均；pooled AUROC 单独命名。Oracle F1 是各图根据 GT 选择 PR 曲线最大 F1 的诊断值，不能用于部署阈值或测试选模。SOD 指标显式 normalize=False，无逐图 min-max。

Quick/扩大生成配置关闭保留测试。全量每个 arm 完成全部30轮后，才评估最终 checkpoint 以及对应任务最佳验证 checkpoint；相同 hash 的重复评估缓存复用。测试不用于模型/epoch/阈值选择。

## 输出与恢复

运行目录包含 `protocol.json`、`config_resolved.json`、`status.json`、`metrics.jsonl`、`last.pt`、每轮 `epoch_XXX.pt`、任务 `best_*.pt`，全量最终还有 `test_results.json`。

稀疏 checkpoint 省略冻结通用参数，但保留所有学习因子和可变 buffer，以及原通用权重 SHA256。恢复前需重新加载相同通用权重。优化器记录任务矩、计数器和 sum/mean schema；当前实验使用 sum。

恢复只接受原 output_dir 的完整 epoch `last.pt`，并检查协议、初始化、manifest 内容和源码。中断 epoch 从完整轮次边界重算。跨目录移动或修改冻结源码后，不应声称原协议的精确恢复。

## 发布整理的边界

模型、训练器及所需依赖按字节复制，未调整其算法。发布新增了轻量公开入口、配置生成和单图预测。`release_source_manifest.json` 记录复制文件 SHA256；配置中的包文件清单按发布实际保留子集重建。没有把服务器绝对路径或运维控制器作为可移植入口。

GPU 源码预检验证过9个任务/优化器运行组合；本地回归套件验证小模型路由、梯度、BF16加法及恢复行为。工程验证与精度结论分别记录。
