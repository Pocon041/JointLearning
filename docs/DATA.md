# 数据格式与划分

## 当前训练池

COD 使用 COD10K 的伪装目标图像及 mask：Quick 684、扩大 1361、全量训练 2723，固定验证 312，官方 COD10K 伪装测试 2026。全量训练 2723 + 验证 312 构成 3035 张官方伪装训练图；没有将 COD10K 的全部非伪装/背景图纳入本协议。

IML 使用 CASIA v2 篡改图及对应二值标注：Quick 1152、扩大及全量训练 4610，固定验证 256，来源分组保留测试 256。训练只用篡改图，不加 authentic 图。CASIA 测试是同数据来源的内部保留集，不是跨数据集泛化测试。

数据入口参考：

- [COD10K / SINet](https://github.com/DengPingFan/SINet/)
- [CASIA v2 ground truth](https://github.com/namtpham/casia2groundtruth)

本仓库不分发图像、mask 或原实验逐样本划分名单。上述计数描述原实验协议；自行重新随机划分的数据不能标为同一 split 的精确复现。源图关联与重复检查应在划分前完成，尤其注意 CASIA donor/host 的关联。

## JSONL 格式

每行一条记录，路径默认相对于 manifest 所在目录解析：

```json
{"id":"sample_001","image":"../../images/sample_001.jpg","mask":"../../masks/sample_001.png","task":"iml","source_id":"source_group_001","family":"splicing"}
```

- `id`：本 manifest 内唯一标识。
- `image`、`mask`：RGB 图像和对应原生分辨率二值 mask 的路径。图像与 mask 尺寸必须一致。
- `task`：`cod` 或 `iml`。
- `source_id`：来源分组标识；数据读取器保留它，但不会自动替你执行正确的数据划分。
- `family`：CASIA 使用 `copy_move` 或 `splicing`。原配置用 `tampered_f1` 选模，因此应提供准确 family。
- `mask_threshold`：默认 0，原 mask 大于该值为前景，可处理 0/1 和 0/255 mask。
- `mask_invert`：可选布尔值，只在经数据审核确认标签反向时设置。

也可在配置中将 manifest 写为 `{"manifest":"path.jsonl","root":"dataset-root"}`，显式指定图像根目录。不要将 padding 标为背景参与损失。

## 规范文件名

`tools.prepare_configs` 约定：

```text
<manifest-root>/cod/train_quick.jsonl
<manifest-root>/cod/train_expanded.jsonl
<manifest-root>/cod/train_full.jsonl
<manifest-root>/cod/val.jsonl
<manifest-root>/cod/test.jsonl
<manifest-root>/iml/train_quick.jsonl
<manifest-root>/iml/train_full.jsonl
<manifest-root>/iml/val.jsonl
<manifest-root>/iml/test.jsonl
```

IML 扩大与全量均读取 `train_full.jsonl`，不能把历史环境中误名为 train_quick、实际含 4610 条的文件当作 Quick1152 清单。

## 处理顺序

训练只做成对水平翻转、垂直翻转和 90 度旋转，然后等比例全图缩放；当前没有随机裁剪、JPEG 或颜色增强。数据采样由任务/epoch/样本/重复次数确定随机种子，并按形状重组 batch，不丢弃或增加采样次数。

验证保留原生 mask。概率在去 padding、还原到原生 mask 尺寸后计算。原实验在已有 source ID、文件名解析 donor/host 和图像哈希层面检查了 split 交叉；这不等于证明所有近重复或真实场景关系均已穷尽。
