# 当前方法：共享 Q/KV、私有尺度出口与任务独立优化

## 任务与参数划分

COD 的前景是伪装目标，IML 的前景是篡改区域。两类图像不需配对，标注分别用于自己的任务损失。当前 IML 数据是 CASIA v2 篡改数据，任务范围已超出早期 inpainting-only 设定。

模型参数分为冻结通用参数 theta_0、共享可训练参数 theta_s、COD 私有参数 phi_c、IML 私有参数 phi_i。

- theta_0：通用 ImageNet MiT-B4 的原始投影、卷积、归一化等参数。
- theta_s：四尺度频域嵌入、图像嵌入降维、提示升维，以及全部 block 的共享 Q/KV LoRA。
- phi_t：各 block 的任务提示变换、四尺度私有出口、任务 decoder。

共享参数是同一批 Parameter 对象，但两任务经过不同的提示变换，因而不要求激活相同。没有主任务，也没有将一个 decoder 的预测用作另一个任务的标签。

## 频域提示与逐 block 注入

对归一化 RGB 的有效图像矩形，执行 FFT、高通、逆 FFT 和绝对值：

```text
HFC(I) = abs(real(IFFT((1 - M_low) * FFT(I))))
```

使用 EVP 的中心方形低频掩码及 rate=0.25。代码通过 `line=int(sqrt(H*W*rate)//2)` 构造中心区域；该参数不是径向频率阈值，对非方形图像的比例也不应作严格面积解释。逆变换后的 HFC 是空间响应图。

HFC 经四阶段重叠 patch embedding 得到 H_s，提示通道为 C_s/4。主干 stage 入口 token 经共享 Linear 得到 E_s，令 Z_s=E_s+H_s。对任务 t、block i：

```text
P_sit = U_s(GELU(W_sit Z_s + b_sit))
X_next = Block_si(X + P_sit)
```

W_sit 及其偏置任务私有，U_s 共享。Z_s 在每个 stage 入口计算，不在每个 block 重算。频域融合采用原 EVP 的相加方式，当前配置未启用任务频域 gate 或旧结构/痕迹增强分支。

## 共享 Q/KV LoRA

MiT-B4 的通道为 (64,128,320,512)，深度为 (3,8,27,3)，attention heads 为 (1,2,5,8)，空间缩减比例为 (8,4,2,1)。41 个 block 的 Q 和融合 KV 各增加低秩更新：

```text
Linear_shared(x) = Linear_frozen(x) + (alpha/r) * B(A(x))
r = 8; alpha = 8
```

Q 作用于当前 token，KV 作用于该 block 的空间缩减特征。更新同时影响注意力匹配和被聚合的值。COD 和 IML 通过各自损失更新同一组低秩因子。

冻结权重与低秩输出分别计算后相加，训练/评估时均不合并权重。A 使用 Kaiming uniform，B 初始化为零，初始函数保留通用投影。

## 私有四尺度出口

每个 stage 的最后一个 block 接到同一份含任务提示的输入 token：

```text
                    ┌─ normal block ─ norm ─ next stage
tokens + prompt ────┤
                    └─ private exit ─ norm ─ task decoder
```

私有出口复用该 block 的通用权重和共享 Q/KV，在 attention output projection、FFN FC1、FFN FC2 增加任务私有低秩输出，rank=4、alpha=4。FC1 后保留 MiT 的 depthwise convolution、GELU 和 dropout；FC2 接收私有路径自己的中间激活。

没有新增私有 QKV，也没有复制一套可训练主干。私有出口结果仅供对应 decoder，不作为下一 stage 的前向输入。出口损失仍通过所依赖的共享组件反传。DropPath/dropout 的随机数通过 fork_rng 与主路径耦合，零增量时保持训练/评估初始输出和全局 RNG 行为。

私有出口增加 98,304 参数（49,152/任务）。总可训练参数 7,640,434，共享可训练参数 772,400。

## Decoder、损失与几何

两套独立 SegFormer Head 分别将四尺度特征线性投影到 768 通道，上采样到 1/4 尺度，按 c4/c3/c2/c1 拼接，接 1x1 Conv、BN、ReLU、Dropout2d(0.1) 和 1x1 单通道输出。BN 统计也任务私有。

COD 使用逐图 BCE + soft IoU；IML 使用逐图 BCE。均在有效像素内求平均，再对 batch 求平均。当前无额外 Dice、边界损失、跨任务预测一致性或蒸馏损失。

COD 全图等比例 fit 长边 512；IML 保持原生尺寸，长边超过 512 才下采样。batch 右侧/底部补 ImageNet 均值，归一化后为零。FFT 只对有效矩形计算，再补零并运行共享频域嵌入金字塔。损失忽略 padding；评估先从 logits 去掉 padding，再插值回原生 GT 尺寸，最后 sigmoid。不作逐图 min-max。

这种处理避免 padding 边缘直接进入 FFT 和损失，但后续 attention、卷积、BN 等仍可能受 padding 影响，并非完全 padding invariant。

## 任务独立 Adam 统计

两个任务在同一份更新前参数上分别前向/反向，中间不 optimizer.step。共享参数的 COD 和 IML 梯度分别保存；以两任务梯度之和及私有梯度计算一次全局 clipping 系数，并将同一系数施加到各保存梯度。

对每个共享参数分别维护 (m_c,v_c) 与 (m_i,v_i)，各自进行 Adam 偏差校正后求和更新：

```text
theta <- (1 - lr*wd) theta
theta <- theta - lr * [mhat_c/(sqrt(vhat_c)+eps) + mhat_i/(sqrt(vhat_i)+eps)]
```

这是 sum aggregation，不是先合并原始梯度后运行普通 Adam，也不是两个任务依次更新权重。权重衰减每次共享更新只执行一次。私有参数使用普通 AdamW；没有外任务梯度时，不更新矩与衰减。

独立统计用于减少不同任务梯度历史与尺度的混用，不执行显式梯度冲突投影，不保证消除方向冲突。单活跃任务时退化为普通 AdamW。

## 与来源工作的关系

- EVP：保留 FFT 高通、embedding/HFC 提示及 MiT/SegFormer 基础；加入任务路由和共享注意力适配。
- AdaTask：保留按任务维护 Adam 矩、归一化更新求和的原则。AdamW 衰减、角色学习率、缺失梯度和状态恢复为本项目适配。9 步等 LR、零衰减 CPU 对照与作者参考更新最大误差 0。
- MTLoRA：借鉴共享流继续传播、私有多尺度读出及低秩适配原则。这里是 MiT+EVP，出口只加在每个 stage 末尾；私有 FC2 消费自己的 hidden，不是完整移植作者 Swin forward。

单个私有线性投影公式与作者算子 float64 对照误差约 8.9e-16，不能据此宣称整模型等价。来源 URL、文件哈希和执行记录见 `reference_sources.json`、`reference_executed_formula_fidelity.json`。
