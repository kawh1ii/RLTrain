# 目标函数、内存和梯度遍历

## 数学契约

输入 `H[N,D]`、`W[V,D]`，目标 token 为 `y[N]`。设 `z = H W^T`，

`l_i = z[i,y_i] - logsumexp(z[i,:])`。

固定旧策略 `o_i`、参考策略 `f_i`、优势 `A_i` 和归一化权重 `q_i`：

`r_i = exp(l_i - o_i)`

`L_i = q_i × [-min(r_i A_i, clip(r_i,1-eps,1+eps) A_i) + beta × (exp(f_i-l_i) - (f_i-l_i) - 1)]`。

本模块支持这个基础 token-ratio GRPO 目标。它没有默默加入 veRL 默认 vanilla 路径中的 dual clipping，也没有 DAPO、GSPO、熵项或采样重要性修正。比较 Liger 时显式选择 GRPO 序列均值，不能沿用其 DAPO 默认值。

对 `l_i` 的导数，在裁剪阻断分支之外为 `-q_i A_i r_i`；当正优势且 `r_i > 1+eps`，或负优势且 `r_i < 1-eps`，策略梯度项为零。KL 项导数是 `q_i beta (1-exp(f_i-l_i))`。这是 k3 样本估计项，不是精确全词表 KL，也没有宣称新估计器。

得到一维 `g_i=dL/dl_i` 后：

`dL/dz[i,j] = g_i × (1[j=y_i] - softmax(z[i,:])_j)`

`dH = dZ W`，`dW = dZ^T H`。

公式、梯度和一阶优化器更新通过 FP64 有限差分/参考实现检查；不支持二阶导数。FP32/BF16 有舍入和裁剪拐点敏感性，不能把数学公式相同等同于任意 GPU 路径逐 bit 相同。

## 为什么改遍历维度

最初按 token 分块，每步重算部分 logits，再把整块 `dW[V,D]` 累加到 FP32 缓冲。对于大词表，V 远大于 N，这个累加缓冲和反复读写昂贵；该版本在开发测量中输给 torch.compile。

新版按词表分块：

1. `H @ W[v:v+C].T` 只产生一块 BF16 logits。
2. 每行分成 4096 元素的并行归约，得到局部 max/sum，再稳定合并到全词表 LSE；目标 token 的 logit 从真正的 GEMM 结果中提取。
3. 完整 LSE 就绪后，计算标量损失和一维 `g[N]`，只保存 H/W/标签/LSE/g，不保存完整概率矩阵。
4. 反向重算各词表块，将 logits 原地改写为 BF16 `dZ`。
5. 每个词表块的 `dW[v:v+C]` 直接用覆盖全部 N 的 GEMM 一次写出；在较小的 `[N,D]` FP32 缓冲累加 `dH`。

这是减少外部中间状态与遍历读写的实现选择，不是降低 GEMM 的数学复杂度。CUDA GEMM 仍由 PyTorch/cuBLAS 提供；内核自己实现的是归约、损失导数和原地梯度。

默认 `C=32768`。N=4096 时，一块 BF16 logits 为 256MiB；不是说整个算子只占 256MiB，输入权重、返回梯度、autograd、模型、优化器及 CUDA 工作区都另计。渐近临时状态从全量 `N×V` 转为 `N×C` 和较小归约缓冲；`dW[V,D]` 本身仍不可省，冻结输出头时才不分配它。

归约中的跨块 max/sum 保持数值稳定；核算接受范围和内存索引是独立问题。大矩阵地址使用 64 位行偏移，支持非整除的词表尾块。公开调用先检查标签范围和数值；基准对固定输入统一检查，计时段不重复同步。

## 掩码、归一化与微批

序列均值下 `q[b,t] = mask[b,t] / max(length_b,1) / B`。两个响应长度分别 2、1 时，权重是 `[.25,.25]` 和 `[.5]`，不是对所有 token 一律 1/3。

完整优化器批的 q 在切微批前构造。每个微批重新除以自己的样本数、然后直接累加 loss，会改变目标；Liger 对照需要额外乘 `micro_B/full_B`。测试不仅比权重，还比较分微批后的 loss、hidden/weight 梯度。

EOS 本身参与训练，首次 EOS 后全部排除。预测第一个响应 token 的位置是 `prompt_length-1`，之后依次右移；不是直接用该 token 自己的 hidden 预测自己。变长右填充可能使无效位置越界，必须 clamp 后同时给零权重。

## 冻结策略状态

输入轨迹、old log-probs 与参考 log-probs 分开存储并记录散列。old log-probs 的策略版本必须与 rollout 一致；在声明的有限 PPO epoch 范围内复用旧批，而不是在每个更新后刷新 old 概率。参考策略始终固定，不能因为 actor 更新就跟着变化。

采样与 teacher forcing 即使同一检查点也可能走不同数值路径。本轮 teacher-forced old/reference 来自检查点 0，固定后供所有后端共享；不声称逐 bit 等于缓存生成时的 logits。

## 实际模型步

Qwen2.5-0.5B，FP32 master 参数，BF16 autocast，偏置为空的 LM head，全参数反向，原生 fused AdamW。模型处于关闭 dropout 的 eval 模式但启用梯度，未开启骨干激活 checkpoint。所有后端都只对响应位置投影，避免把跳过提示词计算误算成自写融合的收益。

真实轨迹的正确性奖励全零，优势来自预定义长度项。本轮证明的是非零计算与参数更新路径，不是强化学习收敛。多步 BF16 更新漂移单列，不能称无损替换。

## 公平对照

统一数据、目标、梯度需求、dtype、TF32 设置、warmup、分配器与时钟口径。Liger 的 forward 内部已计算梯度，因此六个版本都计 forward+backward；编译和第一次初始化不混入稳态中位数。训练步另计完整骨干前向、反向和优化器；生成不在这个时钟中。

PyTorch allocated 峰值不包含所有驱动/CUDA 非张量内存，不能当成整机物理显存峰值。CUDA events 是当前流的一段时间，可能含 CPU 发射空隙，不是硬件计数器或单个 kernel 的纯执行时间。
