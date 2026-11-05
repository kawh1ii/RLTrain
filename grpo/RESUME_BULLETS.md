# 第三个项目简历表述

\project{RLTrain v2：GRPO 大词表损失算子与显存优化}{2026.11}
\begin{itemize}
\item \textbf{算子与反向重写：}实现词表分块线性损失与自定义 autograd，编写 Triton 在线 LSE、损失导数及原地 logits 梯度内核；反向重算分块 logits，直接写出输出头梯度、累加较小 FP32 hidden 梯度，GEMM 复用 cuBLAS。
\item \textbf{算子性能对照：}在单卡 RTX 4090、BF16、$N{=}4096,H{=}896,V{=}151936$ 的前反向三次对照中，相比 \texttt{torch.compile}，峰值 allocated 显存由 \textbf{3.755 降至 1.044 GiB（降低 72.2\%）}，耗时由 \textbf{37.64 降至 32.50ms（降低 13.7\%）}；保留 checkpoint、Torch 手写反向及 Liger 对照。
\item \textbf{真实训练接入：}接入 Qwen2.5-0.5B、冻结真实轨迹及 FP32 master / BF16 AMP；microbatch 8 时，训练步峰值相比 compile 由 \textbf{14.724 降至 13.758 GiB}，耗时为 255.8 / 259.1ms，未声称整体提速或收敛收益。
\item \textbf{数值与复现：}完成 \textbf{42 项测试、29 个冻结源码实验阶段}，覆盖 FP64 梯度及三步 Adam、EOS 与微批归一化；真实模型首步梯度通过容差检查，保留 BF16 多步更新漂移的未通过诊断及全部日志。
\end{itemize}


指标来源：results/final/REPORT.md、summary.json 和 accuracy/comparison.json。原有两个项目未改动。
