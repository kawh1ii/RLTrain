# RLTrain：GRPO 大词表损失算子与显存优化

实现词表分块、Triton 归约和显式反向。RTX 4090、BF16、N=4096、H=896、V=151936 的损失前反向峰值显存相比 torch.compile 由 3.755 降至 1.044 GiB；真实模型训练步省显存，BF16 多步更新漂移仍保留。

- [实现与复现步骤](grpo/README.md)
- [实验结果和适用边界](grpo/RESULTS.md)
- [设计与实现细节](grpo/docs/DESIGN.md)
- [源码](grpo/src) · [测试](grpo/tests) · [原始实验记录](grpo/results)

当前实现及其测量证据在 `grpo/`。旧实现与实验仍可从[此前的项目说明](README.previous.md)查看。

- [TiledGRPO 缓冲生命周期与在线策略更新实验](grpo/followups/tile-lifetime/README.md)
