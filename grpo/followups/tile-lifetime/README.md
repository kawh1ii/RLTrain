# TiledGRPO 实机扩展实验

本轮在迁移后的 RTX 4090 上验证两个问题：词表分块实现是否还有可解释的显存冗余，以及损失算子是否能接入每轮重新采样的 GRPO 更新。测量使用驱动 570.124.04、PyTorch 2.8.0+cu128 和 Triton 3.4.0。线上采样使用从源机器迁移的 Transformers 5.19.0 和通过原始 SHA-256 清单核验的 Qwen2.5-0.5B-Instruct 权重。

## 一块 logits 的额外峰值

原实现逐块计算 logits。Python 在赋值完成前会保留上次循环的 logits，反向中 grad 也别名到同一块存储。在下一块 GEMM 分配输出时，两块同时可达。4,096 × 32,768 个 BF16 元素恰好是 0.250 GiB。补丁在本块归约或梯度 GEMM 完成入队后释放引用，保持运算、遍历次序和分块宽度。

另实现可选的块宽选择器。单块估算为 `N × (2C + 8⌈C/4096⌉)` 字节，覆盖 BF16 logits 和两份 FP32 部分归约统计。80 MiB 预算从 32,768、16,384、8,192、4,096 中选择最大可放入块。显存口径为 PyTorch CUDA 峰值 allocated。预算仅覆盖列出的临时缓冲，整体峰值还包含权重、梯度、激活与库工作区。

先以种子 93、94、95 和 512/4,096 token 扫描四种词表块宽度并保留 eager 与 torch.compile 对照。随后冻结上述规则，以种子 197、198、199 验证三个形状。每次 3 次预热、11 次测量，表中为三次运行中位数。各组输入散列相同。

| 有效位置数 | 原实现 ms / GiB | 固定块宽，仅释放 ms / GiB | 释放并按 80 MiB 选块 ms / GiB |
| ---: | ---: | ---: | ---: |
| 1,024 | 8.689 / 0.654 | 8.700 / 0.592 | 8.675 / 0.592 |
| 2,048 | 16.832 / 0.784 | 16.805 / 0.659 | 17.136 / 0.596 |
| 4,096 | 34.574 / 1.044 | 34.577 / 0.794 | 34.188 / 0.607 |

固定块宽的独立检查覆盖三个形状与三个种子，loss、hidden 梯度与输出头梯度逐元素相等。原有 42 项测试通过。不同块宽单独和 dense 参考比较，保持原数值容差。4,096 token 时，单独释放节省 0.250 GiB，时间几乎持平。选择更小块进一步降低峰值，2,048 token 时也观察到约 2% 的延迟增加。

另在 4,096 token 的相同三个留出输入上跑同进程对照。dense 为 57.508 ms / 7.232 GiB，torch.compile 为 37.581 ms / 3.755 GiB，释放并选块后的 Triton 为 33.994 ms / 0.607 GiB。相对 torch.compile，算子峰值下降 83.8%，耗时下降 9.5%。原始组见 `grpo-cu128/final-paired`。

## 每轮刷新 rollout 的策略更新

每个种子从同一初始模型开始，三轮各采样 4 道固定算术题 × 4 条响应。对最终整数做 0/1 奖励，在题内计算相对优势。用当前 rollout 版本重算旧策略 log-prob，参考模型固定在版本 0。执行 Triton GRPO loss、反向及 AdamW，再从更新后的策略生成下一轮。每轮保留生成文本、完整 token、解析结果、优势、版本与实际权重变化。

最初的严格整段格式拒绝了包含正确算术结果的解释文本，3 轮均无奖励方差，实验留在 `grpo-cu128/online`。调整提示和末尾整数解析后，用新的训练题、留出题和采样种子重新运行。三次各完成 3 次非零更新。16 道独立贪心留出题的得分分别为 3→3、3→1、3→3，支持训练链路与真实任务奖励验证，模型能力改善仍无证据。

## 复现

以公开 RLTrain 仓库提交 f00119fd8c915691ddfeb2ba1e5dc01f756adc92 的 grpo 目录作为原始源码。CUDA 12.8 复现实验先应用 [基准脚本兼容补丁](torch28-benchmark.patch)，在独立候选副本应用 [释放补丁](release-logits-tile.patch)。Torch 2.8 的兼容分支只影响对照脚本的 AOT 配置，固定块宽和显存预算实验均走相同 Triton 路径。

```sh
bash run_liveness.sh /path/to/original /path/to/candidate /path/to/output /path/to/python
PYTHONPATH=/path/to/candidate/src:/path/to/candidate/tools python verify_lifetime.py /path/to/original /path/to/candidate
PYTHONPATH=/path/to/candidate/src python online_grpo.py --model /path/to/Qwen2.5-0.5B-Instruct --output online.json --seed 201
python audit_evidence.py
```

[原始测量与轨迹](grpo-cu128/) · [逐元素检查日志](rl-lifetime-exact.log)
