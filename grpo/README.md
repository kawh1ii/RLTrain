# RLTrain v2：GRPO 大词表损失算子与显存优化

将原来的多卡架构对比主线，收敛为一个可在单张 RTX 4090 上审查和复现的训练计算模块：**分块线性投影、稳定 log-sum-exp、GRPO 损失及显式反向传播**。

自己实现 Triton 归约/损失梯度/原地 logits 梯度内核与自定义 autograd；GEMM 使用 PyTorch/cuBLAS。不是新提出的 GRPO 算法，也不将 Liger 已有的分块损失思想记作自己的创新。

优先阅读 [实测结果与边界](RESULTS.md)、[设计和推导](docs/DESIGN.md)、[数值排错记录](docs/DEBUG_LOG.md)。**本原型的混合精度多步更新存在数值漂移，不声称训练轨迹等价、无损生产替换或解题能力提升。**

## 重写了什么

| 入口 | 职责 |
|---|---|
| `src/rltrain/kernels.py` | 分段 LSE、跨词表块合并、选中 token 的损失/导数、原地 logits 梯度 |
| `src/rltrain/fused.py` | 仅保存 hidden/weight/标签/LSE/一维导数的 autograd；词表分块重算；小缓冲累加 hidden 梯度 |
| `src/rltrain/objective.py` | 独立 PyTorch 参考、分组优势、EOS 掩码、全优化器批归一化、checkpoint 对照 |
| `src/rltrain/batch.py` | 因果预测位置、冻结 rollout/old/reference 版本和微批边界 |
| `src/rltrain/baselines.py` | eager、torch.compile、checkpoint、手写 Torch、Triton、固定版本 Liger 对照 |
| `tools/compare_training.py` | 全模型初始权重、全部梯度和实际参数更新向量检查；精度运行不供性能数字 |
| `tools/train_step.py` | 真实 Qwen2.5-0.5B 前向、损失、反向、FP32 master/BF16 AMP 和 fused AdamW |
| `tools/benchmark_loss.py` | 独立算子 forward+backward、完整时间样本、allocated/reserved 显存及输入不变性 |
| `tools/full_validation.py` | 冻结源码、顺序管理单卡、三次重复、分块消融与失败保留 |

默认按词表分块，块宽 32768；早期按 token 分块的实现保留为 `strategy="sequence"` 对照。自定义反向只支持一阶导数，old/ref/advantage/weights 必须为冻结元数据。公开入口默认做数值和标签范围检查；实验先对同一批输入统一检查，再在计时段关闭重复检查。

## 基线与正确性

- PyTorch eager 和完整 `torch.compile` forward/backward 都参与比较，避免仅战胜朴素大矩阵写法。
- checkpoint 分块重计算、同一手写反向的 Torch 版本用于区分遍历次序、重算与 Triton 融合的作用。
- Liger 版本固定在 `8c7a5078318da1295dc3ca445f9d85e9e9e153f5`；核对 285 个源码文件散列和实际导入路径。显式选择 `loss_type="grpo"`、相同 beta/epsilon/掩码/归一化，计时包含其 forward 中预计算的反向。
- FP64 有限差分和三步 Adam 更新作为公式正确性检查；GPU 按 FP32/BF16 的预定容差检查。真实模型检查所有参数的首步梯度，而非只看 loss 或几个梯度切片。
- 真实 BF16 的三步更新向量未达到最初的等价阈值，连 eager/compile 对照也出现此现象。旧失败记录和未放宽的原阈值保留；多步差异为诊断，不能据此宣称收敛或训练轨迹等价。

## 数据与测量范围

GSM8K 固定公开版本、前四个满足提示词长度上限的训练问题，每题采样四个真实响应。奖励预先定义为 `####` 数字正确性减去 `0.001 × 响应长度`。128-token 批全部截断、优势全零，保留但不用于训练验证；相同题目的 256-token 批产生 3830 个有效响应 token 和 16 个非零优势。该批数学正确性奖励全零，非零信号来自长度项。

冻结轨迹的多轮 PPO 更新用于检查计算路径与资源占用；不代表完成了一个有收敛结果的数学强化学习实验。生成和奖励准备不计入“训练步耗时”，不能把算子或训练步变化写成完整 RL 流水线吞吐变化。

## 安装与复现

实测依赖：Linux、RTX 4090 24GB、PyTorch 2.13.0+cu130、Triton 3.7.1、Transformers 5.19.0。`requirements.lock` 是测量环境快照，不能当作通用的跨平台依赖锁。

在已准备好相同 CUDA/PyTorch 依赖的隔离环境中安装工程：

```bash
python -m pip install -e '.[gpu,model]'
git clone https://github.com/linkedin/Liger-Kernel.git work/liger
git -C work/liger checkout 8c7a5078318da1295dc3ca445f9d85e9e9e153f5
export RLTRAIN_LIGER_PATH="$PWD/work/liger/src"
python -m unittest discover -s tests -v
python tools/full_validation.py \
  --model /path/to/Qwen2.5-0.5B-Instruct \
  --data "$PWD/data/rollouts256" \
  --liger "$RLTRAIN_LIGER_PATH" --output results/reproduction
python tools/report.py --root results/reproduction
```

为了直接复算随包数据，使用 Python 标准库即可生成表格，无需加载模型：

```bash
python tools/report.py --root results/final
python docs/verify_experiments.py
```

要重新生成响应，可先运行 `tools/prepare_data.py`，再运行 `tools/make_rollouts.py --tokens 256`，输出至新目录。新生成内容可能因浮点执行路径不同而变化；跨实现比较应重放同一份轨迹文件。

本次远端工程为 `/root/autodl-tmp/rltrain-v2`，复用依赖解释器 `/root/autodl-tmp/turbollm-v2/.venv/bin/python`；Liger 位于 `/root/autodl-tmp/liger-baseline/src`。模型权重不包含在轻量交付包内，文件清单和 SHA-256 见 `results/models/`。

## 原多卡项目

旧 8×4090 的 veRL 架构对比可作为系统搭建与通信诊断背景，另存于 `legacy/`。新版的显存/时间结论全部来自本轮单卡实测，不沿用旧版 2.13×。4090 不支持 NVLink 的事实不使原实验无效，但其通信路径与数据中心互联不同，旧架构对比也不能做单变量因果解释。

上游依据：[NVIDIA 4090 规格](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)、[NCCL 通信路径](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)、[Liger 固定源码](https://github.com/linkedin/Liger-Kernel/tree/8c7a5078318da1295dc3ca445f9d85e9e9e153f5)、[PyTorch 编译反向与 AMP](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_backward.html)、[GSM8K 固定来源](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c)。
