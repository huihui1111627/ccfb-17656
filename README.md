# memlab · 训练显存预估实验台

提交训练任务前, 不真跑就能回答三个问题:

1. **这个配置能不能跑?** —— 给定模型/批大小/序列长度/精度/优化器/并行方式, 给出一步训练每个时刻显存里有什么、峰值出现在哪一步;
2. **峰值会用多少?** —— 参数、梯度、优化器状态、保存的激活、临时工作区、通信缓冲、碎片, 逐字节记账;
3. **最少要几张卡?** —— 给定批大小求最少 GPU, 或给定卡型求最大批大小。

核心保证: **预估器与真实小框架共用同一份层代码和同一个分块分配器**,
预估路径用 `SpecTensor`(只推形状、不做数值计算), 真跑路径用 `RealTensor`(NumPy 实算),
两条路径产生逐条相同的 alloc/free 事件。在 360 个组合上, 预估与真跑的峰值
**逐字节相等、OOM 判定零错误**。

## 目录结构

```
memlab/
  common.py        显存类别与训练阶段常量
  dtype.py         fp32/fp16/bf16/混合精度的字节数与预设
  allocator.py     分块显存分配器 (取整/分裂/合并/碎片统计), 预估与真跑共用
  tensor.py        Context + Spec/Real 双模式张量
  ops.py           函数式算子 (linear/conv/attention/layernorm/...)
  nn.py            层模块、ParamStore、TransformerBlock/MHA
  engine.py        一步训练引擎、gradient checkpointing 包装器
  optimizer.py     SGD/Adam, 优化器状态与混合精度主权重
  parallel.py      单卡 / 数据并行 / ZeRO-3 / 流水并行
  zero.py          ZeRO-3 分片参数存储 (逐层 all-gather)
  estimator.py     预估器: 从事件流重建完整显存时间线
  framework.py     真跑框架: NumPy 实算 + 真实 OOM 判定
  solver.py        最大批大小 / 最少卡数 求解器
  checkpoint.py    逐层 checkpoint 成本/收益分析与选择建议
  hardware.py      GPU 卡型库 (A100/H100/T4/...)
  models.py        small_cnn / tiny_gpt / long_gpt 三种结构
  cli.py           命令行入口
webapp/            零依赖 http.server 后端 + 原生 JS/Canvas 页面
experiments/       批量验证脚本
tests/             一致性测试与数值收敛测试
results/           验证结果 JSON
```

## 快速开始

```bash
# 命令行预估
python -m memlab.cli estimate --model tiny_gpt --batch 8 --gpu T4
python -m memlab.cli max-batch --model long_gpt_real --gpu T4
python -m memlab.cli min-gpus --model long_gpt_real --batch 64 --gpu T4
python -m memlab.cli checkpoint --model long_gpt_real --batch 4 --gpu T4

# 真跑一步 (NumPy 实算, 对照用)
python -m memlab.cli real --model small_cnn --batch 8

# 批量验证 (360 组合, 预估 vs 真跑)
python -m experiments.run_validation --out results/validation.json

# 网页
python -m webapp.server --port 8765   # 打开 http://127.0.0.1:8765

# 测试
python -m tests.test_spec_vs_real       # 事件序列与峰值一致性
python -m tests.test_training_converges # 真跑 loss 收敛
```

## 设计要点

### 为什么预估能和真跑严格一致?

同一份算子代码只在 `ctx.mode == "real"` 时做 NumPy 运算, 否则只推形状;
所有张量都经同一个 `BlockAllocator.alloc/free`。两条路径的调用序列完全相同,
因此分配器的取整、块分裂/合并、复用、OOM 判定都逐条一致。
任何偏差都会在 `test_spec_vs_real` 里立刻暴露。

### 分块分配器与碎片

模拟 PyTorch caching allocator: 小分配 (≤1MiB) 向上取整到 512B,
大分配取整到 2MiB; 块可分裂、相邻空闲块合并、同尺寸优先复用。
碎片拆成两个可加的量 (在峰值 reserved 同一时刻记账):

- **内部碎片**: 活跃块取整浪费 (`block_size - request`);
- **外部碎片**: 已预留但空闲、暂时无人能用的块。

恒等式 `reserved = allocated + 内部碎片 + 外部碎片` 始终成立。
容量判定按 `reserved`(而非 `allocated`) 走, 因此碎片真的会让可用容量变少。

### gradient checkpointing

`Checkpoint` 包装器前向只保留段输入与段输出、丢弃段内激活,
反向时重算一次前向再反传。分析器逐层给出"省多少 MiB / 多花多少 FLOP",
并贪心推荐组合。长序列模型上, 中间的 Transformer 块性价比最高
(靠近输入和靠近 loss 的块收益低)。

### 并行方式

- **单卡 / 数据并行 (DP)**: 每卡完整参数+梯度+状态, 反向后 all-reduce;
- **ZeRO-3**: 参数/梯度/状态全部按 rank 分片, 前向逐层 all-gather 权重、
  用完即释放, 层代码通过 `ZeroShardedStore` 无感知接入;
- **流水并行 (GPipe 风格)**: 模型按层切段, micro-batch 串行流过,
  逐 rank 推演各自峰值。

### 验证结论 (results/validation.json)

| 指标 | 结果 |
|---|---|
| 组合数 (3 模型 × 批大小 × 精度 × 优化器 × 并行 × checkpoint) | 360 |
| 能跑/不能跑判定错误 | **0** |
| 峰值相对误差 最大 / 均值 | **0.0% / 0.0%** |
| 精确字节匹配 | **360 / 360** |

## 边界与取舍

- 这是显存**记账模型**, 不是真实 CUDA 程序。取整阈值、块策略是对
  PyTorch caching allocator 的简化模拟; 真实框架还有 CUDA context、
  捕获池、库内部 workspace 等额外开销, 由 `GPUSpec.usable_ratio`(默认 0.92)
  统一扣减。
- 真跑在宿主机用 NumPy, bf16/fp16 统一用 float32 计算 (字节记账仍按
  模拟 dtype), 因此只验证显存事件与梯度数学, 不反映低精度数值误差。
- 长序列生产规模 (`long_gpt`, S=1024) 只走预估; 真跑对照用同结构的
  `long_gpt_real` (S=512) 以控制宿主机内存与耗时。
