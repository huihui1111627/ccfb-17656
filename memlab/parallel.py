"""并行策略: 数据并行 (含 ZeRO 式状态分片) 与流水并行。

三种模式都复用 engine.TrainEngine 的层代码, 差异只在:
1. 参数/梯度/优化器状态的归属 (整卡 / 按 rank 分片 / 按 stage 切层);
2. 集合通信缓冲 (all-gather 权重、all-reduce 梯度、stage 间激活)。

DP:  每卡完整参数+梯度+状态, 反向后 all-reduce 梯度;
ZeRO: 参数保留完整 (前向需要), 梯度与优化器状态按 rank 等分,
      每步前向前 all-gather 一次权重 (模拟 ZeRO-3 行为);
PP:  层按 stage 切段, 每卡只持有本段参数, micro-batch 串行流过。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import ops
from .common import (
    CAT_ACT, CAT_COMM, CAT_GRAD, CAT_PARAM, CAT_WORK,
    PHASE_BUILD, PHASE_COMM, PHASE_FWD, PHASE_BWD, PHASE_OPT,
)
from .dtype import Precision, nbytes
from .engine import Checkpoint, TrainEngine, allocate_inputs
from .nn import Embedding, Module, ParamStore
from .optimizer import Optimizer
from .tensor import Context, Tensor


# ---------------------------------------------------------------------- #
# 工具
# ---------------------------------------------------------------------- #
def shard_shape(shape, world_size, rank):
    """把参数按第 0 维均分到 world_size 个 rank, 返回该 rank 的形状。"""
    first = shape[0]
    base = first // world_size
    rem = first % world_size
    start = rank * base + min(rank, rem)
    size = base + (1 if rank < rem else 1) - 1 + (1 if rank < rem else 0)
    # 上面等价写法保持直白:
    size = base + (1 if rank < rem else 0)
    if len(shape) == 1:
        return (size,)
    return (size,) + tuple(shape[1:])


def split_param_data(param: Tensor, world_size, rank):
    shape = shard_shape(param.shape, world_size, rank)
    base = param.shape[0] // world_size
    rem = param.shape[0] % world_size
    start = rank * (param.shape[0] // world_size) + min(rank, rem)
    if param.ctx.mode == "real":
        return param.data[start:start + shape[0]]
    return None


@dataclass
class ParallelConfig:
    mode: str = "single"          # single | dp | zero | pipeline
    world_size: int = 1
    zero_stage: int = 3           # ZeRO 模式下使用
    pipeline_microbatches: int = 2
    checkpoint_layers: Optional[set] = None


# ---------------------------------------------------------------------- #
# 单卡 / 数据并行 / ZeRO
# ---------------------------------------------------------------------- #
def _build_store_replicated(ctx, layers, prec, seed_base=0):
    store = ParamStore(ctx, prec)
    ctx.set_phase(PHASE_BUILD)
    for i, layer in enumerate(layers):
        for pname, shape in layer.params():
            store.declare(pname, shape)
        layer.build(ctx, store)
    return store


def run_data_parallel(
    layers, spec, batch, prec: Precision, optim: Optimizer,
    ctx: Context, world_size: int, mode: str,
    checkpoint_layers: Optional[set] = None,
):
    """在 *一张卡的视角* 跑一步 DP/ZeRO 训练, 返回该卡的上下文数据。"""
    ctx.set_phase(PHASE_BUILD)

    if mode == "zero":
        from .zero import ZeroShardedStore
        store = ZeroShardedStore(ctx, prec, world_size, rank=0)
        for layer in layers:
            for pname, shape in layer.params():
                store.declare(pname, shape)
            layer.build(ctx, store)
        # 梯度/状态按本卡分片形状分配
        owned_shapes = {
            name: shard_shape(shape, world_size, 0)
            for name, shape in store.shapes.items()
        }
        optim.build(ctx, store, owned_params=owned_shapes)

        # 逐层前向/反向后回收 all-gather 出的完整权重
        def _release_layer(layer):
            for pname, _ in layer.params():
                store.release_param(pname)

        engine = TrainEngine(
            layers, ctx, store, prec, checkpoint_layers,
            after_forward_hook=_release_layer,
            after_backward_hook=_release_layer,
        )
    else:
        store = _build_store_replicated(ctx, layers, prec)
        owned_shapes = None
        optim.build(ctx, store)
        engine = TrainEngine(layers, ctx, store, prec,
                             checkpoint_layers)

    # ---- 输入 ---- #
    inputs = allocate_inputs(ctx, spec, batch, prec)
    engine.train_step(inputs)

    # ---- 反向后通信 ---- #
    ctx.set_phase(PHASE_COMM)
    if mode == "dp":
        # all-reduce 梯度: 每卡已有完整梯度, 逐层产生等大通信缓冲
        for name, shape in store.shapes.items():
            buf = ctx.alloc(shape, prec.grad, CAT_COMM,
                            f"allreduce.grad:{name}")
            ctx.free_later(buf)
    else:  # zero: reduce-scatter, 通信量等于完整梯度 (逐层)
        for name, shape in store.shapes.items():
            buf = ctx.alloc(shape, prec.grad, CAT_COMM,
                            f"reducescatter.grad:{name}")
            ctx.free_later(buf)

    # ---- 优化器步 ---- #
    owned_names = (set(owned_shapes.keys()) if owned_shapes else None)
    optim.step(ctx, store, owned=owned_names)

    for t in inputs.values():
        if t.alive:
            ctx.free_later(t)
    return store, engine


# ---------------------------------------------------------------------- #
# 流水并行
# ---------------------------------------------------------------------- #
def _split_stages(layers, world_size):
    """按层数均匀切 stage, 嵌入层与首 stage 绑定。"""
    n = len(layers)
    bounds = [round(i * n / world_size) for i in range(world_size + 1)]
    stages = [layers[bounds[i]:bounds[i + 1]]
              for i in range(world_size)]
    return stages


def run_pipeline(
    layers, spec, batch, prec: Precision, optim: Optimizer,
    ctx: Context, world_size: int, microbatches: int,
    rank: int = 0, checkpoint_layers: Optional[set] = None,
):
    """单 rank 视角的流水并行一步 (GPipe 风格: 全前向再全反向)。"""
    stages = _split_stages(layers, world_size)
    my_layers = stages[rank]
    ctx.set_phase(PHASE_BUILD)
    store = ParamStore(ctx, prec)
    for layer in my_layers:
        for pname, shape in layer.params():
            store.declare(pname, shape)
        layer.build(ctx, store)
    optim.build(ctx, store)

    mb = max(1, batch // microbatches)
    assert batch % microbatches == 0, "batch 需能被 microbatch 数整除"

    # 本 rank 前向产出、要传给下一 stage 的边界激活
    send_bufs: List[Tensor] = []
    # 上一 stage 传进来的输入 (首个 microbatch 的输入由数据分配)
    recv_bufs: List[Tensor] = []

    wrapped = []
    for layer in my_layers:
        if (checkpoint_layers and layer.name in checkpoint_layers
                and not isinstance(layer, Embedding)):
            wrapped.append(Checkpoint(layer))
        else:
            wrapped.append(layer)

    saved_records = []

    # ---------------- 前向阶段 (所有 microbatch) ---------------- #
    ctx.set_phase(PHASE_FWD)
    for m_i in range(microbatches):
        if rank == 0:
            inputs = allocate_inputs(ctx, spec, mb, prec)
            x = inputs["x"] if spec.kind == "cnn" else inputs["tokens"]
        else:
            # 模拟从上一 stage recv: 用上一 stage 输出形状,
            # 形状通过 dry 形状推导获得 (见 _stage_boundary_shapes)
            in_shape = _stage_boundary_shapes(
                stages, spec, mb, prec, rank
            )
            x = ctx.alloc(in_shape, prec.compute, CAT_COMM,
                          f"recv.fwd.mb{m_i}")
            if ctx.mode == "real":
                x.data[:] = np.random.default_rng(
                    200 + m_i
                ).standard_normal(in_shape).astype(x.data.dtype)
            recv_bufs.append(x)
        records = []
        for layer in wrapped:
            out, saved = layer.forward(ctx, store, x)
            records.append((layer, saved))
            x = out
        if rank < world_size - 1:
            # 发给下一 stage 的边界缓冲 (本卡仍需保留用于反向)
            send = ctx.alloc(x.shape, x.dtype, CAT_COMM,
                             f"send.fwd.mb{m_i}")
            send_bufs.append(send)
        saved_records.append(records)

    # 最后一个 stage 在这里做 loss; 其他 stage 的反向梯度从末段回传
    if rank == world_size - 1:
        grads_in = []
        for m_i in range(microbatches):
            last_out = None
            # 末段输出即最后一层记录的输出, 重新由形状构造 loss 输入
            out_shape = _stage_boundary_shapes(
                stages, spec, mb, prec, world_size
            )
            # transformer 末段 lm_head 输出带序列维 (mb, S, V)
            if spec.kind != "cnn" and len(out_shape) == 2:
                out_shape = (mb, spec.seq_len, out_shape[1])
            logits = ctx.alloc(out_shape, prec.compute, CAT_WORK,
                               f"pp.logits.mb{m_i}")
            if ctx.mode == "real":
                logits.data[:] = np.random.default_rng(
                    300 + m_i
                ).standard_normal(out_shape).astype(logits.data.dtype)
            if spec.kind == "cnn":
                target = ctx.alloc((mb, out_shape[1]), prec.compute,
                                   CAT_COMM, f"pp.target.mb{m_i}")
                if ctx.mode == "real":
                    target.data[:] = np.random.default_rng(
                        50 + m_i
                    ).standard_normal(target.shape).astype(target.data.dtype)
                loss, loss_saved = TrainEngine._mse_forward(
                    ctx, logits, target
                )
                n_elem = int(np.prod(logits.shape))
                gy = TrainEngine._mse_backward(ctx, loss_saved, n_elem)
                # loss_saved = [diff, target], 一并回收
                for t in loss_saved:
                    ctx.free_later(t)
                ctx.free_later(loss)
            else:
                targets = ctx.alloc((mb * spec.seq_len,), "int32",
                                    CAT_COMM, f"pp.targets.mb{m_i}")
                if ctx.mode == "real":
                    targets.data[:] = np.random.default_rng(
                        100 + m_i
                    ).integers(0, spec.vocab, size=targets.shape)
                loss, probs, loss_saved = ops.softmax_ce_forward(
                    ctx, logits, targets
                )
                gy = ops.softmax_ce_backward(
                    ctx, probs, targets, targets.shape[0],
                    logits_shape=loss_saved[2],
                )
                ctx.free_later(probs)
                ctx.free_later(targets)
                ctx.free_later(loss)
            ctx.free_later(logits)
            grads_in.append(gy)
    else:
        grads_in = []
        for m_i in range(microbatches):
            g = ctx.alloc(send_bufs[m_i].shape, prec.grad, CAT_COMM,
                          f"recv.bwd.mb{m_i}")
            grads_in.append(g)

    # ---------------- 反向阶段 (逆序 microbatch) ---------------- #
    ctx.set_phase(PHASE_BWD)
    total_recompute_flops = 0
    for m_i in reversed(range(microbatches)):
        gy = grads_in[m_i]
        for layer, saved in reversed(saved_records[m_i]):
            before = ctx.flops
            incoming = gy
            gy = layer.backward(ctx, store, gy, saved)
            if (incoming is not None and incoming.alive
                    and incoming.category == CAT_WORK):
                ctx.free_later(incoming)
            if isinstance(layer, Checkpoint):
                total_recompute_flops += ctx.flops - before
            from .engine import release_saved_activations
            release_saved_activations(ctx, saved)
        if gy is not None and gy.alive and gy.category == CAT_WORK:
            ctx.free_later(gy)
        if rank < world_size - 1:
            ctx.free_later(send_bufs[m_i])
        if rank > 0 and gy is not None:
            # 回传给上一 stage 的梯度缓冲
            ctx.alloc(gy.shape, gy.dtype, CAT_COMM,
                      f"send.bwd.mb{m_i}")
        if rank > 0:
            ctx.free_later(recv_bufs[m_i])

    # ---------------- 优化器 ---------------- #
    ctx.set_phase(PHASE_OPT)
    optim.step(ctx, store)
    for t in recv_bufs:
        if t.alive:
            ctx.free_later(t)
    return store, total_recompute_flops


def _stage_boundary_shapes(stages, spec, mb, prec, boundary):
    """用 spec 张量做一次纯形状推导, 求第 boundary 个 stage 的输出形状。"""
    from .tensor import Tensor
    desc = spec.input_shape_fn(mb)
    if spec.kind == "cnn":
        shape = desc["x"]["shape"]
        dtype = prec.compute
        x = Tensor(shape, dtype, -1, None, "shape.x", "", None)
        for layer in [l for stage in stages[:boundary] for l in stage]:
            # 形状推导只需要 forward 的形状语义; 用轻量规则推演
            x = _infer_shape(layer, x)
        return x.shape
    else:
        seq = spec.seq_len
        x = Tensor((mb, seq), "int32", -1, None, "shape.tok", "", None)
        for layer in [l for stage in stages[:boundary] for l in stage]:
            x = _infer_shape(layer, x)
        return x.shape


def _infer_shape(layer, x):
    """层输出形状的纯规则推导 (不分配显存)。"""
    from .nn import (
        Conv2d, Embedding, Flatten, GELU, LayerNorm, Linear, MaxPool2d,
        MultiHeadAttention, ReLU, TransformerBlock,
    )
    if isinstance(layer, Embedding):
        return Tensor((x.shape[0], x.shape[1], layer.dim), x.dtype,
                      -1, None, "s", "", None)
    if isinstance(layer, Linear):
        return Tensor((x.shape[0], layer.n), x.dtype, -1, None,
                      "s", "", None)
    if isinstance(layer, (ReLU, GELU, LayerNorm, TransformerBlock)):
        return Tensor(x.shape, x.dtype, -1, None, "s", "", None)
    if isinstance(layer, MultiHeadAttention):
        return Tensor(x.shape, x.dtype, -1, None, "s", "", None)
    if isinstance(layer, MaxPool2d):
        N, C, H, W = x.shape
        Ho = (H - layer.kernel) // layer.stride + 1
        Wo = (W - layer.kernel) // layer.stride + 1
        return Tensor((N, C, Ho, Wo), x.dtype, -1, None, "s", "", None)
    if isinstance(layer, Conv2d):
        N, C, H, W = x.shape
        Ho = (H + 2 * layer.padding - layer.kh) // layer.stride + 1
        Wo = (W + 2 * layer.padding - layer.kw) // layer.stride + 1
        return Tensor((N, layer.cout, Ho, Wo), x.dtype, -1, None,
                      "s", "", None)
    if isinstance(layer, Flatten):
        return Tensor((x.shape[0], int(np.prod(x.shape[1:]))),
                      x.dtype, -1, None, "s", "", None)
    if isinstance(layer, Checkpoint):
        return _infer_shape(layer.layer, x)
    raise NotImplementedError(f"形状推导未覆盖: {type(layer)}")
