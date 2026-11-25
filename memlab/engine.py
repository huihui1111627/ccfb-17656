"""一步训练的执行引擎: 前向、反向、checkpoint 重算。

关键设计: spec 与 real 两种模式执行 *完全相同* 的 Python 调用序列,
只在 Context.alloc 返回的张量种类上不同, 因而 alloc/free 事件逐条一致。

gradient checkpointing 由 Checkpoint 包装器实现:
- 前向: 正常跑一遍被包装层, 但只保留 *段输入* 与 *段输出*,
  层内部保存给反向的激活全部立即释放;
- 反向: 用段输入重新前向一遍, 再做反向, 额外花一份前向 FLOP。
"""
from __future__ import annotations

from typing import List, Optional, Set

import numpy as np

from . import ops
from .common import (
    CAT_ACT, CAT_INPUT, CAT_WORK,
    PHASE_BWD, PHASE_FWD,
)
from .dtype import Precision
from .nn import Embedding, Module, ParamStore
from .tensor import Context, Tensor


def allocate_inputs(ctx: Context, spec, batch: int, prec: Precision):
    """按模型输入描述分配本步输入 (token / 图像 / 标签)。"""
    desc = spec.input_shape_fn(batch)
    out = {}
    if spec.kind == "cnn":
        x = ctx.alloc(desc["x"]["shape"], prec.compute, CAT_INPUT,
                      "input.x")
        y = ctx.alloc(desc["y"]["shape"], prec.compute, CAT_INPUT,
                      "input.target")
        if ctx.mode == "real":
            x.data[:] = np.random.default_rng(1).standard_normal(
                x.shape
            ).astype(x.data.dtype)
            y.data[:] = np.eye(10)[
                np.random.default_rng(2).integers(0, 10, size=batch)
            ].astype(y.data.dtype)
        out["x"], out["y"] = x, y
    else:
        seq = spec.seq_len
        tokens = ctx.alloc((batch, seq), "int32", CAT_INPUT,
                           "input.tokens")
        targets = ctx.alloc((batch * seq,), "int32", CAT_INPUT,
                            "input.targets")
        if ctx.mode == "real":
            rng = np.random.default_rng(3)
            tokens.data[:] = rng.integers(0, spec.vocab,
                                          size=(batch, seq))
            targets.data[:] = rng.integers(0, spec.vocab,
                                           size=batch * seq)
        out["tokens"], out["targets"] = tokens, targets
    return out


def _collect_tensors(obj, bag):
    if isinstance(obj, Tensor):
        bag.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_tensors(v, bag)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_tensors(v, bag)


def release_saved_activations(ctx: Context, saved):
    """释放 saved 结构里所有仍存活的激活张量。"""
    bag = []
    _collect_tensors(saved, bag)
    for t in bag:
        if t.alive and t.category == CAT_ACT:
            ctx.free_later(t)


# ---------------------------------------------------------------------- #
class Checkpoint(Module):
    """gradient checkpointing 包装器。"""

    def __init__(self, layer: Module):
        self.name = f"ckpt:{layer.name}"
        self.layer = layer
        self.ckpt_input: Optional[Tensor] = None

    def params(self):
        return self.layer.params()

    def build(self, ctx, store):
        self.layer.build(ctx, store)

    def forward(self, ctx, store, x):
        # 段输入必须独立存活到反向: 复制一份同形张量占位,
        # 原 x 作为上游层的 saved 激活可被引擎正常回收
        self.ckpt_input = ctx.alloc(
            x.shape, x.dtype, CAT_ACT, f"{self.name}.input_keep"
        )
        if ctx.mode == "real":
            self.ckpt_input.data[:] = x.data
        out, saved = self.layer.forward(ctx, store, x)
        # 段内部激活立即释放, 只留段输出 (out)
        release_saved_activations(ctx, saved)
        return out, {"ckpt": True}

    def backward(self, ctx, store, gy, saved):
        flops_before = ctx.flops
        out, recomputed = self.layer.forward(
            ctx, store, self.ckpt_input
        )
        gx = self.layer.backward(ctx, store, gy, recomputed)
        ctx.free_later(out)
        release_saved_activations(ctx, recomputed)
        # 段输入占位张量完成使命
        if self.ckpt_input.alive:
            ctx.free_later(self.ckpt_input)
        self.ckpt_input = None
        ctx.add_flops(0)  # 重算 FLOP 已自然计入 ctx.flops
        return gx


# ---------------------------------------------------------------------- #
class TrainEngine:
    def __init__(self, layers: List[Module], ctx: Context,
                 store: ParamStore, precision: Precision,
                 checkpoint_layers: Optional[Set[str]] = None,
                 after_forward_hook=None, after_backward_hook=None):
        self.ckpt_names = checkpoint_layers or set()
        self.layers = []
        for layer in layers:
            if layer.name in self.ckpt_names and not isinstance(
                layer, Embedding
            ):
                self.layers.append(Checkpoint(layer))
            else:
                self.layers.append(layer)
        self.ctx = ctx
        self.store = store
        self.prec = precision
        # 并行策略钩子 (ZeRO 用它逐层回收 all-gather 权重)
        self.after_forward_hook = after_forward_hook
        self.after_backward_hook = after_backward_hook

    @property
    def recompute_flops(self):
        """重算引入的额外 FLOP (= 重算层前向 FLOP 之和)。"""
        total = 0
        # 由调用方在 step 前后用 ctx.flops 差值亦可, 这里显式统计:
        return getattr(self, "_recompute_flops", 0)

    def train_step(self, inputs: dict):
        ctx = self.ctx
        is_cnn = "x" in inputs

        # ---- 前向 ---- #
        ctx.set_phase(PHASE_FWD)
        x = inputs["x"] if is_cnn else inputs["tokens"]
        records = []
        for layer in self.layers:
            out, saved = layer.forward(ctx, self.store, x)
            if self.after_forward_hook is not None:
                self.after_forward_hook(layer)
            records.append((layer, saved))
            x = out

        # ---- Loss ---- #
        if is_cnn:
            loss, loss_saved = self._mse_forward(ctx, x, inputs["y"])
        else:
            loss, probs, loss_saved = ops.softmax_ce_forward(
                ctx, x, inputs["targets"]
            )
        # 在释放前取出标量值 (real 模式有效; spec 模式为 None)
        loss_value = float(loss.data[0]) if ctx.mode == "real" else None
        ctx.free_later(loss)

        # ---- 反向 ---- #
        ctx.set_phase(PHASE_BWD)
        if is_cnn:
            n_elem = int(np.prod(x.shape))
            gy = self._mse_backward(ctx, loss_saved, n_elem)
            for t in loss_saved:
                ctx.free_later(t)
        else:
            probs, targets, logits_shape = loss_saved
            gy = ops.softmax_ce_backward(
                ctx, probs, targets, targets.shape[0],
                logits_shape=logits_shape,
            )
            ctx.free_later(probs)

        for layer, saved in reversed(records):
            flops_before = ctx.flops
            incoming = gy
            gy = layer.backward(ctx, self.store, gy, saved)
            # 层的输入梯度是工作区张量, 层本身不释放, 由引擎统一回收
            if (incoming is not None and incoming.alive
                    and incoming.category == CAT_WORK):
                ctx.free_later(incoming)
            if isinstance(layer, Checkpoint):
                # 记录重算额外开销
                self._recompute_flops = getattr(
                    self, "_recompute_flops", 0
                ) + (ctx.flops - flops_before)
            if self.after_backward_hook is not None:
                self.after_backward_hook(layer)
            release_saved_activations(ctx, saved)
        # 最后一层 (embedding/首层) 的输入梯度也回收 (可能为 None)
        if gy is not None and gy.alive and gy.category == CAT_WORK:
            ctx.free_later(gy)
        return loss_value

    # ------------------------------------------------------------------ #
    @staticmethod
    def _mse_forward(ctx, logits, target):
        diff = ctx.alloc(logits.shape, logits.dtype, CAT_WORK, "mse.diff")
        loss = ctx.alloc((1,), "fp32", CAT_WORK, "mse.loss")
        if ctx.mode == "real":
            diff.data[:] = logits.data - target.data
            loss.data[:] = (diff.data.astype(np.float32) ** 2).mean(
                keepdims=True
            )
        return loss, [diff, target]

    @staticmethod
    def _mse_backward(ctx, saved, n_elem):
        diff, _ = saved
        gl = ctx.alloc(diff.shape, diff.dtype, CAT_WORK, "mse.grad")
        if ctx.mode == "real":
            gl.data[:] = (2.0 / n_elem) * diff.data
        return gl
