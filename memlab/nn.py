"""层模块与模型结构描述。

ParamStore 是参数/梯度/优化器状态的持有者:
- 单卡与数据并行: 参数整体在本卡;
- ZeRO: 参数/梯度/状态按卡分片, 前向时 all-gather 出临时完整权重;
- 流水并行: 只有属于本 stage 的层在这里建参数。

层代码不感知并行方式, 只通过 store.get_param(层名) 拿当前可用权重。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import ops
from .common import (
    CAT_ACT, CAT_GRAD, CAT_PARAM, CAT_WORK,
)
from .dtype import Precision, nbytes
from .tensor import Context, Tensor


# ---------------------------------------------------------------------- #
# 参数存储
# ---------------------------------------------------------------------- #
class ParamStore:
    def __init__(self, ctx: Context, precision: Precision):
        self.ctx = ctx
        self.prec = precision
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
        self.shapes: Dict[str, tuple] = {}

    def declare(self, name, shape):
        """注册参数 (是否分片由 build 阶段决定, 见 parallel.py)。"""
        self.shapes[name] = tuple(int(d) for d in shape)

    def alloc_param(self, name, shape, dtype=None, seed=0):
        t = ops.init_weight(
            self.ctx, shape, dtype or self.prec.param, CAT_PARAM,
            f"param:{name}", seed=seed,
        )
        self.params[name] = t
        self.shapes[name] = t.shape
        return t

    def alloc_grad(self, name, shape, dtype=None):
        t = self.ctx.alloc(
            shape, dtype or self.prec.grad, CAT_GRAD, f"grad:{name}"
        )
        # 梯度槽靠 += 累加, 必须零初始化 (np.empty 否则是垃圾内存)
        if self.ctx.mode == "real":
            t.data.fill(0)
        self.grads[name] = t
        return t

    def get_param(self, name) -> Tensor:
        return self.params[name]

    def _shard_start(self, first_dim: int) -> int:
        """普通模式不分片, 起始偏移恒为 0。"""
        return 0

    def release_param(self, name):
        """普通模式权重常驻, 无需释放 (ZeRO 子类覆盖)。"""
        pass

    def get_grad_slot(self, name) -> Optional[Tensor]:
        return self.grads.get(name)

    def accumulate_grad(self, name, work_grad: Tensor):
        """把反向算出的工作区梯度累加进持久梯度槽, 并释放工作张量。

        普通模式槽与工作张量同形; ZeRO 模式槽是第 0 维分片,
        只累加属于本 rank 的那一段。
        """
        slot = self.grads[name]
        if self.ctx.mode == "real":
            if work_grad.shape == slot.shape:
                slot.data[:] += work_grad.data
            else:
                # 第 0 维分片: 由分片形状反推起止
                full = work_grad.shape[0]
                part = slot.shape[0]
                start = getattr(self, "_shard_start", lambda d: 0)(full)
                idx = tuple(
                    slice(start, start + part) if i == 0
                    else slice(None)
                    for i in range(work_grad.data.ndim)
                )
                slot.data[:] += work_grad.data[idx].reshape(slot.shape)
        self.ctx.free_later(work_grad)

    def persistent_blocks(self) -> set:
        ids = {t.block_id for t in self.params.values()}
        ids.update(t.block_id for t in self.grads.values())
        return ids


# ---------------------------------------------------------------------- #
# 层基类
# ---------------------------------------------------------------------- #
class Module:
    name: str = "module"

    def params(self):
        """返回 [(param_name, shape)]。"""
        return []

    def build(self, ctx, store):
        """创建本层参数 (由并行构建器决定整体/分片)。"""

    def forward(self, ctx, store, x: Tensor) -> Tuple[Tensor, list]:
        raise NotImplementedError

    def backward(self, ctx, store, gy: Tensor, saved: list) -> Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------- #
# Linear
# ---------------------------------------------------------------------- #
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, name="linear"):
        self.name = name
        self.k = in_features
        self.n = out_features
        self.use_bias = bias

    def params(self):
        ps = [(f"{self.name}.w", (self.n, self.k))]
        if self.use_bias:
            ps.append((f"{self.name}.b", (self.n,)))
        return ps

    def build(self, ctx, store):
        store.alloc_param(f"{self.name}.w", (self.n, self.k), seed=ops.stable_seed(self.name))
        if self.use_bias:
            store.alloc_param(f"{self.name}.b", (self.n,),
                              seed=ops.stable_seed(self.name, 7))

    def forward(self, ctx, store, x):
        w = store.get_param(f"{self.name}.w")
        b = store.get_param(f"{self.name}.b") if self.use_bias else None
        out, saved = ops.linear_forward(ctx, x, w, b)
        return out, saved

    def backward(self, ctx, store, gy, saved):
        w = store.get_param(f"{self.name}.w")
        gx, gw, gb = ops.linear_backward(ctx, gy, saved, w, self.use_bias)
        store.accumulate_grad(f"{self.name}.w", gw)
        if self.use_bias:
            store.accumulate_grad(f"{self.name}.b", gb)
        return gx


# ---------------------------------------------------------------------- #
# 激活
# ---------------------------------------------------------------------- #
class ReLU(Module):
    name = "relu"

    def forward(self, ctx, store, x):
        return ops.relu_forward(ctx, x)

    def backward(self, ctx, store, gy, saved):
        return ops.relu_backward(ctx, gy, saved)


class GELU(Module):
    name = "gelu"

    def forward(self, ctx, store, x):
        return ops.gelu_forward(ctx, x)

    def backward(self, ctx, store, gy, saved):
        return ops.gelu_backward(ctx, gy, saved)


# ---------------------------------------------------------------------- #
# Conv / Pool
# ---------------------------------------------------------------------- #
class Conv2d(Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0,
                 bias=True, name="conv"):
        self.name = name
        self.cin, self.cout = in_ch, out_ch
        self.kh = self.kw = kernel
        self.stride, self.padding = stride, padding
        self.use_bias = bias

    def params(self):
        ps = [(f"{self.name}.w", (self.cout, self.cin, self.kh, self.kw))]
        if self.use_bias:
            ps.append((f"{self.name}.b", (self.cout,)))
        return ps

    def build(self, ctx, store):
        store.alloc_param(
            f"{self.name}.w",
            (self.cout, self.cin, self.kh, self.kw),
            seed=ops.stable_seed(self.name),
        )
        if self.use_bias:
            store.alloc_param(f"{self.name}.b", (self.cout,),
                              seed=ops.stable_seed(self.name, 3))

    def forward(self, ctx, store, x):
        w = store.get_param(f"{self.name}.w")
        b = store.get_param(f"{self.name}.b") if self.use_bias else None
        return ops.conv2d_forward(ctx, x, w, b, self.stride, self.padding)

    def backward(self, ctx, store, gy, saved):
        w = store.get_param(f"{self.name}.w")
        gx, gw, gb = ops.conv2d_backward(
            ctx, gy, saved, w,
            (self.cout, self.cin, self.kh, self.kw),
            self.stride, self.padding, self.use_bias,
        )
        store.accumulate_grad(f"{self.name}.w", gw)
        if self.use_bias:
            store.accumulate_grad(f"{self.name}.b", gb)
        return gx


class MaxPool2d(Module):
    def __init__(self, kernel, stride=None):
        self.name = "maxpool"
        self.kernel = kernel
        self.stride = stride or kernel

    def forward(self, ctx, store, x):
        return ops.maxpool2d_forward(ctx, x, self.kernel, self.stride)

    def backward(self, ctx, store, gy, saved):
        return ops.maxpool2d_backward(ctx, gy, saved)


class Flatten(Module):
    name = "flatten"

    def forward(self, ctx, store, x):
        out = ctx.alloc(
            (x.shape[0], int(np.prod(x.shape[1:]))),
            x.dtype, CAT_ACT, "flatten.out",
        )
        if ctx.mode == "real":
            out.data[:] = x.data.reshape(out.shape)
        return out, [x.shape]

    def backward(self, ctx, store, gy, saved):
        (x_shape,) = saved
        gx = ctx.alloc(x_shape, gy.dtype, CAT_WORK, "flatten.gx")
        if ctx.mode == "real":
            gx.data[:] = gy.data.reshape(x_shape)
        return gx


# ---------------------------------------------------------------------- #
# LayerNorm / Embedding
# ---------------------------------------------------------------------- #
class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5, name="ln"):
        self.name = name
        self.dim = dim
        self.eps = eps

    def params(self):
        return [(f"{self.name}.gamma", (self.dim,)),
                (f"{self.name}.beta", (self.dim,))]

    def build(self, ctx, store):
        store.alloc_param(f"{self.name}.gamma", (self.dim,),
                          seed=ops.stable_seed(self.name))
        store.alloc_param(f"{self.name}.beta", (self.dim,),
                          seed=ops.stable_seed(self.name, 11))

    def forward(self, ctx, store, x):
        g = store.get_param(f"{self.name}.gamma")
        b = store.get_param(f"{self.name}.beta")
        out, saved, temps = ops.layernorm_forward(ctx, x, g, b, self.eps)
        for t in temps:
            ctx.free_later(t)
        return out, saved

    def backward(self, ctx, store, gy, saved):
        gamma = store.get_param(f"{self.name}.gamma")
        gx, gg, gb = ops.layernorm_backward(
            ctx, gy, saved, gamma, saved[0].shape
        )
        store.accumulate_grad(f"{self.name}.gamma", gg)
        store.accumulate_grad(f"{self.name}.beta", gb)
        return gx


class Embedding(Module):
    def __init__(self, vocab, dim, name="embed"):
        self.name = name
        self.vocab = vocab
        self.dim = dim

    def params(self):
        return [(f"{self.name}.table", (self.vocab, self.dim))]

    def build(self, ctx, store):
        store.alloc_param(f"{self.name}.table", (self.vocab, self.dim),
                          seed=42)

    def forward(self, ctx, store, tokens):
        table = store.get_param(f"{self.name}.table")
        return ops.embedding_forward(ctx, tokens, table, self.vocab,
                                     tokens.shape[1])

    def backward(self, ctx, store, gy, saved):
        (tokens,) = saved
        g = ops.embedding_backward(
            ctx, gy, tokens, (self.vocab, self.dim)
        )
        store.accumulate_grad(f"{self.name}.table", g)
        return None


# ---------------------------------------------------------------------- #
# 多头注意力 (组合模块)
# ---------------------------------------------------------------------- #
class MultiHeadAttention(Module):
    def __init__(self, dim, num_heads, causal=True, name="mha"):
        self.name = name
        self.dim = dim
        self.heads = num_heads
        self.dh = dim // num_heads
        self.causal = causal
        self.qkv = Linear(dim, 3 * dim, bias=True, name=f"{name}.qkv")
        self.proj = Linear(dim, dim, bias=True, name=f"{name}.proj")

    def params(self):
        return self.qkv.params() + self.proj.params()

    def build(self, ctx, store):
        self.qkv.build(ctx, store)
        self.proj.build(ctx, store)

    def _split(self, ctx, t, which, name):
        # t: (N, S, 3*dim); which: 0=Q,1=K,2=V
        N, S, _ = t.shape
        start = which * self.dim
        out = ctx.alloc((N, self.heads, S, self.dh), t.dtype,
                        CAT_ACT, name)
        if ctx.mode == "real":
            part = t.data[:, :, start:start + self.dim]
            out.data[:] = part.reshape(
                N, S, self.heads, self.dh
            ).transpose(0, 2, 1, 3)
        return out

    def _merge(self, ctx, t, name):
        N, H, S, Dh = t.shape
        # 层间传递的张量是激活 (要活到反向), 不是工作区
        out = ctx.alloc((N, S, H * Dh), t.dtype, CAT_ACT, name)
        if ctx.mode == "real":
            out.data[:] = t.data.transpose(0, 2, 1, 3).reshape(N, S, H * Dh)
        return out

    def forward(self, ctx, store, x):
        N, S, _ = x.shape
        qkv, qkv_saved = self.qkv.forward(ctx, store, x)
        # 拆 Q/K/V: 工作区张量, 拆完即释放合并张量
        q = self._split(ctx, qkv, 0, "attn.q")
        k = self._split(ctx, qkv, 1, "attn.k")
        v = self._split(ctx, qkv, 2, "attn.v")
        ctx.free_later(qkv)
        scale = 1.0 / (self.dh ** 0.5)
        out, att_saved, att_temps = ops.attention_forward(
            ctx, q, k, v, self.causal, scale
        )
        for t in att_temps:
            ctx.free_later(t)
        merged = self._merge(ctx, out, "attn.merged")
        ctx.free_later(out)
        proj_out, proj_saved = self.proj.forward(ctx, store, merged)
        # merged 已被 proj 保存在 proj_saved 中, 由引擎统一释放
        # 保存: qkv 的输入 x, q/k/v, attn probs, proj 输入已随 proj_saved
        saved = {
            "qkv_saved": qkv_saved,
            "q": q, "k": k, "v": v,
            "att_saved": att_saved,
            "proj_saved": proj_saved,
        }
        return proj_out, saved

    def backward(self, ctx, store, gy, saved):
        gmerged = self.proj.backward(ctx, store, gy, saved["proj_saved"])
        N, S, _ = gmerged.shape
        gout = ctx.alloc((N, self.heads, S, self.dh),
                         gmerged.dtype, CAT_WORK, "attn.gout_split")
        if ctx.mode == "real":
            gout.data[:] = gmerged.data.reshape(
                N, S, self.heads, self.dh
            ).transpose(0, 2, 1, 3)
        ctx.free_later(gmerged)
        gq, gk, gv = ops.attention_backward(
            ctx, gout, saved["att_saved"],
            saved["q"], saved["k"], saved["v"],
            1.0 / (self.dh ** 0.5),
        )
        ctx.free_later(gout)
        for t in saved["att_saved"]:
            ctx.free_later(t)
        # 合并梯度: 三份 (N,H,S,dh) -> (N,S,3*dim)
        gqkv = ctx.alloc(
            (gq.shape[0], gq.shape[2], 3 * self.dim), gq.dtype,
            CAT_WORK, "attn.gqkv",
        )
        if ctx.mode == "real":
            parts = []
            for g in (gq, gk, gv):
                # (N,H,S,dh) -> (N,S,H,dh) -> (N,S,H*dh=dim)
                parts.append(g.data.transpose(0, 2, 1, 3).reshape(
                    gqkv.shape[0], gqkv.shape[1], self.heads * self.dh
                ))
            gqkv.data[:] = np_concat(parts, axis=-1)
        ctx.free_later(gq)
        ctx.free_later(gk)
        ctx.free_later(gv)
        gx = self.qkv.backward(ctx, store, gqkv, saved["qkv_saved"])
        ctx.free_later(gqkv)
        for t in (saved["q"], saved["k"], saved["v"]):
            ctx.free_later(t)
        return gx


def np_concat(parts, axis):
    import numpy as np
    return np.concatenate(parts, axis=axis)


# ---------------------------------------------------------------------- #
# Transformer Block (pre-norm 残差)
# ---------------------------------------------------------------------- #
class TransformerBlock(Module):
    def __init__(self, dim, num_heads, ff_mult=4, causal=True,
                 name="block"):
        self.name = name
        self.ln1 = LayerNorm(dim, name=f"{name}.ln1")
        self.attn = MultiHeadAttention(dim, num_heads, causal,
                                       name=f"{name}.attn")
        self.ln2 = LayerNorm(dim, name=f"{name}.ln2")
        self.ff1 = Linear(dim, ff_mult * dim, name=f"{name}.ff1")
        self.act = GELU()
        self.ff2 = Linear(ff_mult * dim, dim, name=f"{name}.ff2")
        self._residual = None

    def params(self):
        return (self.ln1.params() + self.attn.params()
                + self.ln2.params() + self.ff1.params()
                + self.ff2.params())

    def build(self, ctx, store):
        for m in (self.ln1, self.attn, self.ln2, self.ff1, self.ff2):
            m.build(ctx, store)

    def forward(self, ctx, store, x):
        residual = x
        h, s1 = self.ln1.forward(ctx, store, x)
        a, s2 = self.attn.forward(ctx, store, h)
        # 残差相加: 新张量 (残差路径显式可见)
        add1 = ctx.alloc(x.shape, x.dtype, CAT_ACT, f"{self.name}.res1")
        if ctx.mode == "real":
            add1.data[:] = residual.data + a.data
        ctx.free_later(a)
        residual2 = add1
        h2, s3 = self.ln2.forward(ctx, store, add1)
        f1, s4 = self.ff1.forward(ctx, store, h2)
        f1a, s5 = self.act.forward(ctx, store, f1)
        # f1 被 GELU 保存在 s5 中供反向使用, 不能释放
        f2, s6 = self.ff2.forward(ctx, store, f1a)
        out = ctx.alloc(x.shape, x.dtype, CAT_ACT, f"{self.name}.res2")
        if ctx.mode == "real":
            out.data[:] = residual2.data + f2.data
        ctx.free_later(f2)
        saved = dict(residual=residual, ln1=s1, attn=s2, res1=residual2,
                     ln2=s3, ff1=s4, act=s5, ff2=s6)
        return out, saved

    def backward(self, ctx, store, gy, saved):
        # 残差2: out = res1 + f2
        gres2 = gy
        gf2 = self.ff2.backward(ctx, store, gres2, saved["ff2"])
        gf1a = self.act.backward(ctx, store, gf2, saved["act"])
        ctx.free_later(gf2)
        gf1 = self.ff1.backward(ctx, store, gf1a, saved["ff1"])
        ctx.free_later(gf1a)
        gh2 = self.ln2.backward(ctx, store, gf1, saved["ln2"])
        ctx.free_later(gf1)
        gadd1 = gh2  # 流入 res1 节点
        ga = self.attn.backward(ctx, store, gadd1, saved["attn"])
        ctx.free_later(gadd1)
        gh = self.ln1.backward(ctx, store, ga, saved["ln1"])
        ctx.free_later(ga)
        # 残差1 的另一路是输入 x
        gx = ctx.alloc(gh.shape, gh.dtype, CAT_WORK, f"{self.name}.gx")
        if ctx.mode == "real":
            gx.data[:] = gh.data + gres2.data
        ctx.free_later(gh)
        return gx
