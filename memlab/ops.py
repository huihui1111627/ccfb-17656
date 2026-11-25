"""函数式算子: 每个算子负责自己的工作区分配/释放与反向保存。

约定:
- 所有输出张量经 ctx.alloc 分配 (spec/real 两条路径事件序列一致);
- 前向返回 (out, saved), saved 中的张量在反向结束前必须由层代码释放;
- 反向返回 (grad_out_input, grad_weight...), 中间临时张量在函数内释放;
- real 模式用 NumPy 真算, spec 模式只推形状、跳过数值计算。
"""
from __future__ import annotations

import numpy as np

from .common import CAT_ACT, CAT_WORK


# ---------------------------------------------------------------------- #
# 小工具
# ---------------------------------------------------------------------- #
def _rng(seed):
    return np.random.default_rng(seed)


def stable_seed(name: str, salt: int = 0) -> int:
    """进程稳定的字符串 -> 非负整数种子 (不能用内置 hash, 它带随机化)。"""
    import zlib
    return (zlib.crc32(name.encode("utf-8")) + salt) % (2 ** 32)


def init_weight(ctx, shape, dtype, category, name, seed=0, gain=0.1):
    """参数/张量初始化 (real 模式填随机数, spec 模式仅占位)。

    参数顺序与 ctx.alloc 一致: (shape, dtype, category, name)。
    """
    t = ctx.alloc(shape, dtype, category, name)
    if ctx.mode == "real":
        rng = np.random.default_rng(abs(int(seed)) % (2 ** 32))
        t.data[:] = (rng.standard_normal(shape) * gain).astype(t.data.dtype)
    return t


def add_flops_linear(ctx, m, k, n):
    # Y = X @ W^T: m*k*n 乘加
    ctx.add_flops(2 * m * k * n)


# ---------------------------------------------------------------------- #
# Linear
# ---------------------------------------------------------------------- #
def linear_forward(ctx, x, w, b=None):
    # 支持任意前导维: (..., K) @ (N, K)^T -> (..., N)
    x_shape = x.shape
    k = x_shape[-1]
    n = w.shape[0]
    lead = x_shape[:-1]
    m = int(np.prod(lead)) if lead else 1
    out_shape = lead + (n,)
    out = ctx.alloc(out_shape, x.dtype, CAT_ACT, f"linear({w.name}).out")
    if ctx.mode == "real":
        x2 = x.data.reshape(m, k)
        y = x2 @ w.data.T
        if b is not None:
            y += b.data
        out.data[:] = y.reshape(out_shape)
    add_flops_linear(ctx, m, k, n)
    # 只保存输入激活; 权重反向时从 store 重新取 (兼容 ZeRO 分片)
    return out, [x]


def linear_backward(ctx, gy, saved, w, has_bias):
    (x,) = saved
    x_shape = x.shape
    k = x_shape[-1]
    n = w.shape[0]
    lead = x_shape[:-1]
    m = int(np.prod(lead)) if lead else 1
    gx = ctx.alloc(x_shape, gy.dtype, CAT_WORK, "linear.gx")
    gw = ctx.alloc(w.shape, gy.dtype, CAT_WORK, "linear.gw.work")
    if ctx.mode == "real":
        gy2 = gy.data.reshape(m, n)
        gx.data[:] = (gy2 @ w.data).reshape(x_shape)
        gw.data[:] = gy2.T @ x.data.reshape(m, k)
    add_flops_linear(ctx, m, n, k)
    add_flops_linear(ctx, n, m, k)
    if has_bias:
        gb = ctx.alloc((n,), gy.dtype, CAT_WORK, "linear.gb.work")
        if ctx.mode == "real":
            gb.data[:] = gy.data.reshape(m, n).sum(axis=0)
    else:
        gb = None
    return gx, gw, gb


# ---------------------------------------------------------------------- #
# 激活
# ---------------------------------------------------------------------- #
def relu_forward(ctx, x):
    out = ctx.alloc(x.shape, x.dtype, CAT_ACT, "relu.out")
    if ctx.mode == "real":
        out.data[:] = np.maximum(x.data, 0)
    # 反向只需 mask; 为显式起见单独存一个 bool mask (1 byte/elem)
    mask = ctx.alloc(x.shape, "bool", CAT_ACT, "relu.mask")
    if ctx.mode == "real":
        mask.data[:] = x.data > 0
    ctx.add_flops(x.nbytes)
    return out, [mask]


def relu_backward(ctx, gy, saved):
    (mask,) = saved
    gx = ctx.alloc(gy.shape, gy.dtype, CAT_WORK, "relu.gx")
    if ctx.mode == "real":
        gx.data[:] = gy.data * mask.data
    return gx


def gelu_forward(ctx, x):
    # tanh 近似 GELU, 保存输入即可
    out = ctx.alloc(x.shape, x.dtype, CAT_ACT, "gelu.out")
    if ctx.mode == "real":
        xd = x.data.astype(np.float32)
        out.data[:] = (
            0.5 * xd * (1 + np.tanh(
                np.sqrt(2 / np.pi) * (xd + 0.044715 * xd ** 3)
            ))
        ).astype(out.data.dtype)
    ctx.add_flops(x.nbytes * 8)
    return out, [x]


def gelu_backward(ctx, gy, saved):
    (x,) = saved
    gx = ctx.alloc(gy.shape, gy.dtype, CAT_WORK, "gelu.gx")
    if ctx.mode == "real":
        xd = x.data.astype(np.float32)
        inner = np.sqrt(2 / np.pi) * (xd + 0.044715 * xd ** 3)
        tanh = np.tanh(inner)
        deriv = 0.5 * (1 + tanh) + 0.5 * xd * (
            1 - tanh ** 2
        ) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * xd ** 2)
        gx.data[:] = (gy.data.astype(np.float32) * deriv).astype(gx.data.dtype)
    return gx


# ---------------------------------------------------------------------- #
# Conv2d (显式 im2col, 工作区可见、可统计)
# ---------------------------------------------------------------------- #
def conv2d_forward(ctx, x, w, b, stride, padding):
    N, C, H, Wd = x.shape
    K, _, kh, kw = w.shape
    Ho = (H + 2 * padding - kh) // stride + 1
    Wo = (Wd + 2 * padding - kw) // stride + 1

    cols = ctx.alloc((N, C * kh * kw, Ho * Wo), x.dtype, CAT_WORK,
                     "conv.im2col")
    if ctx.mode == "real":
        xp = np.pad(
            x.data,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        )
        n, ckk, l = cols.shape
        cols_np = np.empty((n, ckk, l), dtype=cols.data.dtype)
        for i in range(kh):
            i_s = i * stride
            for j in range(kw):
                j_s = j * stride
                patch = xp[:, :, i_s:i_s + Ho, j_s:j_s + Wo]
                cols_np[:, (i * kw + j) * C:(i * kw + j + 1) * C, :] = \
                    patch.reshape(N, C, Ho * Wo)
        cols.data[:] = cols_np

    out = ctx.alloc((N, K, Ho, Wo), x.dtype, CAT_ACT, "conv.out")
    if ctx.mode == "real":
        wcol = w.data.reshape(K, -1)
        out.data[:] = (
            (wcol @ cols.data).reshape(N, K, Ho, Wo)
            + (b.data.reshape(1, K, 1, 1) if b is not None else 0)
        )
    add_flops_linear(ctx, N * Ho * Wo, C * kh * kw, K)
    return out, [x, cols]


def conv2d_backward(ctx, gy, saved, w, w_shape, stride, padding,
                    has_bias):
    x, cols = saved
    N, C, H, Wd = x.shape
    K, _, kh, kw = w_shape
    _, _, Ho, Wo = gy.shape

    gw = ctx.alloc(w_shape, gy.dtype, CAT_WORK, "conv.gw.work")
    if ctx.mode == "real":
        wcol = (gy.data.reshape(N, K, -1) @
                cols.data.transpose(0, 2, 1)).sum(axis=0)
        gw.data[:] = wcol.reshape(w_shape)
    gb = None
    if has_bias:
        gb = ctx.alloc((K,), gy.dtype, CAT_WORK, "conv.gb.work")
        if ctx.mode == "real":
            gb.data[:] = gy.data.sum(axis=(0, 2, 3))

    gcols = ctx.alloc(cols.shape, gy.dtype, CAT_WORK, "conv.gcols")
    if ctx.mode == "real":
        # 输入梯度必须用前向 *权重*, 不是刚算出的梯度 gw
        wcol = w.data.reshape(K, -1)
        gcols.data[:] = wcol.T @ gy.data.reshape(N, K, -1)

    gx = ctx.alloc(x.shape, gy.dtype, CAT_WORK, "conv.gx")
    if ctx.mode == "real":
        gxp = np.zeros(
            (N, C, H + 2 * padding, Wd + 2 * padding), dtype=gx.data.dtype
        )
        for i in range(kh):
            i_s = i * stride
            for j in range(kw):
                j_s = j * stride
                part = gcols.data[
                    :, (i * kw + j) * C:(i * kw + j + 1) * C, :
                ].reshape(N, C, Ho, Wo)
                gxp[:, :, i_s:i_s + Ho, j_s:j_s + Wo] += part
        if padding:
            gx.data[:] = gxp[:, :, padding:-padding, padding:-padding]
        else:
            gx.data[:] = gxp
    ctx.add_flops(2 * N * K * Ho * Wo * C * kh * kw * 2)
    # 工作区临时张量即时回收
    ctx.free_later(gcols)
    ctx.free_later(cols)
    return gx, gw, gb


def maxpool2d_forward(ctx, x, kernel, stride):
    N, C, H, Wd = x.shape
    Ho = (H - kernel) // stride + 1
    Wo = (Wd - kernel) // stride + 1
    out = ctx.alloc((N, C, Ho, Wo), x.dtype, CAT_ACT, "maxpool.out")
    idx = ctx.alloc((N, C, Ho, Wo), "int32", CAT_ACT, "maxpool.idx")
    if ctx.mode == "real":
        windows = np.lib.stride_tricks.sliding_window_view(
            x.data, (kernel, kernel), axis=(2, 3)
        )[:, :, ::stride, ::stride]
        flat = windows.reshape(N, C, Ho, Wo, -1)
        m = flat.argmax(axis=-1)
        idx.data[:] = m
        out.data[:] = np.take_along_axis(flat, m[..., None], -1)[..., 0]
    return out, [idx, (kernel, stride, x.shape)]


def maxpool2d_backward(ctx, gy, saved):
    idx, (kernel, stride, x_shape) = saved
    gx = ctx.alloc(x_shape, gy.dtype, CAT_WORK, "maxpool.gx")
    if ctx.mode == "real":
        gx.data.fill(0)
        N, C, Ho, Wo = gy.shape
        for n in range(N):
            for c in range(C):
                for i in range(Ho):
                    for j in range(Wo):
                        k = idx.data[n, c, i, j]
                        di, dj = divmod(k, kernel)
                        gx.data[n, c, i * stride + di, j * stride + dj] += \
                            gy.data[n, c, i, j]
    return gx


# ---------------------------------------------------------------------- #
# LayerNorm
# ---------------------------------------------------------------------- #
def layernorm_forward(ctx, x, gamma, beta, eps=1e-5):
    mean = ctx.alloc(x.shape[:-1] + (1,), x.dtype, CAT_WORK, "ln.mean")
    var = ctx.alloc(x.shape[:-1] + (1,), x.dtype, CAT_WORK, "ln.var")
    xn = ctx.alloc(x.shape, x.dtype, CAT_ACT, "ln.xnorm")
    out = ctx.alloc(x.shape, x.dtype, CAT_ACT, "ln.out")
    if ctx.mode == "real":
        xd = x.data.astype(np.float32)
        m = xd.mean(axis=-1, keepdims=True)
        v = xd.var(axis=-1, keepdims=True)
        mean.data[:] = m.astype(mean.data.dtype)
        var.data[:] = v.astype(var.data.dtype)
        norm = (xd - m) / np.sqrt(v + eps)
        xn.data[:] = norm.astype(xn.data.dtype)
        out.data[:] = (norm * gamma.data + beta.data).astype(out.data.dtype)
    # mean/var 是临时工作区, 前向结束即可释放 (xn 已保存);
    # gamma/beta 反向时从 store 重新取 (兼容 ZeRO 分片)
    return out, [xn], [mean, var]


def layernorm_backward(ctx, gy, saved, gamma, x_shape):
    (xn,) = saved
    D = x_shape[-1]
    gx = ctx.alloc(x_shape, gy.dtype, CAT_WORK, "ln.gx")
    gg = ctx.alloc((D,), gy.dtype, CAT_WORK, "ln.ggamma.work")
    gb = ctx.alloc((D,), gy.dtype, CAT_WORK, "ln.gbeta.work")
    if ctx.mode == "real":
        gyd = gy.data.astype(np.float32)
        xnd = xn.data.astype(np.float32)
        gb.data[:] = gyd.sum(axis=tuple(range(gyd.ndim - 1)))
        gg.data[:] = (
            gyd * xnd
        ).sum(axis=tuple(range(gyd.ndim - 1)))
        gnorm = gyd * gamma.data
        gx.data[:] = (
            D * gnorm - gnorm.sum(axis=-1, keepdims=True)
            - xnd * (gnorm * xnd).sum(axis=-1, keepdims=True)
        ).astype(gx.data.dtype) / D
    return gx, gg, gb


# ---------------------------------------------------------------------- #
# Embedding
# ---------------------------------------------------------------------- #
def embedding_forward(ctx, tokens, table, vocab, seq):
    N = tokens.shape[0]
    out = ctx.alloc((N, seq, table.shape[1]), table.dtype, CAT_ACT,
                    "embed.out")
    if ctx.mode == "real":
        out.data[:] = table.data[tokens.data]
    return out, [tokens]


def embedding_backward(ctx, gy, tokens, table_shape):
    g = ctx.alloc(table_shape, gy.dtype, CAT_WORK, "embed.gtable.work")
    if ctx.mode == "real":
        g.data.fill(0)
        np.add.at(g.data, tokens.data.reshape(-1),
                  gy.data.reshape(-1, table_shape[1]))
    return g


# ---------------------------------------------------------------------- #
# Softmax 交叉熵 (融合, 工作区最小)
# ---------------------------------------------------------------------- #
def softmax_ce_forward(ctx, logits, targets):
    """logits: (..., V), targets: (M,) int32, M=前导维乘积。返回 loss 与 saved。"""
    V = logits.shape[-1]
    lead = logits.shape[:-1]
    M = int(np.prod(lead)) if lead else 1
    flat_shape = (M, V)
    probs = ctx.alloc(flat_shape, logits.dtype, CAT_WORK, "ce.probs")
    loss = ctx.alloc((1,), "fp32", CAT_WORK, "ce.loss")
    if ctx.mode == "real":
        z = logits.data.reshape(M, V).astype(np.float32)
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=-1, keepdims=True)
        probs.data[:] = p.astype(probs.data.dtype)
        loss.data[:] = -np.log(
            p[np.arange(M), targets.data] + 1e-9
        ).mean(keepdims=True)
    return loss, probs, [probs, targets, logits.shape]


def softmax_ce_backward(ctx, probs, targets, M, logits_shape=None):
    glogits = ctx.alloc(probs.shape, probs.dtype, CAT_WORK, "ce.glogits")
    if ctx.mode == "real":
        g = probs.data.astype(np.float32).copy()
        g[np.arange(M), targets.data] -= 1
        glogits.data[:] = (g / M).astype(glogits.data.dtype)
    if logits_shape is not None and tuple(logits_shape) != probs.shape:
        reshaped = ctx.alloc(logits_shape, glogits.dtype, CAT_WORK,
                             "ce.glogits.reshape")
        if ctx.mode == "real":
            reshaped.data[:] = glogits.data.reshape(logits_shape)
        ctx.free_later(glogits)
        return reshaped
    return glogits


def reshape_forward(ctx, x, new_shape):
    out = ctx.alloc(new_shape, x.dtype, CAT_ACT, "reshape.out")
    if ctx.mode == "real":
        out.data[:] = x.data.reshape(new_shape)
    return out, [x.shape]


# ---------------------------------------------------------------------- #
# 缩放点积注意力 (显式 QK^T 与 softmax, 工作区可见)
# ---------------------------------------------------------------------- #
def attention_forward(ctx, q, k, v, mask_causal, scale):
    # q,k,v: (N, H, S, Dh)
    N, H, S, Dh = q.shape
    scores = ctx.alloc((N, H, S, S), q.dtype, CAT_WORK, "attn.scores")
    if ctx.mode == "real":
        scores.data[:] = (
            q.data @ k.data.transpose(0, 1, 3, 2)
        ) * scale
        if mask_causal:
            causal = np.triu(np.ones((S, S), dtype=bool), k=1)
            scores.data[:, :, causal] = -1e4
    probs = ctx.alloc((N, H, S, S), q.dtype, CAT_ACT, "attn.probs")
    if ctx.mode == "real":
        sd = scores.data.astype(np.float32)
        e = np.exp(sd - sd.max(axis=-1, keepdims=True))
        probs.data[:] = (e / e.sum(axis=-1, keepdims=True)).astype(
            probs.data.dtype
        )
    out = ctx.alloc((N, H, S, Dh), q.dtype, CAT_ACT, "attn.out")
    if ctx.mode == "real":
        out.data[:] = probs.data @ v.data
    ctx.add_flops(4 * N * H * S * S * Dh)
    return out, [probs], [scores]


def attention_backward(ctx, gout, saved, q, k, v, scale):
    (probs,) = saved
    gv = ctx.alloc(v.shape, gout.dtype, CAT_WORK, "attn.gv")
    gp = ctx.alloc(probs.shape, gout.dtype, CAT_WORK, "attn.gprobs")
    gscores = ctx.alloc(probs.shape, gout.dtype, CAT_WORK,
                        "attn.gscores")
    gq = ctx.alloc(q.shape, gout.dtype, CAT_WORK, "attn.gq")
    gk = ctx.alloc(k.shape, gout.dtype, CAT_WORK, "attn.gk")
    if ctx.mode == "real":
        gv.data[:] = probs.data.transpose(0, 1, 3, 2) @ gout.data
        gp.data[:] = gout.data @ v.data.transpose(0, 1, 3, 2)
        pd = probs.data.astype(np.float32)
        gpd = gp.data.astype(np.float32)
        gs = pd * (
            gpd - (gpd * pd).sum(axis=-1, keepdims=True)
        ) * scale
        gscores.data[:] = gs.astype(gscores.data.dtype)
        gq.data[:] = gscores.data @ k.data
        gk.data[:] = gscores.data.transpose(0, 1, 3, 2) @ q.data
    ctx.add_flops(8 * q.shape[0] * q.shape[1] * q.shape[2] ** 2 * q.shape[3])
    # 注意力反向中间工作区即时回收 (gq/gk/gv 由调用方释放)
    ctx.free_later(gp)
    ctx.free_later(gscores)
    return gq, gk, gv
