"""优化器: 状态张量分配与一步更新。

状态构成 (逐参数):
- sgd:        momentum (与参数同 dtype), 无动量时无状态;
- adam:       m (一阶矩) + v (二阶矩), dtype 取 precision.optim;
- 混合精度:   额外 fp32 master 权重, 更新发生在 master 上,
              更新后回写到低精度参数。

ZeRO 分片时, 每张卡只为自己负责的那部分参数分配梯度/状态。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .common import (
    CAT_GRAD, CAT_MASTER, CAT_OPTIM, CAT_PARAM, PHASE_OPT,
)
from .dtype import Precision
from .nn import ParamStore
from .tensor import Context, Tensor


class Optimizer:
    def __init__(self, kind: str = "adam", lr: float = 1e-3,
                 betas=(0.9, 0.999), momentum: float = 0.9):
        assert kind in ("adam", "sgd")
        self.kind = kind
        self.lr = lr
        self.betas = betas
        self.momentum = momentum
        # 状态名 -> 张量
        self.states: Dict[str, List[Tensor]] = {}
        self.masters: Dict[str, Tensor] = {}
        self.state_bytes = 0
        self.master_bytes = 0

    # ------------------------------------------------------------------ #
    def state_specs(self, param_shape, prec: Precision):
        """返回某参数对应的状态张量描述 [(suffix, dtype, category)]。"""
        specs = []
        if self.kind == "adam":
            specs.append(("m", prec.optim, CAT_OPTIM))
            specs.append(("v", prec.optim, CAT_OPTIM))
        elif self.kind == "sgd" and self.momentum > 0:
            specs.append(("mom", prec.param, CAT_OPTIM))
        if prec.master is not None:
            specs.append(("master", prec.master, CAT_MASTER))
        return specs

    def build(self, ctx: Context, store: ParamStore,
              owned_params: Optional[Dict[str, tuple]] = None):
        """为 (本卡拥有的) 参数分配梯度、优化器状态与主权重。

        owned_params: {param_name: shape_on_this_rank};
                      None 表示全部参数整体在本卡。
        """
        ctx.set_phase(PHASE_OPT)
        if owned_params is None:
            owned_params = dict(store.shapes)
        for name, shape in owned_params.items():
            if name not in store.shapes:
                continue
            full_shape = store.shapes[name]
            # 梯度槽 (分片形状)
            store.alloc_grad(name, shape)
            states = []
            for suffix, dt, cat in self.state_specs(full_shape, store.prec):
                t = ctx.alloc(shape, dt, cat, f"{suffix}:{name}")
                if ctx.mode == "real" and suffix in ("m", "v", "mom"):
                    t.data.fill(0)
                if cat == CAT_MASTER:
                    self.masters[name] = t
                else:
                    states.append(t)
                if cat == CAT_MASTER:
                    self.master_bytes += t.nbytes
                else:
                    self.state_bytes += t.nbytes
            self.states[name] = states

    # ------------------------------------------------------------------ #
    def step(self, ctx: Context, store: ParamStore,
             owned: Optional[set] = None):
        """执行一步参数更新 (real 模式做真实数值更新)。"""
        ctx.set_phase(PHASE_OPT)
        names = sorted(owned if owned is not None else store.shapes.keys())
        for name in names:
            if name not in store.params:
                continue
            p = store.params[name]
            g = store.grads[name]
            if self.kind == "adam":
                m, v = self.states[name][0], self.states[name][1]
                if ctx.mode == "real":
                    gd = g.data.astype(np.float32)
                    m.data[:] = (self.betas[0] * m.data
                                 + (1 - self.betas[0]) * gd)
                    v.data[:] = (self.betas[1] * v.data
                                 + (1 - self.betas[1]) * gd * gd)
                    mh = m.data / (1 - self.betas[0])
                    vh = v.data / (1 - self.betas[1])
                    update = self.lr * mh / (np.sqrt(vh) + 1e-8)
                    if name in self.masters:
                        master = self.masters[name]
                        master.data[:] -= update.astype(master.data.dtype)
                        p.data[:] = master.data.astype(p.data.dtype)
                    else:
                        p.data[:] -= update.astype(p.data.dtype)
                    g.data.fill(0)
            else:  # sgd
                if self.states[name]:
                    mom = self.states[name][0]
                    if ctx.mode == "real":
                        mom.data[:] = (self.momentum * mom.data
                                       + g.data.astype(mom.data.dtype))
                        p.data[:] -= (self.lr * mom.data).astype(p.data.dtype)
                        g.data.fill(0)
                elif ctx.mode == "real":
                    p.data[:] -= (self.lr * g.data).astype(p.data.dtype)
                    g.data.fill(0)
