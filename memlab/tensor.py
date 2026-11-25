"""张量与计算上下文。

同一份算子代码根据 Context.mode 返回两种张量:
- spec: SpecTensor, data 恒为 None, 只做形状推导, 不做任何数值计算;
- real: RealTensor, 持有 NumPy 数组, 真实参与计算。
两种模式调用的 allocator 完全相同, 因此事件序列逐条一致。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .allocator import BlockAllocator


class Tensor:
    __slots__ = ("shape", "dtype", "block_id", "ctx", "name", "category",
                 "alive", "_data")

    def __init__(self, shape, dtype, block_id, ctx, name, category, data):
        self.shape = tuple(int(d) for d in shape)
        self.dtype = dtype
        self.block_id = block_id
        self.ctx = ctx
        self.name = name
        self.category = category
        self.alive = True
        self._data = data

    @property
    def data(self):
        return self._data

    @property
    def nbytes(self):
        from .dtype import nbytes
        return nbytes(self.shape, self.dtype)

    def free(self):
        if not self.alive:
            raise RuntimeError(f"张量 {self.name} 已释放 (double free)")
        self.ctx.allocator.free(self.block_id)
        self.alive = False
        self._data = None

    def __repr__(self):
        return f"Tensor({self.name}, shape={self.shape}, {self.dtype})"


class Context:
    """一次单卡执行的上下文: 分配器 + 时钟 + 阶段 + FLOP 计数。"""

    def __init__(self, allocator: BlockAllocator, mode: str):
        assert mode in ("spec", "real")
        self.allocator = allocator
        self.mode = mode
        self.tick = 0
        self.phase = ""
        self.flops = 0
        # 记录每个 tick 的阶段 (峰值定位用)
        self.tick_phase = {}
        self._alloc_seq = 0

    def set_phase(self, phase: str):
        self.phase = phase

    def alloc(self, shape, dtype: str, category: str, name: str) -> Tensor:
        self.allocator.set_clock(self.tick, self.phase)
        block_id, arr = self.allocator.alloc(name, shape, dtype, category)
        self.tick_phase[self.tick] = self.phase
        self.tick += 1
        if self.mode == "spec":
            return Tensor(shape, dtype, block_id, self, name, category, None)
        return Tensor(shape, dtype, block_id, self, name, category, arr)

    def free_later(self, tensor: Tensor):
        """显式释放并推进时钟 (free 也产生一个事件 tick)。"""
        self.allocator.set_clock(self.tick, self.phase)
        tensor.free()
        self.tick_phase[self.tick] = self.phase
        self.tick += 1

    def add_flops(self, n: int):
        self.flops += int(n)

    def zeros_like(self, t: Tensor, dtype: str = None,
                   category: str = None, name: str = None) -> Tensor:
        dt = dtype or t.dtype
        out = self.alloc(
            t.shape, dt, category or t.category,
            name or (t.name + ".like"),
        )
        if self.mode == "real":
            out.data.fill(0)
        return out
