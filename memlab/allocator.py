"""分块显存分配器 (预估与真跑共用)。

模拟 PyTorch caching allocator 的行为:
- 小分配 (<= SMALL_THRESHOLD) 向上取整到 SMALL_ROUND 的倍数;
- 大分配向上取整到 LARGE_ROUND 的倍数;
- 按块 (block) 预留显存, 大块可分裂, 释放后相邻空闲块合并;
- 同尺寸优先复用空闲块, 因此 *碎片* 会随分配/释放交错自然产生。

分配器只记账: 真跑模式下 RealTensor 的 NumPy 数组按"请求字节"
(而非取整后的块字节) 实际创建, 宿主机内存压力约等于活跃字节;
而 reserved/容量判定始终按取整块字节走, 与设备上的真实行为一致。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .dtype import ELEM_BYTES

SMALL_THRESHOLD = 1 * 1024 * 1024      # 1 MiB
SMALL_ROUND = 512
LARGE_ROUND = 2 * 1024 * 1024          # 2 MiB
MIN_SPLIT = 512


class OutOfMemoryError(RuntimeError):
    """模拟设备显存不足。"""

    def __init__(self, requested, reserved, capacity):
        super().__init__(
            f"OOM: 申请 {requested:,} B, 已预留 {reserved:,} B, "
            f"设备容量 {capacity:,} B"
        )
        self.requested = requested
        self.reserved = reserved
        self.capacity = capacity


def round_size(num_bytes: int) -> int:
    """分配器块取整规则。"""
    if num_bytes <= SMALL_THRESHOLD:
        return ((num_bytes + SMALL_ROUND - 1) // SMALL_ROUND) * SMALL_ROUND
    return ((num_bytes + LARGE_ROUND - 1) // LARGE_ROUND) * LARGE_ROUND


@dataclass
class Event:
    """一次 alloc/free 的不可变记录。"""

    tick: int
    kind: str                 # "alloc" | "free"
    name: str
    category: str
    phase: str
    req_bytes: int            # 请求字节 (真实张量大小)
    block_bytes: int          # 实际占用块字节 (取整后)
    allocated: int            # 事件后活跃请求字节
    reserved: int             # 事件后预留块字节
    block_id: int
    reused: bool = False      # 是否命中空闲块复用


@dataclass
class Block:
    bid: int
    start: int
    size: int
    free: bool = True
    name: str = ""
    category: str = ""
    req_bytes: int = 0
    np_arr: Optional[np.ndarray] = None


class BlockAllocator:
    """单卡分块分配器。materialize=True 时创建真实 NumPy 数组。"""

    def __init__(self, capacity_bytes: int, materialize: bool = False):
        self.capacity = int(capacity_bytes)
        self.materialize = materialize
        self.blocks: Dict[int, Block] = {}
        self._free_ids: List[int] = []
        self._next_id = 0
        self._top = 0                      # 已预留区间顶端 (模拟地址)
        self.allocated = 0                 # 活跃请求字节
        self.reserved = 0                  # 预留块字节 (含碎片)
        self.peak_allocated = 0
        self.peak_reserved = 0
        # 内部碎片: 活跃块取整浪费 (block_size - req) 之和
        self.internal_fragment = 0
        self.peak_internal_fragment = 0
        # 峰值 reserved 那一刻的内/外碎片快照 (同一时刻, 可加)
        self.peak_external_fragment = 0
        self.events: List[Event] = []
        self.current_tick = 0
        self.current_phase = ""
        # 碎片统计: 每次事件后 reserved-allocated 的峰值
        self.peak_fragment = 0

    # ------------------------------------------------------------------ #
    def reset(self):
        self.blocks.clear()
        self._free_ids.clear()
        self._next_id = 0
        self._top = 0
        self.allocated = 0
        self.reserved = 0
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.peak_fragment = 0
        self.internal_fragment = 0
        self.peak_internal_fragment = 0
        self.events.clear()

    def set_clock(self, tick: int, phase: str):
        self.current_tick = tick
        self.current_phase = phase

    def external_fragment(self) -> int:
        """外部碎片: 当前所有空闲块字节之和 (已预留但无人使用)。"""
        return sum(self.blocks[b].size for b in self._free_ids)

    def _update_peaks(self):
        self.peak_allocated = max(self.peak_allocated, self.allocated)
        if self.reserved > self.peak_reserved:
            self.peak_reserved = self.reserved
            # 在 reserved 创新高的同一时刻记录, 三者口径一致、可加:
            # reserved = allocated + internal + external
            self.peak_internal_fragment = self.internal_fragment
            self.peak_external_fragment = self.external_fragment()
            # 峰值时刻的总损耗 (与内/外碎片同一时刻)
            self.peak_fragment = (
                self.internal_fragment + self.external_fragment()
            )

    # ------------------------------------------------------------------ #
    def _find_free_block(self, need: int) -> Optional[Block]:
        """最小适配: 在空闲块里找够大的最小块。"""
        best = None
        for bid in self._free_ids:
            b = self.blocks[bid]
            if b.size >= need and (best is None or b.size < best.size):
                best = b
        return best

    def _coalesce(self, block: Block):
        """与地址相邻的空闲块合并。"""
        merged = True
        while merged:
            merged = False
            for bid in list(self._free_ids):
                other = self.blocks[bid]
                if other is block:
                    continue
                if other.start + other.size == block.start:
                    self._free_ids.remove(other.bid)
                    block.start = other.start
                    block.size += other.size
                    del self.blocks[other.bid]
                    merged = True
                elif block.start + block.size == other.start:
                    self._free_ids.remove(other.bid)
                    block.size += other.size
                    del self.blocks[other.bid]
                    merged = True

    def alloc(self, name: str, shape, dtype: str, category: str):
        """分配一个张量, 返回 (block_id, ndarray 或 None)。"""
        req = int(ELEM_BYTES[dtype])
        for d in shape:
            req *= int(d)
        need = round_size(req)

        block = self._find_free_block(need)
        reused = block is not None
        if block is not None:
            # 分裂: 剩余部分够大就切成新空闲块
            if block.size - need >= MIN_SPLIT:
                tail = Block(
                    bid=self._next_id, start=block.start + need,
                    size=block.size - need, free=True,
                )
                self._next_id += 1
                self.blocks[tail.bid] = tail
                self._free_ids.append(tail.bid)
                block.size = need
            self._free_ids.remove(block.bid)
            block.free = False
        else:
            if self.reserved + need > self.capacity:
                raise OutOfMemoryError(req, self.reserved, self.capacity)
            block = Block(
                bid=self._next_id, start=self._top, size=need, free=False,
            )
            self._next_id += 1
            self._top += need
            self.blocks[block.bid] = block
            self.reserved += need

        block.name = name
        block.category = category
        block.req_bytes = req

        arr = None
        if self.materialize:
            np_dtype = _numpy_dtype(dtype)
            # 宿主机对照: 按请求形状创建真实数组参与数值计算
            arr = np.empty(tuple(int(d) for d in shape), dtype=np_dtype)
        block.np_arr = arr

        self.allocated += req
        # 内部碎片 = 活跃块取整后多占的部分
        self.internal_fragment += block.size - req
        self._update_peaks()
        self.events.append(Event(
            tick=self.current_tick, kind="alloc", name=name,
            category=category, phase=self.current_phase,
            req_bytes=req, block_bytes=block.size,
            allocated=self.allocated, reserved=self.reserved,
            block_id=block.bid, reused=reused,
        ))
        return block.bid, arr

    def free(self, block_id: int):
        block = self.blocks[block_id]
        if block.free:
            raise RuntimeError(f"块 {block_id} 已被释放 (double free)")
        req = block.req_bytes
        self.allocated -= req
        self.internal_fragment -= block.size - req
        block.free = True
        block.name = "free"
        block.category = ""
        block.req_bytes = 0
        block.np_arr = None
        self._free_ids.append(block.bid)
        self._coalesce(block)
        self._update_peaks()
        self.events.append(Event(
            tick=self.current_tick, kind="free", name=block.name,
            category="", phase=self.current_phase,
            req_bytes=req, block_bytes=block.size,
            allocated=self.allocated, reserved=self.reserved,
            block_id=block_id,
        ))

    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict:
        return {
            "tick": self.current_tick,
            "phase": self.current_phase,
            "allocated": self.allocated,
            "reserved": self.reserved,
            "internal_fragment": self.internal_fragment,
            "external_fragment": self.external_fragment(),
            "fragment": self.reserved - self.allocated,
            "peak_allocated": self.peak_allocated,
            "peak_reserved": self.peak_reserved,
            "peak_fragment": self.peak_fragment,
            "num_blocks": len(self.blocks),
            "num_free_blocks": len(self._free_ids),
        }


def _numpy_dtype(dtype: str):
    """模拟 dtype -> 宿主机 NumPy dtype。

    宿主机只做小规模对照计算: bf16 无 NumPy 原生支持, 全部浮点
    统一用 float32 计算 (字节记账仍按模拟 dtype 走)。
    """
    mapping = {
        "fp32": np.float32,
        "fp16": np.float32,   # 见模块 docstring
        "bf16": np.float32,
        "fp64": np.float64,
        "int8": np.int8,
        "uint8": np.uint8,
        "int32": np.int32,
        "int64": np.int64,
        "bool": np.bool_,
    }
    return mapping[dtype]
