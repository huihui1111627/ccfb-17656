"""真实小训练框架: NumPy 实算 + 分块分配器真实记账。

与预估器唯一的区别是 BlockAllocator(materialize=True):
- 每次 alloc 都创建真实 NumPy 数组参与数值计算;
- 每次 free 都释放引用;
- 容量判定、块取整、分裂合并规则与预估完全相同,
  因此两边的 reserved 峰值与 OOM 判定应当严格一致。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from .allocator import BlockAllocator, OutOfMemoryError
from .common import PHASE_BUILD
from .dtype import get_precision
from .engine import TrainEngine, allocate_inputs
from .estimator import StepConfig, StepReport, _build_report
from .nn import ParamStore
from .optimizer import Optimizer
from .parallel import run_data_parallel, run_pipeline
from .tensor import Context


@dataclass
class RealRunResult:
    ok: bool
    oom: bool
    error: str
    peak_allocated: int
    peak_reserved: int
    peak_fragment: int
    wall_time_s: float
    flops: int
    events: list
    report: StepReport
    host_bytes_peak: int          # 宿主机 NumPy 实际数组字节峰值
    per_gpu_peak_reserved: List[int]
    rank_allocators: Optional[list] = None


def _host_peak(events):
    """从事件流统计宿主机真实数组的活跃字节峰值 (= allocated 峰值)。"""
    peak = 0
    for e in events:
        peak = max(peak, e.allocated)
    return peak


def run_real_step(cfg: StepConfig, capacity_bytes: int) -> RealRunResult:
    """真跑一步: real 模式, 真实 NumPy 计算。"""
    prec = get_precision(cfg.precision)
    model = cfg.model
    layers = model.build_layers()
    optim = Optimizer(cfg.optimizer)
    allocator = BlockAllocator(capacity_bytes, materialize=True)
    ctx = Context(allocator, mode="real")
    pc = cfg.parallel
    ckpt = pc.checkpoint_layers or None

    t0 = time.perf_counter()
    try:
        if pc.mode == "single":
            ctx.set_phase(PHASE_BUILD)
            store = ParamStore(ctx, prec)
            for layer in layers:
                for pname, shape in layer.params():
                    store.declare(pname, shape)
                layer.build(ctx, store)
            optim.build(ctx, store)
            inputs = allocate_inputs(ctx, model, cfg.batch_size, prec)
            engine = TrainEngine(layers, ctx, store, prec, ckpt)
            engine.train_step(inputs)
            optim.step(ctx, store)
            for t in inputs.values():
                if t.alive:
                    ctx.free_later(t)
            per_gpu = [allocator.peak_reserved]
        elif pc.mode in ("dp", "zero"):
            store, engine = run_data_parallel(
                layers, model, cfg.batch_size, prec, optim, ctx,
                world_size=pc.world_size, mode=pc.mode,
                checkpoint_layers=ckpt,
            )
            per_gpu = [allocator.peak_reserved]
            if pc.mode == "dp":
                per_gpu = per_gpu * pc.world_size
        elif pc.mode == "pipeline":
            per_gpu = []
            rank_allocators = []
            rank_ctxs = []
            for rank in range(pc.world_size):
                alloc_r = BlockAllocator(capacity_bytes,
                                         materialize=True)
                ctx_r = Context(alloc_r, mode="real")
                run_pipeline(
                    layers, model, cfg.batch_size, prec,
                    Optimizer(optim.kind), ctx_r,
                    world_size=pc.world_size,
                    microbatches=pc.pipeline_microbatches, rank=rank,
                    checkpoint_layers=ckpt,
                )
                per_gpu.append(alloc_r.peak_reserved)
                rank_allocators.append(alloc_r)
                rank_ctxs.append(ctx_r)
            # 主结果取峰值最高的 rank
            primary_idx = max(
                range(len(rank_allocators)),
                key=lambda i: rank_allocators[i].peak_reserved,
            )
            allocator = rank_allocators[primary_idx]
            ctx = rank_ctxs[primary_idx]
            rank_allocs_out = rank_allocators
        else:
            raise ValueError(pc.mode)
        rank_allocs_out = locals().get("rank_allocs_out", None)
    except OutOfMemoryError as e:
        wall = time.perf_counter() - t0
        report = _build_report(allocator, ctx, cfg, prec, ok=False,
                               oom=True, error=str(e),
                               per_gpu_peak=[capacity_bytes])
        return RealRunResult(
            ok=False, oom=True, error=str(e),
            peak_allocated=allocator.peak_allocated,
            peak_reserved=allocator.peak_reserved,
            peak_fragment=allocator.peak_fragment,
            wall_time_s=wall, flops=ctx.flops,
            events=list(allocator.events), report=report,
            host_bytes_peak=_host_peak(allocator.events),
            per_gpu_peak_reserved=[capacity_bytes],
        )

    wall = time.perf_counter() - t0
    report = _build_report(
        allocator, ctx, cfg, prec, ok=True, oom=False, error="",
        per_gpu_peak=per_gpu,
    )
    return RealRunResult(
        ok=True, oom=False, error="",
        peak_allocated=allocator.peak_allocated,
        peak_reserved=allocator.peak_reserved,
        peak_fragment=allocator.peak_fragment,
        wall_time_s=wall, flops=ctx.flops,
        events=list(allocator.events), report=report,
        host_bytes_peak=_host_peak(allocator.events),
        per_gpu_peak_reserved=per_gpu,
        rank_allocators=rank_allocs_out,
    )
