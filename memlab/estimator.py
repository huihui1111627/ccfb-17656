"""显存预估器: 不做数值计算, 推演一步训练的完整显存时间线。

输出 StepReport:
- 逐 tick 的 allocated/reserved/碎片;
- 逐 tick、逐类别 (参数/梯度/优化器/激活/工作区/...) 的字节数;
- 峰值出现的 tick、阶段与当时占用最大的张量列表;
- 静态显存 (参数+梯度+状态+主权重) 与动态峰值;
- 能否在给定容量上跑 (按 reserved 判定, 已含碎片)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .allocator import BlockAllocator, OutOfMemoryError
from .common import (
    CAT_GRAD, CAT_MASTER, CAT_OPTIM, CAT_PARAM, CATEGORIES,
    PHASE_BWD, PHASE_FWD, PHASE_OPT,
)
from .dtype import Precision, get_precision, nbytes
from .engine import TrainEngine, allocate_inputs
from .models import ModelSpec
from .nn import ParamStore
from .optimizer import Optimizer
from .parallel import (
    ParallelConfig, run_data_parallel, run_pipeline,
)
from .tensor import Context


@dataclass
class StepConfig:
    model: ModelSpec
    batch_size: int
    precision: str = "fp32"
    optimizer: str = "adam"
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    gpu_memory_bytes: Optional[int] = None
    seq_len_override: Optional[int] = None


@dataclass
class TickPoint:
    tick: int
    phase: str
    allocated: int
    reserved: int
    fragmentation: int
    by_category: Dict[str, int]


@dataclass
class StepReport:
    ok: bool
    oom: bool
    error: str
    ticks: List[TickPoint]
    peak_allocated: int
    peak_reserved: int
    peak_fragment: int
    peak_tick: int
    peak_phase: str
    peak_top_tensors: List[dict]
    static_bytes: int
    persistent_by_category: Dict[str, int]
    total_events: int
    alloc_events: int
    reuse_hits: int
    flops: int
    config: dict
    per_gpu_peak_reserved: List[int] = field(default_factory=list)
    extra_flops_checkpoint: int = 0
    peak_internal_fragment: int = 0
    peak_external_fragment: int = 0

    @property
    def fragmentation_ratio(self):
        return (self.peak_fragment / self.peak_reserved
                if self.peak_reserved else 0.0)


def _make_spec(spec: ModelSpec, seq_override):
    if seq_override is None or seq_override == spec.seq_len:
        return spec
    from .models import lm_input
    import dataclasses
    return dataclasses.replace(
        spec,
        seq_len=seq_override,
        input_shape_fn=lambda b: lm_input(b, seq_override, spec.vocab),
    )


def estimate_step(cfg: StepConfig, capacity_bytes: int) -> StepReport:
    """纯预估: spec 模式跑一步, 从事件流重建时间线。"""
    prec = get_precision(cfg.precision)
    model = _make_spec(cfg.model, cfg.seq_len_override)
    layers = model.build_layers()
    optim = Optimizer(cfg.optimizer)
    allocator = BlockAllocator(capacity_bytes, materialize=False)
    ctx = Context(allocator, mode="spec")
    pc = cfg.parallel

    ckpt = pc.checkpoint_layers or None
    try:
        if pc.mode in ("single", "dp", "zero"):
            world = pc.world_size if pc.mode != "single" else 1
            if pc.mode == "single":
                store = _run_single(layers, model, cfg.batch_size,
                                    prec, optim, ctx, ckpt)
                per_gpu = [allocator.peak_reserved]
            else:
                store, engine = run_data_parallel(
                    layers, model, cfg.batch_size, prec, optim, ctx,
                    world_size=world, mode=pc.mode,
                    checkpoint_layers=ckpt,
                )
                per_gpu = [allocator.peak_reserved]
                if pc.mode == "dp":
                    # 数据并行各卡同构: 每卡峰值相同
                    per_gpu = [allocator.peak_reserved] * world
        elif pc.mode == "pipeline":
            # 逐 rank 推演, 取各卡峰值; 主 allocator 换成峰值最高的 rank
            per_gpu, rank_allocs, rank_ctxs = \
                _estimate_pipeline_per_rank(
                    layers, model, cfg.batch_size, prec, optim, pc,
                    capacity_bytes,
                )
            primary = max(range(len(per_gpu)), key=lambda i: per_gpu[i])
            allocator = rank_allocs[primary]
            ctx = rank_ctxs[primary]
        else:
            raise ValueError(f"未知并行模式: {pc.mode}")
    except OutOfMemoryError as e:
        return _build_report(allocator, ctx, cfg, prec, ok=False,
                             oom=True, error=str(e),
                             per_gpu_peak=[capacity_bytes])

    report = _build_report(
        allocator, ctx, cfg, prec, ok=True, oom=False, error="",
        per_gpu_peak=per_gpu if pc.mode == "pipeline"
        else ([allocator.peak_reserved]
              if pc.mode == "single" else per_gpu),
    )
    return report


def _run_single(layers, model, batch, prec, optim, ctx, ckpt):
    from .common import PHASE_BUILD
    ctx.set_phase(PHASE_BUILD)
    store = ParamStore(ctx, prec)
    for layer in layers:
        for pname, shape in layer.params():
            store.declare(pname, shape)
        layer.build(ctx, store)
    optim.build(ctx, store)
    inputs = allocate_inputs(ctx, model, batch, prec)
    engine = TrainEngine(layers, ctx, store, prec, ckpt)
    engine.train_step(inputs)
    optim.step(ctx, store)
    for t in inputs.values():
        if t.alive:
            ctx.free_later(t)
    return store


def _estimate_pipeline_per_rank(layers, model, batch, prec, optim,
                                pc, capacity):
    peaks = []
    allocs = []
    ctxs = []
    stages = pc.world_size
    mb = pc.pipeline_microbatches
    for rank in range(stages):
        allocator = BlockAllocator(capacity, materialize=False)
        ctx = Context(allocator, mode="spec")
        run_pipeline(
            layers, model, batch, prec,
            Optimizer(pc_mode_optim(optim)), ctx,
            world_size=stages, microbatches=mb, rank=rank,
            checkpoint_layers=pc.checkpoint_layers,
        )
        peaks.append(allocator.peak_reserved)
        allocs.append(allocator)
        ctxs.append(ctx)
    return peaks, allocs, ctxs


def pc_mode_optim(optim):
    return optim.kind if isinstance(optim, Optimizer) else optim


# ---------------------------------------------------------------------- #
# 时间线重建
# ---------------------------------------------------------------------- #
def _build_report(allocator, ctx, cfg, prec, ok, oom, error,
                  per_gpu_peak) -> StepReport:
    events = allocator.events
    # 逐类别存活字节
    live: Dict[str, int] = {c: 0 for c in CATEGORIES}
    # block_id -> (category, req_bytes)
    block_meta: Dict[int, tuple] = {}
    ticks: List[TickPoint] = []

    peak_reserved = 0
    peak_tick = 0
    peak_phase = ""
    peak_live_snapshot: Dict[int, tuple] = {}

    for ev in events:
        if ev.kind == "alloc":
            live[ev.category] = live.get(ev.category, 0) + ev.req_bytes
            block_meta[ev.block_id] = (ev.category, ev.req_bytes, ev.name)
        else:
            cat, req, _ = block_meta.pop(ev.block_id)
            live[cat] -= req
        tp = TickPoint(
            tick=ev.tick, phase=ev.phase,
            allocated=ev.allocated, reserved=ev.reserved,
            fragmentation=ev.reserved - ev.allocated,
            by_category=dict(live),
        )
        ticks.append(tp)
        if ev.reserved > peak_reserved:
            peak_reserved = ev.reserved
            peak_tick = ev.tick
            peak_phase = ev.phase
            peak_live_snapshot = dict(block_meta)

    # 峰值时刻占用最大的张量
    top = sorted(
        ((bid, cat, req, name)
         for bid, (cat, req, name) in peak_live_snapshot.items()),
        key=lambda x: -x[2],
    )[:12]
    peak_top = [
        {"block_id": bid, "category": cat, "bytes": req, "name": name}
        for bid, cat, req, name in top
    ]

    persistent_cat = _persistent_bytes(events)
    static_bytes = sum(persistent_cat.values())

    reuse_hits = sum(1 for e in events if e.kind == "alloc" and e.reused)
    return StepReport(
        ok=ok, oom=oom, error=error, ticks=ticks,
        peak_allocated=allocator.peak_allocated,
        peak_reserved=allocator.peak_reserved,
        peak_fragment=allocator.peak_fragment,
        peak_internal_fragment=getattr(
            allocator, "peak_internal_fragment", 0
        ),
        peak_external_fragment=getattr(
            allocator, "peak_external_fragment", 0
        ),
        peak_tick=peak_tick, peak_phase=peak_phase,
        peak_top_tensors=peak_top,
        static_bytes=static_bytes,
        persistent_by_category=persistent_cat,
        total_events=len(events),
        alloc_events=sum(1 for e in events if e.kind == "alloc"),
        reuse_hits=reuse_hits,
        flops=ctx.flops,
        config={
            "model": cfg.model.name,
            "batch_size": cfg.batch_size,
            "precision": cfg.precision,
            "optimizer": cfg.optimizer,
            "parallel": cfg.parallel.mode,
            "world_size": cfg.parallel.world_size,
            "checkpoint": sorted(cfg.parallel.checkpoint_layers or []),
        },
        per_gpu_peak_reserved=per_gpu_peak,
    )


def _persistent_bytes(events) -> Dict[str, int]:
    """到步末仍存活的张量 (参数/梯度/状态等) 按类别汇总。"""
    alive: Dict[int, tuple] = {}
    for ev in events:
        if ev.kind == "alloc":
            alive[ev.block_id] = (ev.category, ev.req_bytes)
        else:
            alive.pop(ev.block_id, None)
    out: Dict[str, int] = {}
    for cat, req in alive.values():
        out[cat] = out.get(cat, 0) + req
    return out
