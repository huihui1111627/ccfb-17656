"""验证 spec(预估) 与 real(真跑) 两条路径事件序列严格一致。

这是整个工具可信度的根基: 同一份层代码、同一个分配器,
只在张量实现上分叉, 事件必须逐条相同。
"""
from memlab.allocator import BlockAllocator
from memlab.common import PHASE_BUILD
from memlab.dtype import get_precision
from memlab.engine import TrainEngine, allocate_inputs
from memlab.estimator import StepConfig, estimate_step
from memlab.framework import run_real_step
from memlab.nn import ParamStore
from memlab.optimizer import Optimizer
from memlab.tensor import Context


def run_spec_raw(model, batch, precision, optimizer, capacity, ckpt=None):
    layers = model.build_layers()
    prec = get_precision(precision)
    opt = Optimizer(optimizer)
    alloc = BlockAllocator(capacity, materialize=False)
    ctx = Context(alloc, "spec")
    ctx.set_phase(PHASE_BUILD)
    store = ParamStore(ctx, prec)
    for layer in layers:
        for pname, shape in layer.params():
            store.declare(pname, shape)
        layer.build(ctx, store)
    opt.build(ctx, store)
    inp = allocate_inputs(ctx, model, batch, prec)
    engine = TrainEngine(layers, ctx, store, prec, ckpt)
    engine.train_step(inp)
    opt.step(ctx, store)
    for t in inp.values():
        if t.alive:
            ctx.free_later(t)
    return alloc


def event_key(events):
    return [
        (e.kind, e.name, e.category, e.req_bytes, e.block_bytes, e.phase)
        for e in events
    ]


def main():
    from memlab.models import SMALL_CNN, TINY_GPT, LONG_GPT_REAL
    cap = 2 * 1024 ** 3
    cases = [
        (SMALL_CNN, 8, "fp32", "adam", "single", 1, None),
        (SMALL_CNN, 4, "mixed_fp16", "sgd", "single", 1, None),
        (TINY_GPT, 4, "fp32", "adam", "single", 1, None),
        (TINY_GPT, 2, "fp16", "adam", "single", 1, None),
        (SMALL_CNN, 8, "fp32", "adam", "single", 1, {"conv2"}),
        (TINY_GPT, 4, "fp32", "adam", "dp", 2, None),
        (TINY_GPT, 4, "fp32", "adam", "zero", 2, None),
        (TINY_GPT, 4, "fp32", "adam", "zero", 4, None),
        (TINY_GPT, 4, "fp32", "adam", "pipeline", 2, None),
        (LONG_GPT_REAL, 2, "fp32", "adam", "pipeline", 4, None),
    ]
    all_ok = True
    for model, b, prec_name, opt_name, mode, world, ckpt in cases:
        cfg = StepConfig(
            model=model, batch_size=b, precision=prec_name,
            optimizer=opt_name,
        )
        cfg.parallel.mode = mode
        cfg.parallel.world_size = world
        if ckpt:
            cfg.parallel.checkpoint_layers = set(ckpt)
        spec_alloc = run_spec_mode(cfg, cap)
        real = run_real_step(cfg, cap)
        est = estimate_step(cfg, cap)
        if mode == "pipeline":
            # 逐 rank 比较事件序列与峰值
            spec_allocs = run_spec_pipeline_all_ranks(cfg, cap)
            real_allocs = real.rank_allocators
            same = len(spec_allocs) == len(real_allocs) and all(
                event_key(s.events) == event_key(r.events)
                for s, r in zip(spec_allocs, real_allocs)
            )
            peak_match = all(
                s.peak_reserved == r.peak_reserved
                for s, r in zip(spec_allocs, real_allocs)
            ) and max(a.peak_reserved for a in spec_allocs) == \
                est.peak_reserved
            spec_peak = max(a.peak_reserved for a in spec_allocs)
        else:
            same = event_key(spec_alloc.events) == event_key(real.events)
            peak_match = (spec_alloc.peak_reserved == real.peak_reserved
                          == est.peak_reserved)
            spec_peak = spec_alloc.peak_reserved
        status = "OK " if (same and peak_match) else "FAIL"
        if not (same and peak_match):
            all_ok = False
        print(f"[{status}] {model.name:14s} b={b} {prec_name:10s} "
              f"{opt_name:4s} {mode:8s} w={world} ckpt={ckpt}")
        print(f"       events spec~real identical={same} "
              f"peak={spec_peak/1e6:.3f}MB "
              f"real={real.peak_reserved/1e6:.3f}MB")
    print("\n全部一致" if all_ok else "\n存在不一致!")
    return 0 if all_ok else 1


def run_spec_pipeline_all_ranks(cfg, capacity):
    from memlab.allocator import BlockAllocator
    from memlab.tensor import Context
    from memlab.dtype import get_precision
    from memlab.optimizer import Optimizer
    from memlab.parallel import run_pipeline
    prec = get_precision(cfg.precision)
    layers = cfg.model.build_layers()
    pc = cfg.parallel
    allocs = []
    for rank in range(pc.world_size):
        alloc = BlockAllocator(capacity, materialize=False)
        ctx = Context(alloc, "spec")
        run_pipeline(
            layers, cfg.model, cfg.batch_size, prec,
            Optimizer(cfg.optimizer), ctx,
            world_size=pc.world_size,
            microbatches=pc.pipeline_microbatches, rank=rank,
            checkpoint_layers=pc.checkpoint_layers,
        )
        allocs.append(alloc)
    return allocs


def run_spec_mode(cfg, capacity):
    """按 cfg.parallel 跑 spec 模式, 返回 (主) allocator。"""
    from memlab.allocator import BlockAllocator
    from memlab.tensor import Context
    from memlab.dtype import get_precision
    from memlab.optimizer import Optimizer
    from memlab.parallel import (
        run_data_parallel, run_pipeline,
    )
    prec = get_precision(cfg.precision)
    layers = cfg.model.build_layers()
    pc = cfg.parallel
    if pc.mode == "pipeline":
        # 取峰值最高的 rank 作为代表, 同时保留各 rank allocator
        allocs = []
        for rank in range(pc.world_size):
            alloc = BlockAllocator(capacity, materialize=False)
            ctx = Context(alloc, "spec")
            run_pipeline(
                layers, cfg.model, cfg.batch_size, prec,
                Optimizer(cfg.optimizer), ctx,
                world_size=pc.world_size,
                microbatches=pc.pipeline_microbatches, rank=rank,
                checkpoint_layers=pc.checkpoint_layers,
            )
            allocs.append(alloc)
        return max(allocs, key=lambda a: a.peak_reserved)
    alloc = BlockAllocator(capacity, materialize=False)
    ctx = Context(alloc, "spec")
    if pc.mode == "single":
        from memlab.common import PHASE_BUILD
        from memlab.nn import ParamStore
        from memlab.engine import TrainEngine, allocate_inputs
        ctx.set_phase(PHASE_BUILD)
        store = ParamStore(ctx, prec)
        for layer in layers:
            for pn, sh in layer.params():
                store.declare(pn, sh)
            layer.build(ctx, store)
        opt = Optimizer(cfg.optimizer)
        opt.build(ctx, store)
        inp = allocate_inputs(ctx, cfg.model, cfg.batch_size, prec)
        TrainEngine(
            layers, ctx, store, prec, pc.checkpoint_layers
        ).train_step(inp)
        opt.step(ctx, store)
        for t in inp.values():
            if t.alive:
                ctx.free_later(t)
    else:
        run_data_parallel(
            layers, cfg.model, cfg.batch_size, prec,
            Optimizer(cfg.optimizer), ctx,
            world_size=pc.world_size, mode=pc.mode,
            checkpoint_layers=pc.checkpoint_layers,
        )
    return alloc


if __name__ == "__main__":
    raise SystemExit(main())
