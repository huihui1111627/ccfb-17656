"""命令行入口: 不启动页面也能做预估、求解和真跑对照。

示例:
  python -m memlab.cli estimate --model tiny_gpt --batch 8 --gpu T4
  python -m memlab.cli max-batch --model tiny_gpt --gpu T4
  python -m memlab.cli min-gpus --model long_gpt_real --batch 32 --gpu T4
  python -m memlab.cli checkpoint --model long_gpt_real --batch 4
  python -m memlab.cli real --model small_cnn --batch 8
"""
from __future__ import annotations

import argparse
import json

from .checkpoint import analyze_checkpoint_options
from .estimator import StepConfig, estimate_step
from .framework import run_real_step
from .hardware import GPUS, get_gpu
from .models import REGISTRY
from .parallel import ParallelConfig
from .solver import max_batch_size, min_gpus_for_batch


def _build_cfg(a) -> StepConfig:
    pc = ParallelConfig(
        mode=a.mode, world_size=a.world_size,
        pipeline_microbatches=a.microbatches,
        checkpoint_layers=set(a.checkpoint.split(","))
        if a.checkpoint else None,
    )
    return StepConfig(
        model=REGISTRY[a.model], batch_size=a.batch,
        precision=a.precision, optimizer=a.optimizer,
        parallel=pc, seq_len_override=a.seq_len or None,
    )


def _cap(a) -> int:
    return get_gpu(a.gpu).capacity_bytes


def cmd_estimate(a):
    cfg = _build_cfg(a)
    r = estimate_step(cfg, _cap(a))
    out = {
        "ok": r.ok, "oom": r.oom, "error": r.error,
        "peak_reserved_MiB": round(r.peak_reserved / (1024 ** 2), 3),
        "peak_allocated_MiB": round(r.peak_allocated / (1024 ** 2), 3),
        "fragment_internal_MiB": round(
            r.peak_internal_fragment / (1024 ** 2), 3),
        "fragment_external_MiB": round(
            r.peak_external_fragment / (1024 ** 2), 3),
        "fragmentation_ratio": round(r.fragmentation_ratio, 4),
        "peak_phase": r.peak_phase, "peak_tick": r.peak_tick,
        "static_MiB": round(r.static_bytes / (1024 ** 2), 3),
        "persistent_MiB": {
            k: round(v / (1024 ** 2), 3)
            for k, v in r.persistent_by_category.items() if v
        },
        "per_gpu_peak_MiB": [
            round(x / (1024 ** 2), 3)
            for x in r.per_gpu_peak_reserved
        ],
        "flops": r.flops, "events": r.total_events,
        "block_reuse": r.reuse_hits,
        "top_tensors": [
            {"name": t["name"], "category": t["category"],
             "MiB": round(t["bytes"] / (1024 ** 2), 3)}
            for t in r.peak_top_tensors[:8]
        ],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


def cmd_max_batch(a):
    ans = max_batch_size(_build_cfg(a), _cap(a), high=a.high)
    print(f"最大批大小: {ans.value if ans.feasible else '不可行'}")
    print(f"峰值: {ans.peak_bytes / (1024 ** 2):.2f} MiB  "
          f"剩余: {ans.headroom_bytes / (1024 ** 2):.2f} MiB")
    print(ans.reason)


def cmd_min_gpus(a):
    ans = min_gpus_for_batch(
        _build_cfg(a), batch=a.batch, capacity_bytes=_cap(a),
        mode=a.solve_mode,
    )
    print(f"最少卡数 ({a.solve_mode}): "
          f"{ans.value if ans.feasible else '不可行'}")
    print(ans.reason)
    for e in ans.search_log:
        print(f"  world={e['world_size']} ok={e['ok']} "
              f"峰值/卡={e['worst_peak'] / (1024 ** 2):.2f} MiB")


def cmd_checkpoint(a):
    res = analyze_checkpoint_options(_build_cfg(a), _cap(a))
    print(f"基线峰值: {res['baseline_peak'] / (1024 ** 2):.2f} MiB")
    print("逐层成本/收益 (按性价比排序):")
    for o in res["options"]:
        print(f"  {o.layer_name:12s} 省 "
              f"{o.peak_saved_bytes / (1024 ** 2):7.2f} MiB  "
              f"额外FLOP {o.extra_flops_ratio * 100:5.1f}%  {o.note}")
    g = res["greedy_recommendation"]
    print(f"贪心推荐: {g['layers']}  共省 "
          f"{g['total_saved'] / (1024 ** 2):.2f} MiB  "
          f"额外FLOP {g['total_extra_flops'] / res['baseline_flops'] * 100:.1f}%")


def cmd_real(a):
    r = run_real_step(_build_cfg(a), _cap(a))
    print(f"真跑 ok={r.ok} 峰值={r.peak_reserved / (1024 ** 2):.3f} MiB  "
          f"耗时={r.wall_time_s:.3f}s FLOP={r.flops:.2e}")
    if not r.ok:
        print(r.error)


def build_parser():
    p = argparse.ArgumentParser(prog="memlab")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--model", required=True, choices=list(REGISTRY))
        sp.add_argument("--batch", type=int, default=8)
        sp.add_argument("--precision", default="fp32",
                        choices=["fp32", "fp16", "bf16", "mixed_fp16"])
        sp.add_argument("--optimizer", default="adam",
                        choices=["adam", "sgd"])
        sp.add_argument("--mode", default="single",
                        choices=["single", "dp", "zero", "pipeline"])
        sp.add_argument("--world-size", type=int, default=1)
        sp.add_argument("--microbatches", type=int, default=2)
        sp.add_argument("--checkpoint", default="")
        sp.add_argument("--seq-len", type=int, default=0)
        sp.add_argument("--gpu", default="A100-40G", choices=list(GPUS))

    sp = sub.add_parser("estimate"); common(sp)
    sp.set_defaults(func=cmd_estimate)
    sp = sub.add_parser("max-batch"); common(sp)
    sp.add_argument("--high", type=int, default=1024)
    sp.set_defaults(func=cmd_max_batch)
    sp = sub.add_parser("min-gpus"); common(sp)
    sp.add_argument("--solve-mode", default="zero",
                    choices=["dp", "zero", "pipeline"])
    sp.set_defaults(func=cmd_min_gpus)
    sp = sub.add_parser("checkpoint"); common(sp)
    sp.set_defaults(func=cmd_checkpoint)
    sp = sub.add_parser("real"); common(sp)
    sp.set_defaults(func=cmd_real)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
