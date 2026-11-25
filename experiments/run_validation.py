"""批量验证实验: 多模型 × 批大小 × 精度 × 并行 × checkpoint 组合。

对每个组合:
1. 预估器算峰值 (spec 模式, 无数值计算);
2. 小框架真跑一步 (real 模式, NumPy 实算);
3. 逐个对比峰值字节、OOM 判定;
4. 汇总误差分布与判定错误次数, 落盘 JSON 供页面读取。

真跑模型用宿主机可承受的较小规模 (LONG_GPT_REAL, S=512),
预估同时给出生产规模 (LONG_GPT, S=1024) 的数字。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

from memlab.estimator import StepConfig, estimate_step
from memlab.framework import run_real_step
from memlab.hardware import GPUS, get_gpu
from memlab.models import LONG_GPT_REAL, SMALL_CNN, TINY_GPT, get_model
from memlab.parallel import ParallelConfig


def build_matrix(restrict_real=True):
    """生成实验组合矩阵。"""
    rows = []
    # (model, [batches], [precisions], [optimizers], parallel specs)
    parallel_specs = [
        ("single", 1), ("dp", 2), ("zero", 2), ("zero", 4),
        ("pipeline", 2),
    ]
    for model in (SMALL_CNN, TINY_GPT, LONG_GPT_REAL):
        batches = [1, 2, 4, 8] if model.kind == "transformer" else [2, 4, 8, 16]
        for b in batches:
            for prec in ("fp32", "fp16", "mixed_fp16"):
                for opt in ("adam", "sgd"):
                    for mode, world in parallel_specs:
                        if mode == "pipeline" and b % 2 != 0:
                            continue
                        rows.append((model.name, b, prec, opt, mode, world,
                                     None))
        # checkpoint 组合 (单卡, 代表性层)
        ck_choices = [None]
        if model.kind == "cnn":
            ck_choices = [None, {"conv2"}, {"conv1", "conv3"}]
        else:
            ck_choices = [None, {"block1"}, {"block0", "block2"}]
        for b in (4, 8):
            for ck in ck_choices:
                if ck is None:
                    continue
                rows.append((model.name, b, "fp32", "adam", "single", 1,
                             sorted(ck)))
    return rows


def run_one(row, capacity):
    model_name, b, prec, opt, mode, world, ckpt = row
    model = get_model(model_name)
    cfg = StepConfig(
        model=model, batch_size=b, precision=prec, optimizer=opt,
    )
    cfg.parallel = ParallelConfig(
        mode=mode, world_size=world,
        pipeline_microbatches=2,
        checkpoint_layers=set(ckpt) if ckpt else None,
    )
    t0 = time.perf_counter()
    est = estimate_step(cfg, capacity)
    est_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    real = run_real_step(cfg, capacity)
    real_ms = (time.perf_counter() - t0) * 1000

    peak_err = 0
    if est.ok and real.ok and real.peak_reserved > 0:
        peak_err = abs(est.peak_reserved - real.peak_reserved) \
            / real.peak_reserved
    decision_ok = (est.ok == real.ok)
    return {
        "model": model_name, "batch": b, "precision": prec,
        "optimizer": opt, "mode": mode, "world_size": world,
        "checkpoint": ckpt or [],
        "est_ok": est.ok, "real_ok": real.ok,
        "decision_ok": decision_ok,
        "est_peak": est.peak_reserved,
        "real_peak": real.peak_reserved,
        "est_peak_mib": round(est.peak_reserved / (1024 ** 2), 3),
        "real_peak_mib": round(real.peak_reserved / (1024 ** 2), 3),
        "peak_rel_error": peak_err,
        "peak_abs_error_bytes": abs(
            est.peak_reserved - real.peak_reserved
        ),
        "fragment_mib": round(est.peak_fragment / (1024 ** 2), 3),
        "est_time_ms": round(est_ms, 2),
        "real_time_ms": round(real_ms, 2),
        "real_flops": real.flops,
        "per_gpu_peak_mib": [
            round(x / (1024 ** 2), 3) for x in
            (est.per_gpu_peak_reserved or [est.peak_reserved])
        ],
        "oom_error": real.error if not real.ok else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="A100-40G")
    ap.add_argument("--out", default="results/validation.json")
    ap.add_argument("--limit", type=int, default=0,
                    help="只跑前 N 个组合 (调试用)")
    args = ap.parse_args()

    capacity = get_gpu(args.gpu).capacity_bytes
    rows = build_matrix()
    if args.limit:
        rows = rows[:args.limit]
    print(f"共 {len(rows)} 个组合, 卡型 {args.gpu} "
          f"可用 {capacity // (1024 ** 2)} MiB")

    results = []
    decision_errors = 0
    peak_errors = []
    t_start = time.perf_counter()
    for i, row in enumerate(rows):
        r = run_one(row, capacity)
        results.append(r)
        if not r["decision_ok"]:
            decision_errors += 1
        if r["est_ok"] and r["real_ok"]:
            peak_errors.append(r["peak_rel_error"])
        if (i + 1) % 25 == 0 or i == len(rows) - 1:
            print(f"  进度 {i + 1}/{len(rows)}")

    wall = time.perf_counter() - t_start
    summary = {
        "gpu": args.gpu,
        "capacity_bytes": capacity,
        "total_cases": len(results),
        "decision_mismatches": decision_errors,
        "decision_accuracy": (
            len(results) - decision_errors
        ) / max(len(results), 1),
        "peak_error_max": max(peak_errors) if peak_errors else 0,
        "peak_error_mean": (
            sum(peak_errors) / len(peak_errors)
            if peak_errors else 0
        ),
        "exact_byte_match_cases": sum(
            1 for r in results if r["peak_abs_error_bytes"] == 0
        ),
        "wall_time_s": round(wall, 2),
    }
    out = {"summary": summary, "results": results,
           "gpu_catalog": {k: asdict(v) for k, v in GPUS.items()}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n=== 汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n结果已写入 {args.out}")


if __name__ == "__main__":
    main()
