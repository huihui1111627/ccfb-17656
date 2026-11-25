"""容量规划求解器。

- max_batch_size: 给定卡型容量, 二分求能跑的最大批大小;
- min_gpus:       给定批大小, 求最少需要几张卡 (DP/ZeRO/流水);
- 判定标准统一为 peak_reserved <= capacity (已含碎片),
  与真实框架的 OOM 判定使用同一把尺子。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from .estimator import StepConfig, StepReport, estimate_step
from .parallel import ParallelConfig


@dataclass
class CapacityAnswer:
    feasible: bool
    value: int                    # 最大 batch 或最少卡数
    peak_bytes: int
    headroom_bytes: int
    report: Optional[StepReport]
    search_log: List[dict]
    reason: str = ""


def _fits(cfg: StepConfig, capacity: int) -> StepReport:
    return estimate_step(cfg, capacity)


def max_batch_size(
    base_cfg: StepConfig, capacity_bytes: int,
    low: int = 1, high: int = 4096,
) -> CapacityAnswer:
    """二分搜索可运行的最大批大小。"""
    log = []

    def make(b):
        c = StepConfig(
            model=base_cfg.model, batch_size=b,
            precision=base_cfg.precision,
            optimizer=base_cfg.optimizer,
            parallel=base_cfg.parallel,
            gpu_memory_bytes=capacity_bytes,
            seq_len_override=base_cfg.seq_len_override,
        )
        return c

    # 先确认 low 可行
    r0 = _fits(make(low), capacity_bytes)
    log.append({"batch": low, "ok": r0.ok,
                "peak": r0.peak_reserved})
    if not r0.ok:
        return CapacityAnswer(
            feasible=False, value=0, peak_bytes=r0.peak_reserved,
            headroom_bytes=0, report=r0, search_log=log,
            reason=f"最小批大小 {low} 都无法运行: {r0.error}",
        )

    # 指数扩张找上界
    hi = low
    r_hi = r0
    while hi < high:
        nxt = min(hi * 2, high)
        r = _fits(make(nxt), capacity_bytes)
        log.append({"batch": nxt, "ok": r.ok,
                    "peak": r.peak_reserved})
        if not r.ok:
            break
        hi, r_hi = nxt, r
        if nxt == high:
            return CapacityAnswer(
                feasible=True, value=hi,
                peak_bytes=r_hi.peak_reserved,
                headroom_bytes=capacity_bytes - r_hi.peak_reserved,
                report=r_hi, search_log=log,
                reason="达到搜索上界仍可行",
            )

    # 二分 (lo 可行, hi 不可行)
    lo = hi
    while lo + 1 < nxt:
        mid = (lo + nxt) // 2
        r = _fits(make(mid), capacity_bytes)
        log.append({"batch": mid, "ok": r.ok,
                    "peak": r.peak_reserved})
        if r.ok:
            lo, r_hi = mid, r
        else:
            nxt = mid
    return CapacityAnswer(
        feasible=True, value=lo,
        peak_bytes=r_hi.peak_reserved,
        headroom_bytes=capacity_bytes - r_hi.peak_reserved,
        report=r_hi, search_log=log,
    )


def min_gpus_for_batch(
    base_cfg: StepConfig, batch: int, capacity_bytes: int,
    mode: str = "zero", max_world: int = 64,
    pipeline_microbatches: int = 2,
) -> CapacityAnswer:
    """给定批大小, 求最少 GPU 数。

    mode:
      dp       数据并行 (每卡完整状态, 只分担 batch);
      zero     ZeRO 分片 (状态随卡数下降, 推荐);
      pipeline 层切分流水并行 (每卡只持一段参数)。
    """
    log = []
    for world in range(1, max_world + 1):
        if mode == "pipeline" and batch % pipeline_microbatches != 0:
            continue
        pc = ParallelConfig(
            mode=("single" if world == 1 else mode),
            world_size=world,
            pipeline_microbatches=pipeline_microbatches,
            checkpoint_layers=base_cfg.parallel.checkpoint_layers,
        )
        cfg = StepConfig(
            model=base_cfg.model, batch_size=batch,
            precision=base_cfg.precision,
            optimizer=base_cfg.optimizer, parallel=pc,
        )
        r = _fits(cfg, capacity_bytes)
        per_gpu = (r.per_gpu_peak_reserved or [r.peak_reserved])
        worst = max(per_gpu)
        log.append({
            "world_size": world, "ok": r.ok,
            "peak_per_gpu": per_gpu, "worst_peak": worst,
        })
        if r.ok and worst <= capacity_bytes:
            return CapacityAnswer(
                feasible=True, value=world, peak_bytes=worst,
                headroom_bytes=capacity_bytes - worst,
                report=r, search_log=log,
                reason=f"{mode} 并行下 {world} 卡可容纳批大小 {batch}",
            )
    last = log[-1] if log else None
    return CapacityAnswer(
        feasible=False, value=-1,
        peak_bytes=last["worst_peak"] if last else 0,
        headroom_bytes=0, report=None, search_log=log,
        reason=f"{max_world} 卡内无法容纳批大小 {batch}",
    )
