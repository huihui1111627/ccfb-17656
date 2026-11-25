"""gradient checkpointing 的成本/收益分析与层选择建议。

对每个可 checkpoint 的层 (复合层, 如 TransformerBlock / Conv2d):
- 收益: 该层前向激活不保留, 峰值显存下降量;
- 成本: 反向时重算一次前向, 额外 FLOP (= 该层前向 FLOP);
- 性价比: 每 1% 额外 FLOP 换来多少 MiB 显存。
给出从优到劣的排序与若干推荐组合。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .estimator import StepConfig, estimate_step
from .nn import (
    Conv2d, Embedding, MultiHeadAttention, TransformerBlock,
)


@dataclass
class CheckpointOption:
    layer_name: str
    layer_type: str
    peak_saved_bytes: int          # 相对无 checkpoint 的峰值下降
    extra_flops: int               # 重算额外 FLOP
    extra_flops_ratio: float       # 占基线总 FLOP 比例
    peak_with_this: int
    score: float                   # saved_MiB / extra_flops_ratio
    note: str = ""


def _baseline_flops(cfg: StepConfig, capacity: int) -> int:
    return estimate_step(cfg, capacity).flops


def analyze_checkpoint_options(
    cfg: StepConfig, capacity_bytes: int,
) -> Dict:
    layers = cfg.model.build_layers()
    # 可 checkpoint 的层: 有内部激活、非 Embedding 的复合/计算层
    candidates = []
    for layer in layers:
        if isinstance(layer, Embedding):
            continue
        if isinstance(layer, (TransformerBlock, Conv2d, MultiHeadAttention)):
            candidates.append((layer.name, type(layer).__name__))

    base = estimate_step(cfg, capacity_bytes)
    base_peak = base.peak_reserved
    base_flops = base.flops

    options: List[CheckpointOption] = []
    for name, ltype in candidates:
        c = _clone_cfg(cfg, checkpoint={name})
        r = estimate_step(c, capacity_bytes)
        saved = base_peak - r.peak_reserved
        extra = r.flops - base_flops
        ratio = extra / base_flops if base_flops else 0.0
        score = (saved / (1024 ** 2)) / max(ratio * 100, 1e-6)
        note = ""
        if saved <= 0:
            note = "该层不是峰值瓶颈, 单独 checkpoint 无收益"
        options.append(CheckpointOption(
            layer_name=name, layer_type=ltype,
            peak_saved_bytes=max(saved, 0), extra_flops=extra,
            extra_flops_ratio=ratio, peak_with_this=r.peak_reserved,
            score=score, note=note,
        ))

    options.sort(key=lambda o: -o.score)

    # 推荐组合: 贪心累加, 直到达到目标 (默认省 20%) 或收益归零
    greedy = _greedy_combo(cfg, capacity_bytes, candidates,
                           target_ratio=0.20)
    return {
        "baseline_peak": base_peak,
        "baseline_flops": base_flops,
        "options": options,
        "greedy_recommendation": greedy,
    }


def _greedy_combo(cfg, capacity, candidates, target_ratio):
    chosen: List[str] = []
    base = estimate_step(cfg, capacity)
    target_peak = base.peak_reserved * (1 - target_ratio)
    cur_peak = base.peak_reserved
    cur_flops = base.flops
    remaining = [n for n, _ in candidates]
    steps = []
    while remaining:
        best = None
        for name in remaining:
            r = estimate_step(
                _clone_cfg(cfg, checkpoint=set(chosen + [name])),
                capacity,
            )
            gain = cur_peak - r.peak_reserved
            if best is None or gain > best[1]:
                best = (name, gain, r)
        name, gain, r = best
        if gain <= 0:
            break
        chosen.append(name)
        remaining.remove(name)
        cur_peak = r.peak_reserved
        cur_flops = r.flops
        steps.append({
            "add": name,
            "peak": cur_peak,
            "saved_so_far": base.peak_reserved - cur_peak,
            "flops_overhead": cur_flops - base.flops,
            "flops_overhead_ratio":
                (cur_flops - base.flops) / base.flops,
        })
        if cur_peak <= target_peak:
            break
    return {
        "layers": chosen,
        "final_peak": cur_peak,
        "total_saved": base.peak_reserved - cur_peak,
        "total_extra_flops": cur_flops - base.flops,
        "steps": steps,
        "target_ratio": target_ratio,
    }


def _clone_cfg(cfg: StepConfig, checkpoint: set) -> StepConfig:
    pc = dataclasses.replace(
        cfg.parallel,
        checkpoint_layers=set(checkpoint) if checkpoint else None,
    )
    return StepConfig(
        model=cfg.model, batch_size=cfg.batch_size,
        precision=cfg.precision, optimizer=cfg.optimizer,
        parallel=pc, seq_len_override=cfg.seq_len_override,
    )
