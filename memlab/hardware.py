"""常见 GPU 卡型的显存容量与理论算力 (用于容量判定与吞吐换算)。

容量取厂商标称值; 实际框架预留约 5~10% 给上下文/CUDA 上下文,
这里通过 usable_ratio 暴露给配置, 判定时用可用容量。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPUSpec:
    name: str
    memory_gib: float          # 标称显存 (GiB)
    fp16_tflops: float         # 张量核心 FP16 理论算力
    memory_bw_gbs: float       # 显存带宽 GB/s
    usable_ratio: float = 0.92 # 扣除 CUDA 上下文/框架预留后可用比例

    @property
    def capacity_bytes(self):
        return int(self.memory_gib * (1024 ** 3) * self.usable_ratio)


GPUS = {
    "A100-80G": GPUSpec("A100-80G", 80, 312, 2039),
    "A100-40G": GPUSpec("A100-40G", 40, 312, 1555),
    "A10": GPUSpec("A10", 24, 125, 600),
    "V100-32G": GPUSpec("V100-32G", 32, 125, 900),
    "T4": GPUSpec("T4", 16, 65, 320),
    "L4": GPUSpec("L4", 24, 121, 300),
    "H100-80G": GPUSpec("H100-80G", 80, 989, 3350),
    "RTX4090": GPUSpec("RTX4090", 24, 330, 1008),
    "RTX3090": GPUSpec("RTX3090", 24, 142, 936),
}


def get_gpu(name) -> GPUSpec:
    if name not in GPUS:
        raise KeyError(f"未知卡型: {name}, 可选 {list(GPUS)}")
    return GPUS[name]
