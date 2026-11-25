"""精度定义: 各数值类型字节数与精度组合预设。"""
from __future__ import annotations

from dataclasses import dataclass

# 元素字节数 (模拟显存, 与宿主机 NumPy 实际存储无关)
ELEM_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp64": 8,
    "int8": 1,
    "uint8": 1,
    "int32": 4,
    "int64": 8,
    "bool": 1,
}


def nbytes(shape, dtype):
    """给定形状与 dtype, 返回张量占用字节数。"""
    n = 1
    for d in shape:
        n *= int(d)
    return n * ELEM_BYTES[dtype]


@dataclass
class Precision:
    """一次训练的精度配置。

    name:    配置名
    param:   参数存储 dtype
    grad:    梯度 dtype
    optim:   优化器状态 dtype
    master:  混合精度主权重 dtype, None 表示无独立主权重
    compute: 算子计算 dtype (工作区默认 dtype)
    """

    name: str
    param: str
    grad: str
    optim: str
    master: str = None
    compute: str = None

    def __post_init__(self):
        if self.compute is None:
            self.compute = self.param


PRESETS = {
    "fp32": Precision("fp32", "fp32", "fp32", "fp32"),
    # 纯 fp16: 没有 fp32 主权重 (数值上不稳定, 但显存最省, 用于对比)
    "fp16": Precision("fp16", "fp16", "fp16", "fp16"),
    "bf16": Precision("bf16", "bf16", "bf16", "fp32"),
    # 混合精度: fp16 参数/激活 + fp32 主权重与优化器状态
    "mixed_fp16": Precision(
        "mixed_fp16", param="fp16", grad="fp16", optim="fp32",
        master="fp32", compute="fp16",
    ),
}


def get_precision(name) -> Precision:
    if isinstance(name, Precision):
        return name
    if name not in PRESETS:
        raise KeyError(f"未知精度配置: {name}, 可选 {list(PRESETS)}")
    return PRESETS[name]
