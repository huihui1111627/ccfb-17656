"""可插拔的模型结构: 卷积网络、Transformer、超长序列 Transformer。

每个 ModelSpec 只描述结构与输入形状, 同一份 spec 同时驱动
预估路径 (spec 模式) 与真跑路径 (real 模式)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

from . import nn


@dataclass
class ModelSpec:
    name: str
    kind: str                 # "cnn" | "transformer"
    build_layers: Callable    # () -> List[Module]
    input_shape_fn: Callable  # (batch) -> dict(输入张量描述)
    vocab: int = 0
    seq_len: int = 0
    description: str = ""


# ---------------------------------------------------------------------- #
# 1. 卷积网络 (图像分类)
# ---------------------------------------------------------------------- #
def build_small_cnn():
    return [
        nn.Conv2d(3, 16, 3, padding=1, name="conv1"),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1, name="conv2"),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1, name="conv3"),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 256, name="fc1"),
        nn.ReLU(),
        nn.Linear(256, 10, name="fc2"),
    ]


def cnn_input(batch):
    # CIFAR 风格 32x32 输入, 标签 int32
    return {
        "x": {"shape": (batch, 3, 32, 32), "dtype": "fp32_input"},
        "y": {"shape": (batch, 10), "dtype": "mse_target"},
    }


SMALL_CNN = ModelSpec(
    name="small_cnn",
    kind="cnn",
    build_layers=build_small_cnn,
    input_shape_fn=cnn_input,
    description="3层卷积+池化+2层全连接的小型CNN (32x32图像, 10分类)",
)


# ---------------------------------------------------------------------- #
# 2. Transformer 语言模型
# ---------------------------------------------------------------------- #
def build_tiny_gpt(dim, heads, layers, ff_mult=4):
    def _build():
        mods = [nn.Embedding(1000, dim, name="embed")]
        for i in range(layers):
            mods.append(nn.TransformerBlock(
                dim, heads, ff_mult=ff_mult, causal=True,
                name=f"block{i}",
            ))
        mods.append(nn.LayerNorm(dim, name="ln_f"))
        mods.append(nn.Linear(dim, 1000, bias=False, name="lm_head"))
        return mods
    return _build


def lm_input(batch, seq, vocab):
    return {
        "tokens": {"shape": (batch, seq), "dtype": "int32"},
        "targets": {"shape": (batch * seq,), "dtype": "int32"},
    }


TINY_GPT = ModelSpec(
    name="tiny_gpt",
    kind="transformer",
    build_layers=build_tiny_gpt(dim=128, heads=4, layers=4),
    input_shape_fn=lambda b: lm_input(b, 128, 1000),
    vocab=1000,
    seq_len=128,
    description="4层小型因果Transformer (d=128, h=4, S=128, 词表1000)",
)


# ---------------------------------------------------------------------- #
# 3. 超长序列 Transformer (注意力 SxS 矩阵主导显存)
# ---------------------------------------------------------------------- #
LONG_GPT = ModelSpec(
    name="long_gpt",
    kind="transformer",
    build_layers=build_tiny_gpt(dim=128, heads=4, layers=4),
    input_shape_fn=lambda b: lm_input(b, 1024, 1000),
    vocab=1000,
    seq_len=1024,
    description="同结构但序列长度1024, 注意力矩阵显存随S^2膨胀的长序列模型",
)

# 真跑用的更短版本 (宿主机 NumPy 对照), 结构完全一致
LONG_GPT_REAL = ModelSpec(
    name="long_gpt_real",
    kind="transformer",
    build_layers=build_tiny_gpt(dim=128, heads=4, layers=4),
    input_shape_fn=lambda b: lm_input(b, 512, 1000),
    vocab=1000,
    seq_len=512,
    description="长序列模型的宿主机对照版 (S=512, 结构不变)",
)


REGISTRY: Dict[str, ModelSpec] = {
    m.name: m for m in (SMALL_CNN, TINY_GPT, LONG_GPT, LONG_GPT_REAL)
}


def get_model(name) -> ModelSpec:
    if name not in REGISTRY:
        raise KeyError(f"未知模型: {name}, 可选 {list(REGISTRY)}")
    return REGISTRY[name]
