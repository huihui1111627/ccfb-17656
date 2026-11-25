"""真跑框架的数值正确性: 多步训练后 loss 应当下降。

记账一致性由 test_spec_vs_real 保证; 这里确认 NumPy 实算路径
的前向/反向/优化器构成了一个真正能学习的训练循环。
"""
import numpy as np

from memlab.allocator import BlockAllocator
from memlab.common import PHASE_BUILD
from memlab.dtype import get_precision
from memlab.engine import TrainEngine, allocate_inputs
from memlab.framework import run_real_step
from memlab.estimator import StepConfig
from memlab.nn import ParamStore
from memlab.optimizer import Optimizer
from memlab.models import SMALL_CNN, TINY_GPT


def train_multi_step(model, batch, precision, optimizer, steps, cap):
    prec = get_precision(precision)
    layers = model.build_layers()
    alloc = BlockAllocator(cap, materialize=True)
    from memlab.tensor import Context
    ctx = Context(alloc, "real")
    ctx.set_phase(PHASE_BUILD)
    store = ParamStore(ctx, prec)
    for layer in layers:
        for pn, sh in layer.params():
            store.declare(pn, sh)
        layer.build(ctx, store)
    opt = Optimizer(optimizer, lr=1e-2)
    opt.build(ctx, store)
    losses = []
    for _ in range(steps):
        inp = allocate_inputs(ctx, model, batch, prec)
        engine = TrainEngine(layers, ctx, store, prec)
        loss_val = engine.train_step(inp)
        losses.append(loss_val)
        opt.step(ctx, store)
        for t in inp.values():
            if t.alive:
                ctx.free_later(t)
    return losses


def main():
    cap = 2 * 1024 ** 3
    all_ok = True
    for model, b in [(SMALL_CNN, 16), (TINY_GPT, 8)]:
        losses = train_multi_step(model, b, "fp32", "adam", 8, cap)
        first, last = np.mean(losses[:2]), np.mean(losses[-2:])
        down = last < first
        all_ok &= down
        print(f"[{'OK ' if down else 'FAIL'}] {model.name} "
              f"loss {first:.4f} -> {last:.4f}")
        print("       轨迹:", " ".join(f"{x:.3f}" for x in losses))
    print("\nloss 收敛" if all_ok else "\nloss 未下降!")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
