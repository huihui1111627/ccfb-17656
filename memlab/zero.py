"""ZeRO-3 参数存储: 参数/梯度/优化器状态全部按 rank 分片。

前向/反向需要完整权重时, 通过 get_param 触发 all-gather,
在 ctx 上产生一个临时完整权重张量 (CAT_COMM), 用完由 release_param 释放。
层代码无感知: 它只调用 store.get_param / store.release_param。
"""
from __future__ import annotations

from typing import Dict, Optional

from .common import CAT_COMM, CAT_PARAM
from .nn import ParamStore
from .parallel import shard_shape
from .tensor import Context, Tensor


class ZeroShardedStore(ParamStore):
    def __init__(self, ctx, precision, world_size, rank=0):
        super().__init__(ctx, precision)
        self.world_size = world_size
        self.rank = rank
        # 持久分片参数 (本卡拥有)
        self.shard_params: Dict[str, Tensor] = {}
        # 临时 all-gather 出的完整权重: name -> tensor
        self.gathered: Dict[str, Tensor] = {}
        # 标记当前 step 是否需要在 get 时自动 gather
        self.gather_enabled = True

    def alloc_param(self, name, shape, dtype=None, seed=0):
        full_shape = tuple(int(d) for d in shape)
        self.shapes[name] = full_shape
        sshape = shard_shape(full_shape, self.world_size, self.rank)
        t = self.ctx.alloc(
            sshape, dtype or self.prec.param, CAT_PARAM,
            f"param.shard:{name}",
        )
        if self.ctx.mode == "real":
            import numpy as np
            from .ops import stable_seed
            rng = np.random.default_rng(stable_seed(name))
            full = rng.standard_normal(full_shape) * 0.1
            start = self._shard_start(full_shape[0])
            t.data[:] = full[start:start + sshape[0]].reshape(
                sshape
            ).astype(t.data.dtype)
        self.shard_params[name] = t
        # params 字典里放分片, 保持基类接口
        self.params[name] = t
        return t

    def _shard_start(self, first_dim):
        base = first_dim // self.world_size
        rem = first_dim % self.world_size
        return self.rank * base + min(self.rank, rem)

    def get_param(self, name) -> Tensor:
        if not self.gather_enabled or name in self.gathered:
            return self.gathered.get(name) or self.params[name]
        # all-gather: 产生完整权重临时缓冲
        full = self.ctx.alloc(
            self.shapes[name], self.prec.param, CAT_COMM,
            f"allgather:{name}",
        )
        if self.ctx.mode == "real":
            # 单机模拟: 用本卡分片重复填充出完整张量 (数值不影响记账)
            import numpy as np
            reps = int(np.ceil(
                full.shape[0] / max(1, self.shard_params[name].shape[0])
            ))
            tile = np.repeat(
                self.shard_params[name].data, reps, axis=0
            )[:full.shape[0]]
            full.data[:] = tile.reshape(full.shape).astype(full.data.dtype)
        self.gathered[name] = full
        return full

    def release_param(self, name):
        """层用完权重后释放 all-gather 缓冲。"""
        t = self.gathered.pop(name, None)
        if t is not None and t.alive:
            self.ctx.free_later(t)

    def release_all_gathered(self):
        for name in list(self.gathered.keys()):
            self.release_param(name)

    def persistent_blocks(self) -> set:
        ids = {t.block_id for t in self.shard_params.values()}
        ids.update(t.block_id for t in self.grads.values())
        return ids
