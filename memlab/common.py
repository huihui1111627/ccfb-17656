"""全局常量: 显存类别与训练阶段标记。"""
from __future__ import annotations

# 显存内容类别 (堆叠图的分层依据)
CAT_PARAM = "param"            # 参数权重
CAT_GRAD = "grad"              # 梯度
CAT_OPTIM = "optim"            # 优化器状态 (m, v, momentum ...)
CAT_MASTER = "master"          # 混合精度下的 fp32 主权重
CAT_ACT = "activation"         # 前向保留给反向用的中间结果
CAT_WORK = "workspace"         # 算子临时空间 (softmax/im2col/gemm 缓冲等)
CAT_INPUT = "input"            # 本步输入数据与标签
CAT_COMM = "communication"     # 集合通信缓冲 (all-gather/reduce 缓冲)

CATEGORIES = [
    CAT_PARAM,
    CAT_MASTER,
    CAT_GRAD,
    CAT_OPTIM,
    CAT_ACT,
    CAT_WORK,
    CAT_INPUT,
    CAT_COMM,
]

CAT_LABELS_CN = {
    CAT_PARAM: "参数",
    CAT_MASTER: "FP32主权重",
    CAT_GRAD: "梯度",
    CAT_OPTIM: "优化器状态",
    CAT_ACT: "保存的激活",
    CAT_WORK: "临时工作区",
    CAT_INPUT: "输入数据",
    CAT_COMM: "通信缓冲",
    "fragment": "碎片/预留损耗",
}

# 训练阶段 (峰值定位用)
PHASE_BUILD = "build"
PHASE_FWD = "forward"
PHASE_BWD = "backward"
PHASE_OPT = "optimizer_step"
PHASE_COMM = "communication"
