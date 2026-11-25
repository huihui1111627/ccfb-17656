"""memlab: GPU memory estimation lab.

预测器与真实小框架共用同一套层实现与分块分配器:
- SpecTensor: 只做形状推导, 不做数值计算 (预估路径)
- RealTensor: NumPy 真实计算   (真跑路径)
两条路径产生逐条一致的 alloc/free 事件。
"""

__version__ = "1.0.0"
