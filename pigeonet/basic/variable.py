from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import numpy as np

if TYPE_CHECKING:
    from pigeonet.basic.function import *

__all__ = [name for name in globals() if not name.startswith('_')]


class Variable:
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data: np.ndarray = data
        self._creator: Optional[Function] = None
        self.grad: Optional[np.ndarray] = None
        self.generation: int = 0  # 代数，用于标记函数反向传播先后顺序

    @property
    def creator(self) -> Optional[Function]:
        return self._creator

    @creator.setter
    def creator(self, func: Optional[Function]):
        self._creator = func
        self.generation = func.generation + 1

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'

        res = str(self.data).replace('\n', '\n         ')  # 空格对其输出格式
        return f"Variable({res})"

    def backward(self, need_grad=False):
        if self.grad is None:
            # 初始化梯度
            self.grad = np.ones_like(self.data)

        # 迭代 根据代数进行广度优先遍历计算图
        funcs: list[Function] = []
        backwarded_set = set()

        def add_func(f: Function):
            if f in backwarded_set:
                return
            backwarded_set.add(f)
            funcs.append(f)
            funcs.sort(key=lambda x: x.generation, reverse=False)

        add_func(self._creator)

        while funcs:
            f = funcs.pop()
            gys = [y().grad for y in f.outputs]  # y: ReferenceType[Tuple[Variable]]
            gxs = f.backward(*gys)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad += gx

                if x.creator is not None:
                    add_func(x.creator)

            if not need_grad:
                for y in f.outputs:
                    y().grad = None  # 清除不需要的梯度

    def clear_grad(self):
        self.grad = np.ones_like(self.data)
