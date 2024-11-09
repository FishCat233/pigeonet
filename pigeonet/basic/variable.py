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
        self.generation: int = 0 # 代数，用于标记函数反向传播先后顺序

    @property
    def creator(self) -> Optional[Function]:
        return self._creator

    @creator.setter
    def creator(self, func: Optional[Function]):
        self._creator = func
        self.generation = func.generation + 1

    def backward(self):
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
            gys = [y.grad for y in f.outputs]
            gxs = f.backward(*gys)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad += gx

                if x.creator is not None:
                    add_func(x.creator)

    def clear_grad(self):
        self.grad = np.ones_like(self.data)
