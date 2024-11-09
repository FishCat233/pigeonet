from __future__ import annotations

import weakref

import numpy
import numpy as np
from abc import ABC, abstractmethod

from pigeonet.basic.variable import Variable

__all__ = [name for name in globals() if not name.startswith('_')]

#TODO: 添加一个禁反向传播的功能


class Function(ABC):
    def __call__(self, *args: Variable, **kwargs):
        xs = [x.data for x in args]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([i.generation for i in args])
        for var in outputs:
            var.creator = self
        self.inputs = args
        self.outputs = [weakref.ref(output) for output in outputs]  # 函数对产生的变量是弱引用，变量对函数使用creator引用

        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self, *xs: Variable) -> Variable | tuple[Variable]:
        raise NotImplementedError

    @abstractmethod
    def backward(self, gys) -> tuple[Variable]:
        raise NotImplementedError


class Add(Function):
    def forward(self, x0, x1) -> Variable | tuple[Variable]:
        return x0 + x1

    def backward(self, gys: Variable) -> tuple[Variable, Variable]:
        return gys, gys


def add(x0: Variable, x1: Variable) -> Variable:
    return Add()(x0, x1)


class Square(Function):
    def forward(self, x) -> Variable:
        y = x ** 2
        return y

    def backward(self, gy) -> tuple[Variable]:
        return self.inputs[0].data * 2 * gy,


def square(x: Variable) -> Variable:
    return Square()(x)


def as_array(x) -> numpy.ndarray:
    """
    将标量/非标量自动转化为numpy数组
    :param x:
    :return:
    """
    if np.isscalar(x):
        return np.array(x)

    return x


if __name__ == '__main__':
    from pigeonet.basic.variable import Variable

    x0 = Variable(2)
    x1 = Variable(3)
    x2 = add(x0, x1)
    y = square(x2)
    y.backward()

    print(x0.data)
    print(x0.grad)
    print(x1.data)
    print(x1.grad)
    print(x2.data)
    print(x2.grad)
    print(y.data)
    print(y.grad)
