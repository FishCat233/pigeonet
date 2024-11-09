from __future__ import annotations

import weakref

import numpy
import numpy as np
from abc import ABC, abstractmethod

from pigeonet.basic.variable import Variable

__all__ = [name for name in globals() if not name.startswith('_')]


# TODO: 添加一个禁反向传播的功能


class Function(ABC):
    """
    函数节点类
    每一个Function对应计算图上一个函数节点。输入变量，然后产出变量，通过self.inputs和产出变量的creator属性来连接，并且拥有generation来记录函数节点在反向传播中的顺序
    """

    def __call__(self, *args, **kwargs):
        """
        Variable 前向传播函数节点
        :param args:
        :param kwargs:
        :return:
        """
        args = [as_variable(arg) for arg in args]
        xs = [x.data for x in args]  # x Variable中的data
        ys = self.forward(*xs)  # y Variable中的data

        if not isinstance(ys, tuple):
            ys = (ys,)  # 下面默认是操作元组，所以要转元组
        outputs = [Variable(y) for y in ys]

        self.generation = max([i.generation for i in args])
        for var in outputs:
            var.creator = self
        self.inputs = args
        self.outputs = [weakref.ref(output) for output in outputs]  # 函数对产生的变量是弱引用，变量对函数使用creator引用

        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self, *xs):
        """
        前向传播计算
        :param xs: x.data
        :return: y.data
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, gys):
        """
        后向传播计算
        :param gys: grad ys
        :return: grad xs
        """
        raise NotImplementedError


class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, gys):
        return gys, gys


def add(left, right) -> Variable:
    return Add()(left, as_variable(right))


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        return self.inputs[0].data * 2 * gy,


def square(x) -> Variable:
    return Square()(x)


class Mul(Function):
    def forward(self, left, right):
        return left * as_variable(right)

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0, x1):
    return Mul()(x0, x1)


def as_variable(x):
    """
    自动转换为 Variable
    :param x:
    :return:
    """
    if isinstance(x, Variable):
        return x
    return Variable(x)


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
