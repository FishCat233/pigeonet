from __future__ import annotations

import weakref
from mailcap import subst
from typing import Optional

import numpy as np
from abc import ABC, abstractmethod

from numpy.core.numeric import isscalar


# TODO: 添加一个禁反向传播的功能

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

            if not isinstance(gxs, tuple):
                gxs = gxs,

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

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(self, other)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(self, other)

    def __neg__(self):
        return neg(self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return rsub(self, other)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return rdiv(self, other)

    def __pow__(self, power, modulo=None):
        return pow(self, power)


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
        args = [as_variable(arg) for arg in args]  # TODO：可以优化，没必要所有的输入都转化成Variable，因为不是所有的数据都要梯度
        xs = [x.data for x in args]  # x Variable中的data
        ys = self.forward(*xs)  # y Variable中的data

        if not isinstance(ys, tuple):
            ys = (ys,)  # 下面默认是操作元组，所以要转元组
        outputs = [Variable(y) for y in ys]

        self.generation = max([i.generation for i in args])
        for var in outputs:
            var.creator = self
        self.inputs = args  # TODO： 可以优化：不是所有的函数节点都需要保存输入
        self.outputs = [weakref.ref(output) for output in outputs]  # 函数对产生的变量是弱引用，变量对函数使用creator引用

        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self, *xs):
        """
        前向传播计算，围绕节点数据进行运算。
        :param xs: x.data
        :return: y.data
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, gys):
        """
        后向传播计算，围绕节点数据进行求导。
        :param gys: grad ys
        :return: grad xs
        """
        raise NotImplementedError


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gys):
        return -gys


def neg(x):
    return Neg()(x)


class Add(Function):
    def forward(self, left, right):
        return left + right

    def backward(self, gys):
        return gys, gys


def add(left, right) -> Variable:
    return Add()(left, right)


class Sub(Function):
    def forward(self, left, right):
        return left - right

    def backward(self, gys):
        return gys, -gys


def sub(x0, x1):
    return Sub()(x0, x1)


def rsub(x0, x1):
    return Sub()(x1, x0)


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        return self.inputs[0].data * 2 * gy


def square(x):
    return Square()(x)


class Mul(Function):
    def forward(self, left, right):
        return left * right

    def backward(self, gy):
        left, right = [x.data for x in self.inputs]
        return gy * right, gy * left


def mul(x0, x1):
    return Mul()(x0, x1)


class Div(Function):
    def forward(self, left, right):
        return left / right

    def backward(self, gys):
        left, right = [x.data for x in self.inputs]
        gleft = gys / right  # gys * (1/right)
        gright = gys * -left * (right ** -2)
        return gleft, gright


def div(x0, x1):
    return Div()(x0, x1)


def rdiv(x0, x1):
    return Div()(x1, x0)


class Pow(Function):
    def forward(self, x, c):
        return x ** c

    def backward(self, gys):
        x, c = self.inputs[0].data, self.inputs[1]
        gx = gys * c * (x ** (c - 1))
        # WARNING: 布计算c的导数
        return gx


def pow(x, c):
    """
    乘方，注意返回求导时不计算c的导数
    :param x:
    :param c:
    :return:
    """
    return Pow()(x, c)


def as_variable(x):
    """
    自动转换为 Variable
    :param x:
    :return:
    """
    if isinstance(x, Variable):
        return x
    return Variable(x)


def as_ndarray(x):
    """
    自动转换为array
    :param x:
    :return:
    """
    if np.isscalar(x):
        return x
    return np.array(x)
