from __future__ import annotations

import weakref
from typing import Optional

import numpy as np
from abc import ABC, abstractmethod

from numpy.lib.stride_tricks import broadcast_to
from torchgen.executorch.api.et_cpp import return_names

from pigeonet.basic.global_config import GlobalConfig, config


# TODO: 添加一个禁反向传播的功能
# TODO: common改名core然后打包

class Variable:
    def __init__(self, data, name=''):
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self.data: np.ndarray = data
        self.name: str = name
        self._creator: Optional[Function] = None
        self.grad: Optional[Variable | np.ndarray] = None
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
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'

        res = str(self.data).replace('\n', '\n         ')  # 空格对齐输出格式
        return f"Variable({res})"

    def backward(self, keep_grad=False, build_graph=False):
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

            with config(enable_graph_conn=build_graph):
                gxs = f.backward(*gys)

                if not isinstance(gxs, tuple):
                    gxs = gxs,

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not keep_grad:
                for y in f.outputs:
                    y().grad = None  # 清除不需要的梯度

    def clear_grad(self):
        self.grad = np.zeros_like(self.data)

    def reshape(self, shape):
        return reshape(self, shape)

    @property
    def T(self):
        return transpose(self)

    def transpose(self, dims):
        return transpose(self, dims)

    def sum(self, axis=None, keepdims=False):
        return summary(self, axis, keepdims)

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

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(self, other)

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

        if GlobalConfig.enable_graph_conn:
            # 连接新Variable到Function上
            self.generation = max([i.generation for i in args])
            for var in outputs:
                var.creator = self

        self.inputs = args  # TODO： 可以优化：不是所有的函数节点都需要保存输入
        self.outputs = [weakref.ref(output) for output in outputs]  # 函数对产生的变量是弱引用，变量对函数creator强引用

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
        self.left_shape, self.right_shape = left.shape, right.shape
        return left + right

    def backward(self, gys):
        gyl, gyr = gys, gys
        if self.left_shape != self.right_shape:
            gyl = sum_to(gyl, self.left_shape)
            gyr = sum_to(gyr, self.right_shape)

        return gyl, gyr


def add(left, right) -> Variable:
    return Add()(left, right)


class Sub(Function):
    def forward(self, left, right):
        self.left_shape, self.right_shape = left.shape, right.shape
        return left - right

    def backward(self, gys):
        gyl, gyr = gys, gys
        if self.left_shape != self.right_shape:
            gyl = sum_to(gyl, self.left_shape)
            gyr = sum_to(gyr, self.right_shape)

        return gyl, -gyr


def sub(x0, x1):
    return Sub()(x0, x1)


def rsub(x0, x1):
    return Sub()(x1, x0)


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        return self.inputs[0] * 2 * gy


def square(x):
    return Square()(x)


class Mul(Function):
    def forward(self, left, right):
        self.left_shape, self.right_shape = left.shape, right.shape
        return left * right

    def backward(self, gy):
        left, right = self.inputs
        if self.left_shape != self.right_shape:
            left = sum_to(left, self.left_shape)
            right = sum_to(right, self.right_shape)

        return gy * right, gy * left


def mul(x0, x1):
    return Mul()(x0, x1)


class Div(Function):
    def forward(self, left, right):
        self.left_shape, self.right_shape = left.shape, right.shape
        return left / right

    def backward(self, gys):
        left, right = self.inputs
        if self.left_shape != self.right_shape:
            left = sum_to(left, self.left_shape)
            right = sum_to(right, self.right_shape)

        gleft = gys / right  # gys * (1/right)
        gright = gys * -left * (right ** -2)
        return gleft, gright


def div(x0, x1):
    return Div()(x0, x1)


def rdiv(x0, x1):
    return Div()(x1, x0)


class MatMul(Function):
    def forward(self, left, right):
        return left @ right

    def backward(self, gys):
        left, right = self.inputs
        gyl = gys @ right.T
        gyr = left.T @ gys
        return gyl, gyr

def matmul(left, right):
    return MatMul()(left, right)

def rmatmul(right, left):
    return MatMul()(left, right)


class Pow(Function):
    def forward(self, x, c):
        return x ** c

    def backward(self, gys):
        x, c = self.inputs
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


class Reshape(Function):
    def __init__(self, shape):
        self.y_shape = shape

    def forward(self, x: np.ndarray):
        self.x_shape = x.shape
        y = x.reshape(self.y_shape)
        return y

    def backward(self, gys: np.ndarray | Variable):
        return reshape(gys, self.x_shape)


def reshape(x, shape):
    return Reshape(shape)(x)


class Transpose2D(Function):
    def forward(self, x):
        return x.T

    def backward(self, gys):
        return gys.T


class Transpose(Function):
    def __init__(self, dims):
        self.x_dim = dims
        self.y_dim = []

        dim = 0
        while len(self.y_dim) < len(self.x_dim):
            for i, v in enumerate(self.x_dim):
                if v == dim:
                    self.y_dim.append(dim)
                    dim += 1
                    break

    def forward(self, x: np.ndarray):
        return np.transpose(x, self.x_dim)

    def backward(self, gys):
        # TODO: 测试反向传播
        return np.transpose(gys, self.y_dim)


def transpose(x, dim=None):
    if dim is None:
        return Transpose2D()(x)
    return Transpose(dim)(x)


class Sum(Function):
    def __init__(self, axis=None, keep_dims=False):
        self.axis = axis
        self.keep_dims = keep_dims

    def forward(self, x):
        self.x_shape = x.shape
        return x.sum(axis=self.axis, keepdims=self.keep_dims)

    def backward(self, gys):
        ndim = len(self.x_shape)
        axis_tuple = self.axis
        if self.axis is None:
            axis_tuple = None
        elif not isinstance(axis_tuple, tuple):
            axis_tuple = axis_tuple,

        if ndim != 0 and axis_tuple is not None and not self.keep_dims:
            actual_axis = [a if a >= 0 else a + ndim for a in axis_tuple]
            shape = list(gys.shape)
            for a in sorted(actual_axis):
                shape.insert(a, 1)

            gys.reshape(shape)

        gys = broadcast_to(gys, self.x_shape)
        return gys


def summary(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.y_shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return np.broadcast_to(x, self.y_shape)

    def backward(self, gys):
        return sum_to(gys, self.x_shape)


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.y_shape = shape

    def forward(self, x):
        self.x_shape = x.shape

        def sum_to(x, shape):
            # 头疼
            ndim = len(shape)
            lead = x.ndim - ndim
            lead_axis = tuple(range(lead))

            axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
            y = x.sum(lead_axis + axis, keepdims=True)
            if lead > 0:
                y = y.squeeze(lead_axis)
            return y

        return sum_to(x, self.y_shape)

    def backward(self, gys):
        return broadcast_to(gys, self.x_shape)


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


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
