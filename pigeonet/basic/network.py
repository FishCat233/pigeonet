from __future__ import annotations

import weakref
from abc import ABC, abstractmethod

from Demos.win32cred_demo import target

from pigeonet.basic.core import Variable
from pigeonet.utils.dot import plot_dot_graph


class Parameter(Variable):
    pass


class Layer(ABC):
    def __init__(self):
        self._params_name = set()

    def __setattr__(self, key, value):
        if isinstance(value, (Parameter, Layer)):
            self._params_name.add(key)
        super().__setattr__(key, value)

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError()

    __call__ = forward

    def params(self) -> tuple[Parameter]:
        """
        返回层所有参数的generator 以及子层数的所有参数generator
        :return: generator
        """
        for name in self._params_name:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                yield obj.params()
            else:
                yield obj

    def clear_grads(self) -> None:
        for param in self.params():
            param.clear_grad()


class Model(Layer):
    def dot_plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return plot_dot_graph(y, True, to_file)


class Optimizer(ABC):
    """
    优化器抽象类
    """

    def __init__(self):
        self.target = None

    def setup(self, target: Layer):
        """
        设置目标
        :return: self 链式调用
        """
        self.target = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]  # 梯度为空的参数就不更新 l

        for p in params:
            self.update_one(p)

    @abstractmethod
    def update_one(self, param: Parameter) -> None:
        raise NotImplementedError
