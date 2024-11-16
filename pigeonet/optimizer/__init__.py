from __future__ import annotations

import numpy as np

from pigeonet.basic.network import Optimizer, Parameter


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param: Parameter) -> None:
        param.data -= self.lr * param.grad.data


class Momentum(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.v_s: dict[int, np.ndarray]= {}

    def update_one(self, param: Parameter) -> None:
        key = id(param)
        if key not in self.v_s:
            self.v_s[key] = np.zeros_like(param.data)

        v = self.v_s[key]
        v = v * self.momentum - self.lr * param.grad.data
        param.data += v
