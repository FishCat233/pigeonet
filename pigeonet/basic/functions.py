import numpy as np

import pigeonet.basic
from pigeonet.basic import Function, Variable, as_variable, summary, sum_to, GlobalConfig


class Exp(Function):
    def forward(self, x):
        self.y = np.exp(x)
        return self.y

    def backward(self, gys):
        return gys * self.y


def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        return np.log(x)

    def backward(self, gys):
        x, = self.inputs
        return gys * (1 / x.data)


def log(x):
    return Log()(x)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        return np.clip(x, self.x_min, self.x_max)

    def backward(self, gys):
        # TODO: 反向传播
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gys * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


class Linear(Function):
    def forward(self, x, w, b):
        y = x @ w
        if b is not None:
            y += b
        return y

    def backward(self, gys):
        x, w, b = self.inputs
        gb = None if b.data is None else sum_to(gys, b.shape)
        gw = x.T @ gys
        gx = gys @ w.T
        return gx, gw, gb


def linear(x, w, b=None):
    return Linear()(x, w, b)


class BatchNorm(Function):
    def __init__(self, momentum=0.9, running_mean=None, running_var=None):
        self.momentum = momentum
        self.running_mean = running_mean
        self.running_var = running_var

    def forward(self, x: np.ndarray, gamma, beta):
        self.x_shape = x.shape
        if x.ndim != 2:
            # conv
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        self.N = x.shape[0]

        if self.running_mean is None or self.running_var is None:
            N, D = x.shape
            running_mean = np.zeros(D)
            running_var = np.zeros(D)

        if pigeonet.basic.GlobalConfig.eval_mode:
            # 验证模式
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # 训练模式
            xc = x - self.running_mean
            std = np.sqrt(self.running_var + 10e-7)
            xn = xc / std

        self.xc = xc
        self.std = std
        self.xn = xn
        y = gamma * xn + beta
        return y.reshape(*self.x_shape)

    def backward(self, gys: np.ndarray):
        x, gamma, beta = [i.data for i in self.inputs]

        if gys.ndim != 2:
            N, C, H, W = gys.shape
            gys = gys.reshape(N, -1)

        dbeta = gys.sum(axis=0)
        dgamma = np.sum(self.xn * gys, axis=0)
        dxn = gamma * gys
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.N) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.N

        return dx, dgamma, dbeta


def batch_norm(x, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
    return BatchNorm(momentum, running_mean, running_var)(x, gamma, beta)


class ReLU(Function):
    def forward(self, x):
        self.mask = x > 0
        return x * (self.mask)

    def backward(self, gys):
        x, = self.inputs
        mask = x.data > 0
        return gys * mask


def relu(x):
    return ReLU()(x)


class Softmax(Function):
    """
    略
    """

    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        y = x - x.max(axis=self.axis, keepdims=True)  # 防溢出
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gys):
        # TODO: 俺不会啊
        y = self.outputs[0]()
        gx = gys * y
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)


def softmax_simple(x, axis=1):
    """
    使用框架自动求导
    :param x:
    :param axis:
    :return:
    """
    x = as_variable(x)
    y = exp(x)
    sum_y = summary(y, axis=axis, keepdims=True)
    return y / sum_y


# TODO: softmax with cross entropy functino
# class SoftmaxWithCrossEntropy(Function):
#     def forward(self, x, t):
#         n = x.shape[0]

def softmax_with_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    n = x.shape[0]

    p = softmax(x)
    p = clip(p, 1e-15, 1.0)
    log_p = log(p)
    tlog_p = log_p(np.arange(n), t.data)
    y = -1 * summary(tlog_p) / n
    return y
