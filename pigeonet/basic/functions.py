import numpy as np

from pigeonet.basic import Function, Variable, as_variable, summary, sum_to


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
