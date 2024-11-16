import numpy as np

from pigeonet.basic import Function, Variable, as_variable, summary, exp


def linear(x, w, b=None):
    t = x @ w
    if b is None:
        return t
    y = t + b
    t.data = None  # 因为 y = t + b， t 的 data 反向传播用不到，所以可以清空了
    return y


def relu(x):
    return (x > 0) * x


def batch_softmax(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = summary(y, axis=axis, keepdims=True)
    return y / sum_y


def batch_softmax_with_cross_entropy(x, t):
    x, t = as_variable(x), as_variable(t)
    n = x.shape[0]

    p = batch_softmax(x)
    p = np.clip(p, 1e-15, 1.0)
    log_p = np.log(p)
