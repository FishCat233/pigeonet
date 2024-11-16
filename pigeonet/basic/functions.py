from pigeonet.basic import Function, Variable, as_variable


def linear(x, w, b=None):
    t = x @ w
    if b is None:
        return t
    y = t + b
    t.data = None  # 因为 y = t + b， t 的 data 反向传播用不到，所以可以清空了
    return y


def relu(x):
    if x > 0:
        return x
    else:
        return 0
