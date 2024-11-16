import contextlib

from fontTools.misc.cython import returns


class GlobalConfig:
    # 启用计算图自动连接
    enable_graph_conn = True
    # 测试模式
    eval_mode = False


@contextlib.contextmanager
def config(**kwargs):
    old_value = {}
    for k, v in kwargs.items():
        old_value[k] = getattr(GlobalConfig, k)
        setattr(GlobalConfig, k, v)
    try:
        yield
    finally:
        for k,v in old_value.items():
            setattr(GlobalConfig, k, v)

def no_graph_conn():
    return config(enable_graph_conn=False)

def eval_mode():
    return config(eval_mode=True)