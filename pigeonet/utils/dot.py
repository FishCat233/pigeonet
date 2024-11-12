import os.path
import subprocess

from torch.cuda import graph

from pigeonet.basic import Variable, Function


def _dot_var(v: Variable, detail=False):
    """
    转换Variable为dot节点字符串
    :param v:
    :param detail:
    :return:
    """
    template = '{} [label="{}", color=lightskyblue, style=filled]\n'

    name = '' if v.name is None else v.name
    if detail and v.data is not None:
        # 显示详情
        # id: v.shape v.dtype / v.shape v.dtype
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return template.format(id(v), name)

def _dot_func(f: Function):
    """
    转换Function为dot节点字符串
    :param f:
    :return:
    """
    template =  '{} [label=" {}" , color=darkturquoise , style=filled ,shape=box]\n'
    connect_template = '{} -> {}\n'
    res = template.format(id(f), f.__class__.__name__)

    for x in f.inputs:
        res += connect_template.format(id(x), id(f))
    for y in f.outputs:
        res += connect_template.format(id(f), id(y()))

    return res

def dot_graph_backward(y: Variable, detail=False):
    res = ''
    funcs: list[Function] = []
    seen_set = set()

    def add_func(f: Function):
        if f not in seen_set:
            seen_set.add(f)
            funcs.append(f)

    add_func(y.creator)
    res += _dot_var(y, detail)

    while funcs:
        func = funcs.pop()
        res += _dot_func(func)
        for x in func.inputs:
            res += _dot_var(x, detail)

            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + res + '}\n'

def plot_dot_graph(y: Variable, detail=True, to_file='graph.png'):
    dot_graph_str = dot_graph_backward(y, detail)

    path = os.path.join(os.path.expanduser('.'), 'dot_graph')
    if not os.path.exists(path):
        os.mkdir(path)

    tmp_path = os.path.join(path, 'tmp.dot')
    with open(tmp_path, 'w') as f:
        f.write(dot_graph_str)

    ext = os.path.splitext(to_file)[1][1:]
    cmd = f'dot {tmp_path} -T {ext} -o {to_file}'
    subprocess.run(cmd, shell=True)
