from typing import Optional, Sequence

import numpy as np

from pigeonet.basic.functions import linear, relu
from pigeonet.basic.conv_functions import conv
from pigeonet.basic.network import Layer, Parameter


class Linear(Layer):
    """
    全连接 / 线性变换 / 仿射变换 层
    """

    def __init__(self, out_size: int, in_size: Optional[int] = None, use_bias=True):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.w = None
        if in_size is not None:
            self.__init_w__()
        if not use_bias:
            self.b = None
        self.b = Parameter(np.zeros(self.out_size), name='b')

    def __init_w__(self):
        w_data = np.random.randn(self.in_size, self.out_size) * np.sqrt(1 / self.in_size)
        self.w = Parameter(w_data, name='w')

    def forward(self, x):
        if self.w is None:
            self.in_size = x.shape[1]  # (batch, in_size)
            self.__init_w__()

        return linear(x, self.w, self.b)


class MLP(Layer):
    """
    多层感知机层
    """

    def __init__(self, out_sizes, activation=relu):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, v in enumerate(out_sizes):
            layer = Linear(v)
            setattr(self, f'linear_{i}', layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class Convolution(Layer):
    """
    2d 卷积层
    """

    # TODO: 测试卷积层
    def __init__(self, out_channels, kernel_size, stride=1, pad=0, use_bias=True, in_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.w = Parameter(None, 'w')

        if self.in_channels is not None:
            self._init_w()

        if not use_bias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels), 'b')

    def _init_w(self):
        C, OC = self.in_channels, self.out_channels
        scale = np.sqrt(1 / (C * self.kernel_size ** 2))
        self.w.data = np.random.randn(OC, C, self.kernel_size, self.kernel_size) * scale

    def forward(self, x):
        if self.w.data is None:
            self.in_channels = x.shape[1]  # (N, C, H, W)
            self._init_w()

        return conv(x, self.w, self.b, self.stride, self.pad)

class Sequential(Layer):
    """
    序列层
    """
    def __init__(self, *layers: Layer):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x