import numpy as np
from urllib3.contrib.pyopenssl import orig_util_SSLContext

from pigeonet.basic.functions import linear
from pigeonet.basic.network import Layer, Parameter


class Linear(Layer):
    def __init__(self, out_size, in_size=None, use_bias=True):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size

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
            self.in_size = x.shape[1]
            self.__init_w__()

        return linear(x, self.w, self.b)
