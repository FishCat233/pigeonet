import unittest

import numpy as np
from torch import no_grad

from pigeonet.basic import Variable
from pigeonet.basic.numerical_diff import numerical_diff


class GradientTest(unittest.TestCase):
    """
    梯度测试
    """

    def test_basic_calc(self):
        x = Variable(np.random.rand(5,5,5))

        def f(x):
            # x += 1
            # x *= 5
            # x -= 2
            x /= 10
            return x

        y = f(x)
        y.backward()
        with no_grad():
            numerical_grad = numerical_diff(f, x)
            out = x.grad.data - numerical_grad
        # print(x.data)
        # print(x.grad.data)
        # print(numerical_grad)
        print(out[out > 1e-10])
        self.assertEqual(0, len(out[out > 1e-10]))
