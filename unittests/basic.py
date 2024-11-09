import unittest
import numpy as np

from pigeonet.basic.common import Variable, Function, square, add
from pigeonet.basic.numerical_diff import numerical_diff


class SquareTest(unittest.TestCase):
    def hello_test(self):
        x = Variable(2)
        y = square(x)
        expected = Variable(4)
        self.assertEqual(y.data, expected.data)

    def test_backward(self):
        x = Variable(3.0)
        b = Variable(9)
        y0 = add(x, b)
        y = square(y0)
        y.backward()
        expected_x = Variable(24)
        self.assertEqual(x.grad, expected_x.data)
        self.assertEqual(b.grad, expected_x.data)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        numerical_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, numerical_grad)
        self.assertEqual(True, flg)

    def test_x_reuse(self):
        x = Variable(2)
        a = square(x)
        y = add(square(a), square(a))

        y.backward()
        self.assertEqual(y.data, Variable(32).data)
        self.assertEqual(x.grad, 64)

    def test_overload_operator(self):
        x = Variable(2)
        a = x + 2
        b = 2 + a
        c = b * 2
        d = 2 * c

        self.assertEqual(d.data, Variable(24).data)

        a = x + np.array(2)
        b = np.array(2) + a
        c = b * np.array(2)
        d = np.array(2) * c

        self.assertEqual(d.data, Variable(24).data)
