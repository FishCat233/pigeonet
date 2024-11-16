import unittest
import numpy as np

from pigeonet.basic.core import Variable, Function, square, add
from pigeonet.basic.numerical_diff import numerical_diff
from pigeonet.utils.dot import dot_graph_backward, plot_dot_graph
from pigeonet.basic.functions import relu


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

    def test_complicate_function_grad(self):
        # TODO：复杂函数求导
        pass

    def test_higher_derivative(self):
        # 高阶求导测试
        def f(x: Variable):
            # 4 * x ** 3 - 4 * x
            y = x ** 4 - 2 * (x ** 2)
            return y

        x = Variable(2, name="x")
        y = f(x)
        y.name = "y"
        y.backward(build_graph=True)
        print(x.grad)

        gx = x.grad
        x.clear_grad()
        gx.name = 'gx'
        gx.backward()
        # plot_dot_graph(y, detail=True)
        print(x.grad)

    def test_summary_function(self):
        x = Variable([[1, 2, 3], [4, 5, 6]])
        y = x.sum(axis=0)
        y.backward()
        print(y)
        print(x.grad)

        x = Variable(np.random.randn(2, 3, 4, 5))
        y = x.sum(keepdims=True)
        y.backward()
        print(y)
        print(x.grad)

    def test_matmul(self):
        x0 = Variable([[1, 2], [3, 4]])
        x1 = Variable([[5, 6], [7, 8]])
        print(x0 @ x1)

    def test_relu_backward(self):
        x = Variable([1,0,-1])
        y = relu(x)
        y.backward()
        print(y)
        print(x.grad)