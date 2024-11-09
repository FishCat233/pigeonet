from pigeonet.basic.variable import *
from pigeonet.basic.function import *

if __name__ == '__main__':
    x0 = Variable(2)
    x1 = Variable(3)
    x2 = add(x0, x1)
    y = square(x2)
    y.backward()
    print(y)
    print(y.grad)
    print(x2.grad)
    print(x1.grad)
    print(x0.grad)
