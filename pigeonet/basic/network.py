import weakref
from pigeonet.basic.core import Variable

class Parameter(Variable):
    pass

class Layer:
    def __init__(self):
        self._params_name = set()

    def __setattr__(self, key, value):
        if isinstance(value, Parameter):
            self._params_name.add(key)
        super().__setattr__(key, value)

    def forward(self, x):
        raise NotImplementedError()

    def params(self) -> tuple[Parameter]:
        """
        返回层所有参数的generator
        :return: generator
        """
        for name in self._params_name:
            yield self.__dict__[name]

    def clear_grads(self) -> None:
        for param in self.params():
            param.clear_grad()