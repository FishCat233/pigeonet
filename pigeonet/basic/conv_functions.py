import numpy as np
from cv2.gapi import kernel

from pigeonet.basic import Function


class Conv(Function):
    def __init__(self, stride, pad):
        super().__init__()
        self.stride = stride
        self.pad = pad

    def forward(self, x: np.ndarray, w: np.ndarray, b: np.ndarray):
        FN, C, FH, FW = w.shape
        N, C, H, W = x.shape
        P = self.pad
        OH = int((H + 2 * P - FH) / self.stride) + 1
        OW = int((W + 2 * P - FW) / self.stride) + 1

        col_x = im2col(x, FH, FW, self.stride, self.pad)
        col_w = w.reshape(FN, -1).T
        out = col_x @ col_w + b
        out = out.reshape(N, OH, OW, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col_x = col_x
        self.col_w = col_w
        self.out = out

        return out

    def backward(self, gys):
        if gys.ndim <= 2:
            gys = gys.reshape(self.out.shape)
        FN, C, FH, FW = self.w.shape
        gys = gys.transpose(0, 1, 3, 2).reshape(-1, FN)
        dx = gys @ self.col_w.T
        dx = col2im(dx, self.x.shape, FH, FW, stride=self.stride, pad=self.pad)
        dw = self.col_x.T @ gys

        dw = dw.transpose(1, 0).reshape(FN, C, FH, FW)
        db = np.sum(gys, axis=0)
        return dx, dw, db


def conv(x, w, b, stride, pad):
    return Conv(stride=stride, pad=pad)(x, w, b)


class MaxPool(Function):
    def __init__(self, stride, pad):
        super().__init__()
        self.stride = stride
        self.pad = pad

    def forward(self, x, kernel_size):
        self.kernel_size = kernel_size

        N, C, H, W = x.shape
        OH = 1 + int((H - kernel_size) / self.stride)
        OW = 1 + int((W - kernel_size) / self.stride)

        col_x = im2col(x, kernel_size, kernel_size, stride=self.stride, pad=self.pad)
        col_x = col_x.reshape(-1, kernel_size ** 2)

        self.arg_max: np.ndarray = np.argmax(col_x, axis=1)
        out: np.ndarray = np.max(col_x, axis=1)
        out = out.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)

        return out

    def backward(self, gys: np.ndarray):
        x = self.inputs[0].data

        gys = gys.transpose(0, 2, 3, 1)

        pool_size = self.kernel_size ** 2

        dmax = np.zeros((gys.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = gys.flatten()

        dmax = dmax.reshape(gys.shape + (pool_size,))
        dcol_x = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)

        dx = col2im(dcol_x, x.shape, self.kernel_size, self.kernel_size, self.stride, self.pad)
        return dx


def max_pool(x, kernel_size: int, stride=1, pad=0):
    # TODO: 测试池化
    return MaxPool(stride, pad)(x, kernel_size)


def im2col(input_data: np.ndarray, filter_h, filter_w, stride=1, pad=0) -> np.ndarray:
    """
    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0) -> np.ndarray:
    """
    Parameters
    ----------
    col :
    input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
