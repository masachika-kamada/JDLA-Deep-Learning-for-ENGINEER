import numpy as np


def softmax(x):
    if x.dim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.dim == 1:
        x = np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x


class Softmax:
    def __init__(self):
        self.param, self.grad = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx
