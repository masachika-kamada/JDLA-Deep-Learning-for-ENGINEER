import numpy as np


def convolution_2d(x, W, b, sh, sw, ph, pw):
    """
    :param x: inputs, (batch size, channel, height, width)
    :param W: convolution weights, (output channel, input channel, kernel height, kernel width)
    :param b: bias, (output channel,)
    :param sh: stride
    :param sw: stride
    :param ph: padding
    :param pw: padding
    :return:
    """
    # im2colによって画像を変換
    out_c, c, kh, kw = W.shape
    col = im2col(x, kh, kw, sh, sw, ph, pw)
    n, c, kh, kw, out_h, out_w = col.shape

    # 行列演算できる形に変形
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n, out_h, out_w, -1)
    # --> col.shape = n, out_h, out_w, (c * kh * kw)
    # W: (output channel, input channel, kernel height, kernel width)
    W = W.reshape(out_c, -1).T
    # --> W.shape = ((c * kh * kw), out_c)
    y = np.dot(col, W) + b
    # --> y.shape = n, out_h, out_w, out_c
    return y.transpose(0, 3, 2, 1)
    # --> y.shape = n, out_c, out_h, out_w
