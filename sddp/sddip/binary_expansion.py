# @version : python3.5
# @Time    : 2018/7/16 19:38
# @Author  : zzp
# @FileName: binary_expansion.py
"""
将整数或者浮点数转进行二进制展开
"""
import math
import sys

import numpy as np
from typing import List

log2inv = 1 / math.log(2)
_2i_ = [2 ** i for i in range(math.floor(math.log(sys.maxsize) * log2inv))]
_2i_L = len(_2i_)


def binexpand_(y: List[int], x: int):
    if x < 0:
        raise RuntimeError("Values to be expanded must be nonnegative. Currently x = %d" % x)
    for i in reversed(range(len(y))):
        k = _2i_[i]
        if x >= k:
            y[i] = 1
            x -= k
    if x > 0:
        raise RuntimeError("Unable to expand binary. Overflow of  %d" % x)


def bitsrequired(x: int or float, eps=0.1):
    if type(x) == float:
        x = np.round(x / eps)
    return math.floor(math.log(x) / log2inv) + 1


def binexpand_int(x: int, length: int = -1, maximun: float = -1):
    if x < 0:
        raise RuntimeError("Cannot perform binary expansion on a negative number.")
    if maximun != -1:
        length = bitsrequired(math.floor(maximun))
    if length == -1:
        y = [0] * bitsrequired(x)
    else:
        y = [0] * length
    binexpand_(y, x)
    return y


def binexpand_float(x: float, eps: float = 0.1, length: int = -1, maximun: float = -1):
    if x < 0:
        raise RuntimeError("Cannot perform binary expansion on a negative number.")
    if eps <= 0:
        raise RuntimeError("Epsilon tolerance for Float binary expansion must be strictly greater than 0.")
    xx = np.round(x / eps)
    if maximun != -1:
        length = bitsrequired(math.floor(maximun / eps))
    binexpand_(xx, length=length)
    return xx


def bincontract_2i_(y: List):
    x = 0
    for i in range(len(y)):
        x += _2i_[i] * y[i]
    return x


def bincontract_pow(y: List):
    x = 0
    for i in range(len(y)):
        x += 2 ** i * y[i]
    return x


def bincontract(y: List):
    if len(y) < _2i_L:
        return bincontract_2i_(y)
    else:
        return bincontract_pow(y)


def bincontract_float(y: List, eps: float = 0.1):
    if eps <= 0:
        raise RuntimeError("Epsilon tolerance for Float binary contraction must be strictly greater than 0.")
    return binexpand_(y) * eps

