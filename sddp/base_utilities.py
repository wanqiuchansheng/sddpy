# @version : python3.5
# @Time    : 2018/7/1 21:18
# @Author  : zzp
# @FileName: utilities.py
from typing import Dict

from sddp.typedefinitions import *
import numpy as np


def sample(x: List[float]):
    dim = len(np.shape(x))
    if dim == 0:
        return 0
    res = np.random.choice(len(x), p=x)
    return int(res)
