#  Copyright 2017, Oscar Dowson, Zhao Zhipeng
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

from typing import Dict

from sddp.typedefinitions import *
import numpy as np


def sample(x: List[float]):
    dim = len(np.shape(x))
    if dim == 0:
        return 0
    res = np.random.choice(len(x), p=x)
    return int(res)
