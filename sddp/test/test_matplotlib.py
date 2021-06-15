#  Copyright 2017, Oscar Dowson, Zhao Zhipeng
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at http://mozilla.org/MPL/2.0/.
#############################################################################

import unittest

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）


class MyTestCase(unittest.TestCase):
    def test_something(self):
        x = [1, 2, 3, 4, 5]
        y1 = [1, 1, 2, 3, 5]
        y2 = [0, 4, 2, 6, 8]
        y3 = [1, 3, 5, 7, 9]
        ys = [y1, y2, y3]
        y = np.vstack([y1, y2, y3])

        labels = ["测试0 ", "测试1", "测试2"]

        fig, ax = plt.subplots()
        # ax.stackplot(x, y1, y2, y3, labels=labels)
        ax.stackplot(x, *ys, labels=labels)
        ax.legend(loc='upper left')
        plt.show()
        #
        # fig, ax = plt.subplots()
        # ax.stackplot(x, y)
        # plt.show()


if __name__ == '__main__':
    unittest.main()
