# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/1/3 13:36
# @Author  : Yanjun Hao
# @Site    : 
# @File    : python绘图宋体+新罗马.py
# @Software: PyCharm 
# @Comment :


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

config = {
    "font.family": 'serif',
    "font.size": 18,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
    'axes.unicode_minus': False  # 处理负号，即-号
}
rcParams.update(config)

x = np.random.random((10,))

plt.plot(x, label='随机数')
plt.title('中文：宋体 \n 英文：$\mathrm{Times \; New \; Roman}$ \n 公式： $\\alpha_i + \\beta_i = \\gamma^k$')
plt.xlabel('横坐标02468')
plt.ylabel('纵坐标')
plt.legend()
plt.yticks(fontproperties='Times New Roman', size=18)
plt.xticks(fontproperties='Times New Roman', size=18)
plt.show()
