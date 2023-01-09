# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/28 13:58
# @Author  : Yanjun Hao
# @Site    : 
# @File    : model.py.py
# @Software: PyCharm 
# @Comment :
# 视频链接: https://www.bilibili.com/video/BV1Ta411w78P/?spm_id_from=pageDriver&vd_source=40437a7834b5b148effaa5971e14f8d6

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    net = Net()
    print(net)
