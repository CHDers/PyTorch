# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/28 14:08
# @Author  : Yanjun Hao
# @Site    : 
# @File    : train.py
# @Software: PyCharm 
# @Comment :
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from Tommy.CIFAR10Classifier import Net


def main(epochs: int = 30):
    """
    训练主函数
    :param epochs: 训练epoch
    :return:
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    net = Net()
    # 采用交叉熵损失函数和随机梯度下降优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # data中包含输入图像张量inputs, 标签张量labels
            inputs, labels = data

            # 首先将优化器梯度归零
            optimizer.zero_grad()

            # 输入图像张量进网络, 得到输出张量outputs
            outputs = net(inputs)

            # 利用网络的输出outputs和标签labels计算损失值
            loss = criterion(outputs, labels)

            # 反向传播+参数更新, 是标准代码的标准流程
            loss.backward()
            optimizer.step()

            # 打印轮次和损失值
            running_loss += loss.item()
            if (i + 1) % 2000 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    # 首先设定模型的保存路径
    PATH = 'cifar_net.pth'
    # 保存模型的状态字典
    torch.save(net.state_dict(), PATH)


if __name__ == '__main__':
    main()
