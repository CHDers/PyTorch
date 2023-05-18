# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/27 9:40
# @Author  : Yanjun Hao
# @Site    : 
# @File    : ds.py
# @Software: PyCharm 
# @Comment :

# 导入画图包和numpy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


# 构建展示图片的函数
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 从数据迭代器中读取一张图片
    dataiter = iter(trainloader)
    images, labels = dataiter.__next__()
    print(labels)

    # 展示图片
    imshow(torchvision.utils.make_grid(images))
    # 打印标签label
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


if __name__ == '__main__':
    main()
