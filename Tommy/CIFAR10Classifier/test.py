# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/28 14:15
# @Author  : Yanjun Hao
# @Site    : 
# @File    : test_model.py
# @Software: PyCharm 
# @Comment :
import sys
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from Tommy.CIFAR10Classifier import imshow
from Tommy.CIFAR10Classifier import Net


def main():
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    dataiter = iter(testloader)
    images, labels = dataiter.__next__()

    # 打印原始图片
    imshow(torchvision.utils.make_grid(images))
    # 打印真实的标签
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # 首先实例化模型的类对象
    net = Net()
    # 加载训练阶段保存好的模型的状态字典
    PATH = 'cifar_net.pth'
    net.load_state_dict(torch.load(PATH))

    # 利用模型对图片进行预测
    outputs = net(images)

    # 共有10个类别, 采用模型计算出的概率最大的作为预测的类别
    _, predicted = torch.max(outputs, 1)

    # 打印预测标签的结果
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    # SECTION: -------------------------接下来看一下在全部测试集上的表现------------------------------
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    # SECTION: -----------------------------------------------------------------------------------

    # SECTION: ---------------------------看一下模型在哪些类别上表现更好-------------------------------
    # 为了更加细致的看一下模型在哪些类别上表现更好, 在哪些类别上表现更差, 我们分类别的进行准确率计算.
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    # SECTION: -----------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
