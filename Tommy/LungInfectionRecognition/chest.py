# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/1/5 17:11
# @Author  : Yanjun Hao
# @Site    : 
# @File    : chest.py
# @Software: PyCharm 
# @Comment :
# 数据集来源：https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/download

# SECTION 1 加载库
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np


# SECTION 2 定义一个方法，显示图片
def image_show(inp, title=None):
    plt.figure(figsize=(14, 3))
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()


# SECTION 7 迁移学习：拿到一个成熟的模型，进行模型微调
def get_model():
    model_pre = models.resnet50(pretrained=True)  # 获取预训练模型
    # 冻结预训练模型中所有的参数
    for param in model_pre.parameters():
        param.requires_grad = False
        # 微调模型，替换ResNet最后两层网络，返回一个新的模型
        model_pre.avgpool = AdaptiveConcatPool2()  # 池化层替换
        model_pre.fc = nn.Sequential(
            nn.Flatten(),  # 所有维度拉平
            nn.BatchNorm1d(4096),  # 256*6*6 --> 4096
            nn.Dropout(0.5),  # 丢掉一些神经元
            nn.Linear(4096, 512),  # 线性层的处理
            nn.ReLU(),  # 激活层
            nn.BatchNorm1d(512),  # 正则化处理
            nn.Linear(512, 2),
            nn.LogSoftmax(dim=1),  # 损失函数
        )

    return model_pre


# SECTION 8 更改池化层
class AdaptiveConcatPool2(nn.Module):
    def __init__(self, size=None):
        super().__init__()
        size = size or (1, 1)  # 池化层的卷积核大小，默认为 (1, 1)
        self.pool_one = nn.AdaptiveAvgPool2d(size)  # 池化层1
        self.pool_two = nn.AdaptiveAvgPool2d(size)  # 池化层2

    def forward(self, x):
        return torch.cat([self.pool_one(x), self.pool_two(x)])  # 连接两个池化层


# SECTION 9 定义训练函数
def train(model, device, train_loader, criterion, optimizer, epoch, writer):
    model.train()
    total_loss = 0.0  # 总损失值初始化为0
    for batch_index, (data, target) in enumerate(train_loader):
        # 部署到DEVICE上去
        data, target = data.to(device), target.to(device)
        # 梯度初始化为0
        optimizer.zero_grad()
        # 训练后的结果
        output = model(data)
        # 计算损失
        loss = criterion(output, target)
        # 找到概率值最大的下标
        pred = output.max(1, keepdim=True)  # pred = output.argmax(dim=1)
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        total_loss += loss  # 累计训练损失

    writer.add_scaler("Train Loss", total_loss / len(train_loader), epoch)
    writer.flush()  # 刷新
    return total_loss / len(train_loader)  # 返回平均损失值


# SECTION 10 定义测试函数
def test(model, device, test_loader, criterion, epoch, writer):
    # 模型验证
    model.eval()
    # 正确率
    correct = 0.0
    # 测试损失
    total_loss = 0.0
    with torch.no_grad():  # 不会计算梯度，也不会进行反向传播
        for data, target in test_loader:
            # 部署到DEVICE上去
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            total_loss += criterion(output, target).item()
            # 找到概率值最大的下标
            _, preds = torch.max(output, dim=1)
            # 累计正确率
            correct += torch.sum(preds == target)
        total_loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        writer.add_scaler("Test Loss", total_loss, epoch)
        writer.add_scaler("Accuracy", accuracy, epoch)
        writer.flush()  # 刷新
        print("TEST -- Average loss: {:.4f}, Accuracy: {:.4f}\n".format(total_loss, accuracy))


def main():
    # SECTION 3 定义超参数
    BATCH_SIZE = 64  # 每批处理的数据数量
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 是使用GPU还是CPU训练
    # SECTION 4 图片转换
    data_transforms = {
        'train':
            transforms.Compose([
                transforms.Resize(300),
                transforms.RandomResizedCrop(300),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
        'val':
            transforms.Compose([
                transforms.Resize(300),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),
    }
    # SECTION 5 操作数据集
    # SECTION 5.1 数据集路径
    data_path = "./chest_xray"
    # SECTION 5.2 加载数据集train和val
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x),
                                              data_transforms[x]) for x in ['train', 'val']}
    # SECTION 5.3 为数据集创建一个迭代器，读取数据
    dataloaders = {x: DataLoader(image_datasets[x], shuffle=True, batch_size=BATCH_SIZE)
                   for x in ['train', 'val']}
    # SECTION 5.4 训练集和验证集的大小(图片的数量)
    data_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # SECTION 5.5 获取标签的类别名称     NORMAL 正常    PNEUMONIA 感染
    target_names = image_datasets['train'].classes

    # SECTION 6 显示一个batch_size的图片(8张图片)
    # SECTION 6.1 读取8张图片
    datas, targets = next(iter(dataloaders['train']))
    # SECTION 6.2 将若干图片拼成一张图片
    out = make_grid(datas, nrow=4, padding=10)
    # SECTION 6.3 显示图片
    image_show(out, title=[target_names[x] for x in targets])


if __name__ == '__main__':
    main()
    print("Running Done!")
