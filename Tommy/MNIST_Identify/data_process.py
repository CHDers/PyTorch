# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/12/30 9:10
# @Author  : Yanjun Hao
# @Site    : 
# @File    : ds.py
# @Software: PyCharm 
# @Comment :

# SECTION: 1.加载数据集
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt


# SECTION: 2.下载mnist数据集
trainsets = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())  # 格式转换
testsets = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

class_names = trainsets.classes  # 查看类别/标签
print(class_names)


# SECTION: 3.查看数据集的大小shape
print(trainsets.data.shape)
print(trainsets.targets.shape)
print(testsets.data.shape)
print(testsets.targets.shape)


# SECTION: 4.定义超参数
BATCH_SIZE = 32  # 每批读取的数据大小
EPOCHS = 10  # 训练10轮


# SECTION: 5.创建数据集的可迭代对象，也就是说一个batch一个batch的读取数据
train_loader = torch.utils.data.DataLoader(dataset=trainsets, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testsets, batch_size=BATCH_SIZE, shuffle=True)

# 查看一个batch的数据
images, labels = next(iter(test_loader))
print(images.shape)
print(labels.shape)


# SECTION: 6.定义函数：显示一批数据
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])  # 均值
    std = np.array([0.229, 0.224, 0.225])  # 标准差
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)  # 限速值限制在0-1之间
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# 网格显示
out = torchvision.utils.make_grid(images)
imshow(out)


# SECTION: 7.定义RNN model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # 分离隐藏状态，避免梯度爆炸
        out, hn = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :])
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # 分离隐藏状态，避免梯度爆炸
        out, hn = self.rnn(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


# SECTION: 8.初始化模型
input_dim = 28  # 输入维度
hidden_dim = 100  # 隐藏的维度
layer_dim = 2  # 2层RNN
output_dim = 10  # 输出维度

# 判断是否有GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
model.to(device=device)
# NOTE: model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)


# SECTION: 9.定义损失函数
criterion = nn.CrossEntropyLoss()


# SECTION: 10.定义优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# SECTION: 11.输出模型参数信息
length = len(list(model.parameters()))


# SECTION: 12.循环打印模型参数
for i in range(length):
    print("参数: %d" % (i + 1))
    print(list(model.parameters())[i].size())


# SECTION: 13.模型训练
sequence_dim = 28  # 序列长度
loss_list = []  # 保存loss
accuracy_list = []  # 保存accuracy
iteration_list = []  # 保存循环次数

iter = 0
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        model.train()  # 声明训练
        # 一个batch的数据转为RNN的输入维度
        images = images.view(-1, sequence_dim, input_dim).requires_grad_().to(device)  # [32, 28, 28]
        labels = labels.to(device)
        # 梯度清零(否则会不断累加)
        optimizer.zero_grad()
        # 前向传播
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 计数器自动加1
        iter += 1
        # 模型验证
        if iter % 500 == 0:
            model.eval()  # 声明
            # 计算验证的accuracy
            correct = 0.0
            total = 0.0
            # 迭代测试集，获取数据，预测
            for images, labels in test_loader:
                images = images.view(-1, sequence_dim, input_dim).requires_grad_().to(device)
                # 模型预测
                outputs = model(images)
                # 获取预测概率最大值的下标
                predict = torch.max(outputs.data, 1)[1]
                # 统计测试集的大小
                total += labels.size()[0]
                # 统计判断/预测正确的数量
                if torch.cuda.is_available():
                    correct += (predict.gpu() == labels.gpu()).sum()
                else:
                    correct += (predict == labels).sum()
            # 计算accuracy
            accuracy = correct / total * 100
            # 保存accuracy, loss, iteration
            loss_list.append(loss.data)
            accuracy_list.append(accuracy)
            iteration_list.append(iter)
            # 打印信息
            print("loop: {}, Loss: {}, Accuracy: {}".format(iter, loss.item(), accuracy))


# SECTION: 14.可视化loss
plt.figure(figsize=(10, 8))
plt.plot(iteration_list, loss_list)
plt.xlabel("iteration")
plt.ylabel("Loss")
plt.title("RNN")
plt.savefig('./run/rnn Loss.png')
plt.show()


# SECTION: 14.可视化accuracy
plt.figure(figsize=(10, 8))
plt.plot(iteration_list, accuracy_list, color='r')
plt.xlabel("iteration")
plt.ylabel("Accuracy")
plt.title("RNN")
plt.savefig('./run/rnn Accuracy.png')
plt.show()
