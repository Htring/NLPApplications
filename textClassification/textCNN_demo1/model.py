#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: model.py
@time:2020/12/06
@description:
"""
import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import config
import dataloader
import utils


class TextCNN(nn.Module):
    # output_size为输出类别（2个类别，0和1）,三种kernel，size分别是3,4，5，每种kernel有100个
    def __init__(self, vocab_size, embedding_dim, output_size, filter_num=100, kernel_list=(3, 4, 5), dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 1表示channel_num，filter_num即输出数据通道数，卷积核大小为(kernel, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, filter_num, (kernel, embedding_dim)),
                          nn.LeakyReLU(),
                          nn.MaxPool2d((config.LARGE_SENTENCE_SIZE - kernel + 1, 1)))
            for kernel in kernel_list
        ])
        self.fc = nn.Linear(filter_num * len(kernel_list), output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)  # [128, 50, 200] (batch, seq_len, embedding_dim)
        x = x.unsqueeze(1)     # [128, 1, 50, 200] 即(batch, channel_num, seq_len, embedding_dim)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)   # [128, 300, 1, 1]，各通道的数据拼接在一起
        out = out.view(x.size(0), -1)  # 展平
        out = self.dropout(out)        # 构建dropout层
        logits = self.fc(out)          # 结果输出[128, 2]
        return logits


# 数据获取
VOB_SIZE, pos_text, neg_text, total_text, index2word, word2index = dataloader.get_attrs()
# 数据处理
X_train, X_test, y_train, y_test = dataloader.number_sentence(pos_text, neg_text, word2index)
# 模型构建
cnn = TextCNN(VOB_SIZE, config.EMBEDDING_SIZE, 2)
# print(cnn)
# 优化器选择
optimizer = optim.Adam(cnn.parameters(), lr=config.LEARNING_RATE)
# 损失函数选择
criterion = nn.CrossEntropyLoss()


def train(model, opt, loss_function):
    """
    训练函数
    :param model: 模型
    :param opt: 优化器
    :param loss_function: 使用的损失函数
    :return: 该轮训练模型的损失值
    """
    avg_acc = []
    model.train()  # 模型处于训练模式
    # 批次训练
    for x_batch, y_batch in dataloader.get_batch(X_train, y_train):
        x_batch = torch.LongTensor(x_batch)  # 需要是Long类型
        y_batch = torch.tensor(y_batch).long()
        y_batch = y_batch.squeeze()  # 数据压缩到1维
        pred = model(x_batch)        # 模型预测
        # 获取批次预测结果最大值，max返回最大值和最大索引（已经默认索引为0的为负类，1为为正类）
        acc = utils.binary_acc(torch.max(pred, dim=1)[1], y_batch)
        avg_acc.append(acc)  # 记录该批次正确率
        # 使用损失函数计算损失值，预测值要放在前
        loss = loss_function(pred, y_batch)
        # 清楚之前的梯度值
        opt.zero_grad()
        # 反向传播
        loss.backward()
        # 参数更新
        opt.step()
    # 所有批次数据的正确率计算
    avg_acc = np.array(avg_acc).mean()
    return avg_acc


def evaluate(model):
    """
    模型评估
    :param model: 使用的模型
    :return: 返回当前训练的模型在测试集上的结果
    """
    avg_acc = []
    model.eval()  # 打开模型评估状态
    with torch.no_grad():
        for x_batch, y_batch in dataloader.get_batch(X_test, y_test):
            x_batch = torch.LongTensor(x_batch)
            y_batch = torch.tensor(y_batch).long().squeeze()
            pred = model(x_batch)
            acc = utils.binary_acc(torch.max(pred, dim=1)[1], y_batch)
            avg_acc.append(acc)
    avg_acc = np.array(avg_acc).mean()
    return avg_acc


# 记录模型训练过程中模型在训练集和测试集上模型预测正确率表现
cnn_train_acc, cnn_test_acc = [], []
# 模型迭代训练
for epoch in range(config.EPOCH):
    # 模型训练
    train_acc = train(cnn, optimizer, criterion)
    print('epoch={},训练准确率={}'.format(epoch, train_acc))
    # 模型测试
    test_acc = evaluate(cnn)
    print("epoch={},测试准确率={}".format(epoch, test_acc))
    cnn_train_acc.append(train_acc)
    cnn_test_acc.append(test_acc)


# 模型训练过程结果展示
plt.plot(cnn_train_acc)
plt.plot(cnn_test_acc)

plt.ylim(ymin=0.5, ymax=1.01)
plt.title("The accuracy of textCNN model")
plt.legend(["train", 'test'])
plt.show()

