#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: utils.py
@time:2020/12/27
@description: 相关函数
"""
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def en_seg(sentence):
    """
    简单的英文分词方法，
    :param sentence: 需要分词的语句
    :return: 返回分词结果
    """
    return sentence.split()


def binary_acc(pred, y):
    """
    计算模型的准确率
    :param pred: 预测值
    :param y: 实际真实值
    :return: 返回准确率
    """
    correct = torch.eq(pred, y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, train_data, optimizer, criterion):
    """
    模型训练
    :param model: 训练的模型
    :param train_data: 训练数据
    :param optimizer: 优化器
    :param criterion: 损失函数
    :return: 该论训练各批次正确率平均值
    """
    avg_acc = []
    model.train()  # 进入训练模式
    for i, batch in enumerate(train_data):
        pred = model(batch.text.to(device)).cpu()
        loss = criterion(pred, batch.label.long())
        acc = binary_acc(torch.max(pred, dim=1)[1], batch.label)
        avg_acc.append(acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 计算所有批次数据的结果
    avg_acc = np.array(avg_acc).mean()
    return avg_acc


def evaluate(model, test_data):
    """
    使用测试数据评估模型
    :param model: 模型
    :param test_data: 测试数据
    :return: 该论训练好的模型预测测试数据，查看预测情况
    """
    avg_acc = []
    model.eval()  # 进入测试模式
    with torch.no_grad():
        for i, batch in enumerate(test_data):
            pred = model(batch.text.to(device)).cpu()
            acc = binary_acc(torch.max(pred, dim=1)[1], batch.label)
            avg_acc.append(acc)
    return np.array(avg_acc).mean()
