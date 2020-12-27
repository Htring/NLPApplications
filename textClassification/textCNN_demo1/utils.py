#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: utils.py
@time:2020/12/06
@description:
"""
import torch


def binary_acc(preds, y):
    """
    计算二元结果的准确率
    :param preds:比较值1
    :param y: 比较值2
    :return: 相同值比率
    """
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc
