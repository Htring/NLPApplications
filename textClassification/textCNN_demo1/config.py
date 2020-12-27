#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: config.py
@time:2020/12/06
@description: 配置文件
"""
LARGE_SENTENCE_SIZE = 50  # 句子最大长度
BATCH_SIZE = 128          # 语料批次大小
LEARNING_RATE = 1e-3      # 学习率大小
EMBEDDING_SIZE = 200      # 词向量维度
KERNEL_LIST = [3, 4, 5]   # 卷积核长度
FILTER_NUM = 100          # 每种卷积核输出通道数
DROPOUT = 0.5             # dropout概率
EPOCH = 20                # 训练轮次



