#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: config.py
@time:2020/12/27
@description: 相关参数配置文件
"""

# 模型相关参数
RANDOM_SEED = 1000  # 随机数种子
BATCH_SIZE = 128    # 批次数据大小
LEARNING_RATE = 1e-3   # 学习率
EMBEDDING_SIZE = 200   # 词向量维度
MAX_SENTENCE_SIZE = 50  # 设置最大语句长度
EPOCH = 10            # 训练测轮次

# 语料路径
NEG_CORPUS_PATH = './corpus/neg.txt'
POS_CORPUS_PATH = './corpus/pos.txt'


