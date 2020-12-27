#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: dataloader.py
@time:2020/12/06
@description:
"""
import numpy as np
from collections import Counter
import random
import torch
from sklearn.model_selection import train_test_split
import config

random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)


def read_data(filename):
    """
    数据读取
    :param filename: 文件路径
    :return: 数据读取内容（整个文档的字符串）
    """
    with open(filename, "r", encoding="utf8") as reader:
        content = reader.read()
    return content


def get_attrs():
    """
    获取语料相关参数
    :return: vob_size, pos_text, neg_text, total_text, index2word, word2index
    """
    pos_text, neg_text = read_data("corpus/pos.txt"), read_data("corpus/neg.txt")
    total_text = pos_text + '\n' + neg_text

    text = total_text.split()
    vocab = [w for w, f in Counter(text).most_common() if f > 1]
    vocab = ['<pad>', '<unk>'] + vocab

    index2word = {i: word for i, word in enumerate(vocab)}
    word2index = {word: i for i, word in enumerate(vocab)}

    return len(word2index), pos_text, neg_text, total_text, index2word, word2index


def convert_text2index(sentence, word2index, max_length=config.LARGE_SENTENCE_SIZE):
    """
    将语料转成数字化数据
    :param sentence: 单条文本
    :param word2index: 词语-索引的字典
    :param max_length: text_cnn需要的文本最大长度
    :return: 对语句进行截断和填充的数字化后的结果
    """
    unk_id = word2index['<unk>']
    pad_id = word2index['<pad>']
    # 对句子进行数字化转换，对于未在词典中出现过的词用unk的index填充
    indexes = [word2index.get(word, unk_id) for word in sentence.split()]
    if len(indexes) < max_length:
        indexes.extend([pad_id] * (max_length - len(indexes)))
    else:
        indexes = indexes[:max_length]
    return indexes


def number_sentence(pos_text, neg_text, word2index):
    """
    语句数字化处理
    :param pos_text: 正例全部文本
    :param neg_text: 负例全部文本
    :param word2index: 词到数字的字典
    :return: 经过训练集和测试集划分的结果X_train, X_test, y_train, y_test
    """
    pos_indexes = [convert_text2index(sentence, word2index) for sentence in pos_text.split('\n')]
    neg_indexes = [convert_text2index(sentence, word2index) for sentence in neg_text.split('\n')]

    # 为了方便处理，转化为numpy格式
    pos_indexes = np.array(pos_indexes)
    neg_indexes = np.array(neg_indexes)

    total_indexes = np.concatenate((pos_indexes, neg_indexes), axis=0)

    pos_targets = np.ones((pos_indexes.shape[0]))  # 正例设置为1
    neg_targets = np.zeros((neg_indexes.shape[0]))  # 负例设置为0

    total_targets = np.concatenate((pos_targets, neg_targets), axis=0).reshape(-1, 1)

    return train_test_split(total_indexes, total_targets, test_size=0.2)


def get_batch(x, y, batch_size=config.BATCH_SIZE, shuffle=True):
    """
    构建迭代器，获取批次数据
    :param x: 需要划分全部特征数据的数据集
    :param y: 需要划分全部标签数据的数据集
    :param batch_size: 批次大小
    :param shuffle: 是否打乱
    :return: 以迭代器的方式返回数据
    """
    assert x.shape[0] == y.shape[0], "error shape!"
    if shuffle:
        # 该函数是对[0, x.shape[0])进行随机排序
        shuffled_index = np.random.permutation(range(x.shape[0]))
        # 使用随机排序后的索引获取新的数据集结果
        x = x[shuffled_index]
        y = y[shuffled_index]

    n_batches = int(x.shape[0] / batch_size)  # 统计共几个完整的batch
    for i in range(n_batches - 1):
        x_batch = x[i*batch_size: (i + 1)*batch_size]
        y_batch = y[i*batch_size: (i + 1)*batch_size]
        yield x_batch, y_batch


