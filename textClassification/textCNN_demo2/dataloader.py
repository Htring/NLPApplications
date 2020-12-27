#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: dataloader.py
@time:2020/12/27
@description: 数据加载
"""
from torchtext import data


def get_dataset(corpus_path, text_field, label_field, datatype):
    """
    构建torchtext数据集
    :param corpus_path: 数据路径
    :param text_field: torchtext设置的文本域
    :param label_field: torchtext设置的文本标签域
    :param datatype: 文本的类别
    :return: torchtext格式的数据集以及设置的域
    """
    fields = [('text', text_field), ('label', label_field)]
    examples = []
    with open(corpus_path, encoding='utf8') as reader:
        for line in reader:
            content = line.rstrip()
            if datatype == 'pos':
                label = 1
            else:
                label = 0
            # content[：-2]是由于原始文本最后的两个内容是空格和.，这里直接去掉，并将数据与设置的域对应起来
            examples.append(data.Example.fromlist([content[:-2], label], fields))

    return examples, fields
