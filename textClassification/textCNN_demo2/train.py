#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: train.py
@time:2020/12/27
@description:
"""

import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torchtext import data
import random
import config
import dataloader
import model
import utils
import time

torch.manual_seed(config.RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 文本内容，使用自定义的分词方法，将内容转换为小写，设置最大长度等
TEXT = data.Field(tokenize=utils.en_seg, lower=True, fix_length=config.MAX_SENTENCE_SIZE, batch_first=True)
# 文本对应的标签
LABEL = data.LabelField(dtype=torch.float)

# 构建data数据
pos_examples, pos_fields = dataloader.get_dataset(config.POS_CORPUS_PATH, TEXT, LABEL, 'pos')
neg_examples, neg_fields = dataloader.get_dataset(config.NEG_CORPUS_PATH, TEXT, LABEL, 'neg')
all_examples, all_fields = pos_examples + neg_examples, pos_fields + neg_fields

# 构建torchtext类型的数据集
total_data = data.Dataset(all_examples, all_fields)

# 数据集切分
train_data, test_data = total_data.split(random_state=random.seed(config.RANDOM_SEED), split_ratio=0.8)

# 切分后的数据查看
# # 数据维度查看
print('len of train data: %r' % len(train_data))  # len of train data: 8530
print('len of test data: %r' % len(test_data))  # len of test data: 2132

# # 抽一条数据查看
print(train_data.examples[100].text)
# ['never', 'engaging', ',', 'utterly', 'predictable', 'and', 'completely', 'void', 'of', 'anything', 'remotely',
# 'interesting', 'or', 'suspenseful']
print(train_data.examples[100].label)
# 0

# 为该样本数据构建字典，并将子每个单词映射到对应数字
TEXT.build_vocab(train_data)
LABEL.build_vocab(train_data)

# 查看字典长度
print(len(TEXT.vocab))  # 19206
# 查看字典中前10个词语
print(TEXT.vocab.itos[:10])  # ['<unk>', '<pad>', ',', 'the', 'a', 'and', 'of', 'to', '.', 'is']
# 查找'name'这个词对应的词典序号, 本质是一个dict
print(TEXT.vocab.stoi['name'])  # 2063

# 构建迭代(iterator)类型的数据
train_iterator, test_iterator = data.BucketIterator.splits((train_data, test_data),
                                                           batch_size=config.BATCH_SIZE,
                                                           sort=False)

# 创建模型
text_cnn = model.TextCNN(len(TEXT.vocab), config.EMBEDDING_SIZE, len(LABEL.vocab)).to(device)
# 选取优化器
optimizer = optim.Adam(text_cnn.parameters(), lr=config.LEARNING_RATE)
# 选取损失函数
criterion = nn.CrossEntropyLoss()

# 绘制结果
model_train_acc, model_test_acc = [], []
start = time.time()
# 模型训练
for epoch in range(config.EPOCH):
    train_acc = utils.train(text_cnn, train_iterator, optimizer, criterion)
    print("epoch = {}, 训练准确率={}".format(epoch + 1, train_acc))

    test_acc = utils.evaluate(text_cnn, test_iterator)
    print("epoch = {}, 测试准确率={}".format(epoch + 1, test_acc))

    model_train_acc.append(train_acc)
    model_test_acc.append(test_acc)

print('total train time:', time.time() - start)
# 绘制训练过程
plt.plot(model_train_acc)
plt.plot(model_test_acc)
plt.ylim(ymin=0.5, ymax=1.01)
plt.title("The accuracy of textCNN model")
plt.legend(['train', 'test'])
plt.show()

