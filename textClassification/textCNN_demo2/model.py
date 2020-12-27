#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author: juzipi
@file: model.py
@time:2020/12/27
@description: textcnn模型
"""
import torch
from torch import nn

import config


class TextCNN(nn.Module):
    # output_size为输出类别（2个类别，0和1）,三种kernel，size分别是3,4，5，每种kernel有100个
    def __init__(self, vocab_size, embedding_dim, output_size, filter_num=100, kernel_list=(3, 4, 5), dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 1表示channel_num，filter_num即输出数据通道数，卷积核大小为(kernel, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, filter_num, (kernel, embedding_dim)),
                          nn.LeakyReLU(),
                          nn.MaxPool2d((config.MAX_SENTENCE_SIZE - kernel + 1, 1)))
            for kernel in kernel_list
        ])
        self.fc = nn.Linear(filter_num * len(kernel_list), output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)  # [128, 50, 200] (batch, seq_len, embedding_dim)
        x = x.unsqueeze(1)  # [128, 1, 50, 200] 即(batch, channel_num, seq_len, embedding_dim)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)  # [128, 300, 1, 1]，各通道的数据拼接在一起
        out = out.view(x.size(0), -1)  # 展平
        out = self.dropout(out)  # 构建dropout层
        logits = self.fc(out)  # 结果输出[128, 2]
        return logits
