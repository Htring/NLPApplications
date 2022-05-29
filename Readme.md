# NLPApplications

自然语言处理应用，使用的编程语言为Python，深度学习框架为Pytorch. 由于NLP相关任务较多，涉及算法也较多。本项目就把自己实现的算法、模型放在github上的地址进行汇总。

## 1 文本分类

基于深度学习的主流文本分类算法集合。

| 序号  | 博文                                                                                            | 代码                                                                                                         | 论文                                                                                                                                                                                                                             | 备注             |
|:---:| --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------- |
| 1   | [textCNN论文与原理——短文本分类(基于pytorch和torchtext)](https://mp.weixin.qq.com/s/L9sJJfP2j_PkzHI_4B9JJQ) | [https://github.com/Htring/TextCNN_Classification_PL](https://github.com/Htring/TextCNN_Classification_PL) | [Convolutional Neural Networks for Sentence Classification](https://www.aclweb.org/anthology/D14-1181.pdf)                                                                                                                     | textCNN        |
| 2   | [BERT论文阅读与总结](https://mp.weixin.qq.com/s/zR8lHJWQxdd1_QASoCEEig)                              | [https://github.com/Htring/BERT-Classification_PL](https://github.com/Htring/BERT-Classification_PL)       | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://www.semanticscholar.org/paper/BERT%3A-Pre-training-of-Deep-Bidirectional-for-Devlin-Chang/df2b0e26d0599ce3e70df8a9da02e51594e0e992) | BERT fine-tune |
|     |                                                                                               |                                                                                                            |                                                                                                                                                                                                                                |                |

## 2 命名实体识别

基于深度学习的主流命名实体识别算法集合。

| 序号  | 博文                                                                            | 代码                                                                                           | 论文                                                                                                                                                                                                                             | 备注              |
|:---:| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------- |
| 1   | [再看隐马尔可夫模型（HMM）原理](https://mp.weixin.qq.com/s/BvtsKM-OGuqln4_E3yBCEQ)         | [https://github.com/Htring/HMM_NER](https://github.com/Htring/HMM_NER)                       |                                                                                                                                                                                                                                | HMM             |
| 2   | [基于BiLSTM-CRF的序列标注](https://mp.weixin.qq.com/s/RYMGIN_S5n1uqL4Mj8nB9g)        | [https://github.com/Htring/BiLSTM-CRF_NER_PL](https://github.com/Htring/BiLSTM-CRF_NER_PL)   | [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991)                                                                                                                                         | BiLSTM-CRF      |
| 3   | [命名实体识别——IDCNN-CRF论文阅读与总结](https://mp.weixin.qq.com/s/Snv1L1nJpdL72OlTdyOuVA) | [https://github.com/Htring/IDCNN-CRF_NER_PL](https://github.com/Htring/IDCNN-CRF_NER_PL)     | [Fast and Accurate Entity Recognition with Iterated Dilated Convolutions](https://aclanthology.org/D17-1283/)                                                                                                                  | IDCNN-CRF       |
| 4   | [BERT论文阅读与总结](https://mp.weixin.qq.com/s/zR8lHJWQxdd1_QASoCEEig)              | [https://github.com/Htring/BERT-BiLSTM-CRF_PL](https://github.com/Htring/BERT-BiLSTM-CRF_PL) | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://www.semanticscholar.org/paper/BERT%3A-Pre-training-of-Deep-Bidirectional-for-Devlin-Chang/df2b0e26d0599ce3e70df8a9da02e51594e0e992) | BERT-BiLSTM-CRF |

## 3 文本匹配

常用的文本匹配算法。

| 序号  | 博文                                                                                                           | 代码                                                                                                     | 论文                                                                                                      | 备注       |
|:---:| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- | -------- |
| 1   | [非监督文本匹配算法——BM25](https://mp.weixin.qq.com/s/A4OOmG6YTL0ga6rarNoiwA)                                         | [https://github.com/Htring/BM25](https://github.com/Htring/BM25)                                       |                                                                                                         | BM25,无监督 |
| 2   | [文本匹配——Enhanced LSTM for Natural Language Inference阅读与总结](https://mp.weixin.qq.com/s/4onIQxfR6_5tmFi4xwXLfg) | [https://github.com/Htring/ESIM_Text_Similarity_PL](https://github.com/Htring/ESIM_Text_Similarity_PL) | [Enhanced LSTM for Natural Language Inference](https://arxiv.org/abs/1609.06038)                        | ESIM     |
| 3   | [文本匹配——RE2阅读与总结](https://mp.weixin.qq.com/s/ECx6IWrpGzTR-Zw7VCr4yA)                                          | [https://github.com/Htring/RE2_Text_Similarity_PL](https://github.com/Htring/RE2_Text_Similarity_PL)   | [Simple and Effective Text Matching with Richer Alignment Features](https://aclanthology.org/P19-1465/) | RE2      |
|     |                                                                                                              |                                                                                                        |                                                                                                         |          |

## 4 推荐算法

## 5  知识图谱

## 6 问答系统

## N 语料数据集

1. [https://github.com/SimmerChan/corpus](https://github.com/SimmerChan/corpus)
2. [https://github.com/SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)
3. [https://github.com/liucongg/NLPDataSet](https://github.com/liucongg/NLPDataSet)
4. [https://github.com/CLUEbenchmark/CLUEDatasetSearch](https://github.com/CLUEbenchmark/CLUEDatasetSearch)

## 联系我

1. 我的github：[https://github.com/Htring](https://github.com/Htring)
2. 我的csdn：[科皮子菊](https://piqiandong.blog.csdn.net/)
3. 我订阅号：AIAS编程有道
   ![AIAS编程有道](https://s2.loli.net/2022/05/05/DS37LjhBQz2xyUJ.png)
4. 知乎：[皮乾东](https://www.zhihu.com/people/piqiandong)
