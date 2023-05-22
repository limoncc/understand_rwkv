#### 一、引言

RWKV模型是一种RNN范式下的大语言模型实现范式。效果是相当不错，关键它的训练和推理要求资源低。非常值得研究。

[Chatbot Arena Leaderboard Updates (Week 2)](https://lmsys.org/blog/2023-05-10-leaderboard/)

由于作者还未发表论文。这里只能借助源码来做一些分析。这篇文章就是笔者对源码分析的一些记录。希望对大家有所帮助。如有错误请大家指正。

#### 二、核心机制

源码中的RWKV的模块核心是两个机制：通道混合(Channel mixing)和 时间混合(Time mixing)。针对时间混合(Time mixing)笔者画了一个草图。


RWKV模型的时间混合(Time mixing)机制 这里所用符号与BlinkDL - Overview中的readme有所不同。

##### 2.1、时间混合机制(Time mixing)

先开个头，稍后更新。占坑