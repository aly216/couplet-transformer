import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        """
        初始化词嵌入层
        :param vocab_size: 词汇表大小
        :param d_model: 词嵌入的维度
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        #定义词嵌入层，将输入的词索引映射到词嵌入的向量
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        """
        初始化位置编码层
        :param d_model: 词嵌入的维度
        :param dropout:  dropout概率
        :param max_len: 最大序列长度
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)*( -math.log(10000.0) / d_model))

        position_value=position*div_term
        pe[:, 0::2] = torch.sin(position_value)
        pe[:, 1::2] = torch.cos(position_value)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        """
        前向传播
        :param x: 输入的词嵌入序列，形状为(batch_size, seq_len, d_model)
        :return: 位置编码后的序列，形状为(batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def use_position():
    vocab_size = 10000
    d_model = 512
    my_embed = Embeddings(vocab_size, d_model)
    x=torch.tensor([
        [1,2,3,4],
        [4,5,6,5]
        ])
    
    embed_x=my_embed(x)
    my_position=PositionalEncoding(d_model)
    pos_result=my_position(embed_x)
    return pos_result



       

if __name__ == '__main__':
    vocab_size = 10000
    d_model = 512
    my_embed = Embeddings(vocab_size, d_model)
    x=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])

    result = my_embed(x)
    print(result)
    print(result.shape)
    pos_result=use_position()
    print(pos_result)
    print(pos_result.shape)
