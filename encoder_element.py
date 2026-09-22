import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import copy
from input_part import use_position
#上三角矩阵
def d01_test_triu():
    arr=[
        [1,2,3,4],
        [4,5,6,7],
        [7,8,9,1],
        [2,3,4,5]
    ]
    print(np.triu(arr))
#下三角掩码（含对角线），用于构造因果注意力掩码
def d02_test_triu(size):
    temp=np.triu(m=np.ones((1,size,size)),k=1).astype('uint8')
    print(temp)
    return torch.from_numpy(1-temp)

def d03_test_mask():
    plt.figure(figsize=(5,5))
    plt.imshow(d02_test_triu(10)[0])
    plt.show()

def attention(query, key, value, mask=None,dropout=None):
    """
    计算注意力机制
    :param query: 查询向量，形状为(batch_size, seq_len, d_model)
    :param key: 键向量，形状为(batch_size, seq_len, d_model)
    :param value: 值向量，形状为(batch_size, seq_len, d_model)
    :param mask: 掩码张量，形状为(batch_size, 1, 1, seq_len)
    :param dropout: dropout层,用于随机将注意力权重设为0,以防止过拟合
    :return1: 输出向量，形状为(batch_size, seq_len, d_model)
    :return2: 注意力权重张量，形状为(batch_size, seq_len, seq_len)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) 
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)      
    return torch.matmul(p_attn, value), p_attn

def use_attention():
    position_x=use_position()
    query=key=value=position_x
    result1,p_attn=attention(query,key,value)
    # print(result1.shape) # (3, 3, 512) 3个位置的注意力机制输出
    # print(p_attn.shape) # (3, 3, 3) 3个位置的注意力权重张量

def clones(module, N):
    """
    复制N个模块
    :param module: 模块
    :param N: 复制次数
    :return: 复制后的模块
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim,head,dropout_p=0.1):
        super().__init__()
        assert embed_dim % head == 0
        self.h = head
        self.d_k = embed_dim // head

        self.linears=clones(nn.Linear(embed_dim, embed_dim), 4)
        self.dropout = nn.Dropout(dropout_p)
        self.attn=None


    def forward(self, query, key, value, mask=None):
        """
        前向传播
        :param query: 查询向量，形状为(batch_size, seq_len, d_model)
        :param key: 键向量，形状为(batch_size, seq_len, d_model)
        :param value: 值向量，形状为(batch_size, seq_len, d_model)
        :param mask: 掩码张量，形状为(batch_size, 1, seq_len)或(batch_size, seq_len, seq_len)
        :return: 输出向量，形状为(batch_size, seq_len, d_model)
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        self.batch=query.size(0)
        query,key,value=[
            model(x).view(self.batch, -1, self.h, self.d_k).transpose(1,2)
              for model,x in zip(self.linears, (query,key,value))
             #zip把多个可迭代对象打包成元组
        ]
        #x:(batch_size, seq_len, h, d_k),
        x, self.attn = attention(query, key, value, mask, self.dropout)
        attn_x=x.transpose(1,2).contiguous().view(self.batch, -1, self.h * self.d_k) 
        
        return self.linears[-1](attn_x)

def use_multihead():
    my_attn=MultiHeadAttention(512,8)
    position_x=use_position()
    query=key=value=position_x
    result=my_attn(query,key,value)
    # print(result.shape) 
    # print(my_attn.attn.shape)
    return result

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p=0.1):
        """
        初始化前馈网络
        :param d_model: 输入向量的维度
        :param d_ff: 中间层的维度
        :param dropout_p: dropout概率
        """
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, x):
        x=self.w1(x)
        x=self.dropout(F.relu(x))
        x=self.w2(x)
        return x    

def use_ff():
    attn_x=use_multihead()
    my_ff=FeedForward(512,2048)
    result=my_ff(attn_x)
    # print(result)
    # print(result.shape)
    return result

class LayerNorm(nn.Module):
    def __init__(self, features,eps=1e-6):
        super().__init__()
        self.a=nn.Parameter(torch.ones(features))  
        self.b=nn.Parameter(torch.zeros(features))
        self.eps=eps# 防止除0错误，添加一个小的常量
    def forward(self, x):
        """
        前向传播
        :param x: 输入向量，形状为(batch_size, seq_len, d_model)
        :return: 输出向量，形状为(batch_size, seq_len, d_model)
        """
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True,unbiased=False)
        return self.a*(x-mean)/(std+self.eps)+self.b

def use_layer_norm():
    ff_x=use_ff()
    my_layer_norm=LayerNorm(512)
    result=my_layer_norm(ff_x)
    print(f"规范化后的形状为:{result.shape}")
    return result   









if  __name__ == '__main__':
    pass
    # d01_test_triu()
    # d01_test_triu()
    # print(d02_test_triu(5))
    # print('-'*20)
    # print(d02_test_triu(5)[0])
    # d03_test_mask()
    # use_attention()
    # use_multihead()
    # use_ff()
    # use_layer_norm()