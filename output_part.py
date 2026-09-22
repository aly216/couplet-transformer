import torch.nn as nn
import torch.nn.functional as F

from decoder import use_decoder

#用来把 d_model 维的向量，映射到词表维度，输出每个词的对数概率。
class Generator(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        self.linear=nn.Linear(d_model,vocab_size)

    def forward(self,x):
        x=self.linear(x)   #把模型学到的高维语义特征，投影到词表空间，得到每个词的原始得分（logits）
        return x           # 直接返回 logits，loss 里用 CrossEntropyLoss 做 log_softmax

       
def use_generator():
    result=use_decoder()
    generator=Generator(512,10000)
    output=generator(result)
    return output
    


