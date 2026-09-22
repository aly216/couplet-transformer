import torch
import torch.nn as nn

from encoder_sublayer import SublayerConnection
from encoder_element import clones, MultiHeadAttention, FeedForward
from input_part import use_position

class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout_p=0.1):
        """
        初始化编码器层
        :param d_model: 输入向量的维度
        :param self_attn: 自注意力模块
        :param feed_forward: 前馈网络
        :param dropout_p: dropout概率
        """
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout_p), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, lambda x: self.feed_forward(x))
        return x


def use_encoder_layer():
    x = use_position()
    encoder_layer = EncoderLayer(512, MultiHeadAttention(512, 8), FeedForward(512, 2048))
    mask = torch.ones((x.size(0), 1, x.size(1)))  # 无 padding，全 1 的 padding mask
    result = encoder_layer(x, mask)
    print(result.shape)
    return result


if __name__ == '__main__':
    use_encoder_layer()
