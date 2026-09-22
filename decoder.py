import copy

import torch
import torch.nn as nn

from encoder_element import clones, LayerNorm, MultiHeadAttention, FeedForward
from input_part import Embeddings, PositionalEncoding
from encoder import use_encoder
from decoder_layer import DecoderLayer

class Decoder(nn.Module):
    def __init__(self,layer,N):
        super().__init__()
        self.layers=clones(layer,N)
        self.norm=LayerNorm(layer.d_model)

    def forward(self,x,encoder_output,src_mask,tgt_mask):
        """
        :param x: 输入序列(一开经过位置处理的Q)
        :param encoder_output: 编码器输出
        :param src_mask: 源序列掩码,用于编码器-解码器注意力机制
        :param tgt_mask: 目标序列掩码,用于自注意力机制
        """
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)


def use_decoder():
    y=torch.LongTensor([[1,2,3,4],[4,5,6,2]])
    #把上述转为词向量
    my_embed=Embeddings(10000,512)
    embed_y=my_embed(y)
    #将embed_y-->位置编码
    my_position=PositionalEncoding(512,0.1)
    pos_y=my_position(embed_y)
    
    mult_attn=MultiHeadAttention(512,8)
    self_attn=copy.deepcopy(mult_attn)
    src_attn=copy.deepcopy(mult_attn)

    ff=FeedForward(512,2048)
    #获取编码器输出
    encoder_output=use_encoder()
    
    # 源 padding mask：无 padding 全 1，形状 (batch, 1, src_len)
    src_mask=torch.ones((encoder_output.size(0),1,encoder_output.size(1)))
    # 目标因果掩码：下三角=1（可看到当前及之前），上三角=0（遮住未来）
    tgt_mask=torch.tril(torch.ones(pos_y.size(0),pos_y.size(1),pos_y.size(1)))

    my_decoder_layer=DecoderLayer(512,self_attn,src_attn,ff)
    my_decoder=Decoder(my_decoder_layer,8)
    result=my_decoder(pos_y,encoder_output,src_mask,tgt_mask)
    # print(result.shape)
    # print(f"\n{my_decoder}")
    # print('-----------------')
    # print(f"\n{my_decoder_layer}")
    return result


if __name__ == '__main__':
    use_decoder()