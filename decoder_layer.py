import copy

import torch
import torch.nn as nn

from encoder_sublayer import SublayerConnection
from encoder_element import clones, MultiHeadAttention, FeedForward
from input_part import Embeddings, PositionalEncoding
from encoder import use_encoder

class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn,src_attn, feed_forward, dropout_p=0.1):
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(d_model, dropout_p), 3)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.src_attn(x, encoder_output, encoder_output, src_mask))  
        x = self.sublayers[2](x, lambda x: self.feed_forward(x))
        return x


def use_decoder_layer():
    y=torch.LongTensor([[1,2,3,4],[4,5,6,2]])
    my_embed=Embeddings(1000,512)
    embed_y=my_embed(y)

    my_position=PositionalEncoding(512,0.1)
    pos_y=my_position(embed_y)
    
    mult_attn=MultiHeadAttention(512,8)
    self_attn=copy.deepcopy(mult_attn)
    src_attn=copy.deepcopy(mult_attn)
    ff=FeedForward(512,2048)
    encoder_output=use_encoder()
    
    src_mask = torch.ones((encoder_output.size(0), 1, encoder_output.size(1)))
    tgt_mask = torch.tril(torch.ones(pos_y.size(0), pos_y.size(1), pos_y.size(1)))

    my_decoder_layer=DecoderLayer(512,self_attn,src_attn,ff)
    result=my_decoder_layer(pos_y,encoder_output,src_mask,tgt_mask)
    print(result.shape)
    return result

    

