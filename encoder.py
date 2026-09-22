import torch
import torch.nn as nn

from encoder_element import clones, LayerNorm, MultiHeadAttention, FeedForward
from encoder_layer import EncoderLayer
from input_part import use_position

class Encoder(nn.Module):
    def __init__(self,layer,N):
        super().__init__()
        self.N=N
        self.layer=layer
        self.layers=clones(layer,N)
        self.norm=LayerNorm(layer.d_model)

    def forward(self,x,mask):
        for i in range(self.N):
            x=self.layers[i](x,mask)
        return self.norm(x)


def use_encoder():
    x = use_position()
    encoder_layer = Encoder(EncoderLayer(512, MultiHeadAttention(512, 8), FeedForward(512, 2048)),6)
    mask = torch.ones((x.size(0), 1, x.size(1)))  # 无 padding，全 1 的 padding mask
    result = encoder_layer(x, mask)
    print(result.shape)
    return result