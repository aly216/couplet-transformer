import copy

import torch
import torch.nn as nn

from input_part import Embeddings, PositionalEncoding
from encoder_element import MultiHeadAttention, FeedForward
from encoder_layer import EncoderLayer
from encoder import Encoder
from decoder_layer import DecoderLayer
from decoder import Decoder
from output_part import Generator

# 超参数：统一集中管理，避免魔法数字散落各处
VOCAB_SIZE = 9132    # 词表大小（data/vocabs 9130 词 + <pad> + <unk>）
D_MODEL = 512        # 词嵌入 / 模型维度
D_FF = 2048          # 前馈网络中间层维度
N_HEADS = 8          # 多头注意力头数
N_LAYERS = 6         # 编码器 / 解码器层数
DROPOUT = 0.1        # dropout 概率
MAX_LEN = 64         # 位置编码支持的最大序列长度（对联最长 32 字 + 特殊 token）


class EncoderDecoder(nn.Module):
    def __init__(self, src_embed, encoder, tgt_embed, decoder, generator):
        super().__init__()
        # 注意：src_embed / tgt_embed 是 nn.Sequential(Embeddings, PositionalEncoding)，
        # 已经同时包含“词嵌入 + 位置编码”两步，并不是单纯的嵌入层
        self.src_embed = src_embed
        self.encoder = encoder
        self.tgt_embed = tgt_embed
        self.decoder = decoder
        self.generator = generator

    def forward(self, source_x, target_y, source_mask, target_mask):
        encoder_result = self.encode(source_x, source_mask)
        decoder_result = self.decode(target_y, encoder_result, source_mask, target_mask)
        return self.generator(decoder_result)

    def encode(self, source_x, source_mask):
        embed_x = self.src_embed(source_x)
        return self.encoder(embed_x, source_mask)

    def decode(self, target_y, encoder_result, source_mask, target_mask):
        embed_y = self.tgt_embed(target_y)
        return self.decoder(embed_y, encoder_result, source_mask, target_mask)



def make_model():
    # 编码器
    source_embed = Embeddings(vocab_size=VOCAB_SIZE, d_model=D_MODEL)
    source_position = PositionalEncoding(d_model=D_MODEL, max_len=MAX_LEN)
    self_attn = MultiHeadAttention(D_MODEL, N_HEADS)
    ff = FeedForward(d_model=D_MODEL, d_ff=D_FF)
    encoder_layer = EncoderLayer(d_model=D_MODEL, self_attn=self_attn,
                                 feed_forward=ff, dropout_p=DROPOUT)
    encoder = Encoder(encoder_layer, N=N_LAYERS)

    # 解码器
    target_embed = copy.deepcopy(source_embed)
    target_position = copy.deepcopy(source_position)
    self_attn1 = copy.deepcopy(self_attn)
    source_attn = copy.deepcopy(self_attn)
    feed_forward = copy.deepcopy(ff)
    decoder_layer = DecoderLayer(d_model=D_MODEL, self_attn=self_attn1, src_attn=source_attn,
                                 feed_forward=feed_forward, dropout_p=DROPOUT)
    decoder = Decoder(decoder_layer, N=N_LAYERS)

    # 生成器
    generator = Generator(d_model=D_MODEL, vocab_size=VOCAB_SIZE)

    # 组装完整模型
    model = EncoderDecoder(
        nn.Sequential(source_embed, source_position),   # src_embed：词嵌入 + 位置编码
        encoder,
        nn.Sequential(target_embed, target_position),   # tgt_embed：词嵌入 + 位置编码
        decoder,
        generator,
    )
    # print(model)
    return model


if __name__ == '__main__':
    model = make_model()

