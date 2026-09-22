import torch.nn as nn
from encoder_element import LayerNorm, MultiHeadAttention
from input_part import use_position


class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout_p=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.norm = LayerNorm(d_model)

    def forward(self, x, sublayer):
        """
        前向传播（pre-norm：先规范化输入，再经过子层，最后残差连接）
        :param x: 输入向量，形状为(batch_size, seq_len, d_model)
        :param sublayer: 子层模块
        """
        return x + self.dropout(sublayer(self.norm(x)))


def use_sublayer_connection():
    x = use_position()
    sublayer_conn = SublayerConnection(512)
    # 采用匿名函数（lambda 表达式）把子层作为参数传入
    result = sublayer_conn(x, lambda x: MultiHeadAttention(512, 8)(x, x, x))
    return result
