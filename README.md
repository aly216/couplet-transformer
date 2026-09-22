# 从零实现 Transformer（中文对联生成）

从零实现 Transformer 序列到序列模型，不依赖 HuggingFace 等高层封装。多头注意力、位置编码、编码器 / 解码器、掩码、LayerNorm、残差连接、Noam 学习率调度均自行编写，并在上联 -> 下联的对联生成任务上验证模型能正常工作。

## 整体结构

编码器 6 层、解码器 6 层、8 头注意力、d_model=512、d_ff=2048。

```
输入 -> 词嵌入 + 位置编码
     -> Encoder ×6（自注意力 + 前馈，各带残差 + LayerNorm）
     -> Decoder ×6（因果自注意力 + 交叉注意力 + 前馈）
     -> 线性层 -> 词表 logits
```

## 核心组件

### 多头注意力

- 4 个线性层做 Q / K / V / O 投影，8 个头，每头 d_k = 512 / 8 = 64；
- 流程：拆头 -> 注意力 -> 拼头 -> 输出投影。

### 位置编码

- 正弦 sin/cos 编码，用 `register_buffer` 保存（不参与训练）。

### 前馈网络

- `Linear(512 -> 2048) -> ReLU -> Dropout -> Linear(2048 -> 512)`。

### 残差连接 + LayerNorm

- 采用 pre-norm 结构（先 LayerNorm 再子层）；
- LayerNorm 自行实现。

### 掩码

- encoder 使用 padding mask；
- decoder 自注意力使用因果掩码 + padding mask；
- 交叉注意力使用源序列 mask。

## 训练

- Adam(β=0.9, 0.98) + Noam 学习率调度（warmup=4000）；
- 标签平滑 0.1、梯度裁剪 1.0、teacher forcing、dropout 0.1；
- 词表 9132，5 个 epoch，batch 64（RTX 4060 8GB）。

## 任务与评测

中文对联生成（字符级建模），训练集 77 万对、测试集 4000 对。

| 指标 | 值 |
|---|---|
| 字数对齐率 | 79.53% |
| distinct-1 / distinct-2 | 3.62% / 36.53% |
| BLEU-1 / 2 / 3 / 4 | 15.44 / 6.01 / 2.78 / 1.48 |
| chrF | 3.12 |

对联是一对多开放生成任务，BLEU / chrF 等参考式指标适用性有限，故以字数对齐率、distinct 为主要参考，BLEU 仅作纵向对比。

## 局限

- 尚无对照实验与验证集早停；
- 贪心解码对极短（0-5 字）和超长（16+ 字）对联的对齐偏弱。

## 运行

```bash
python train.py      # 训练，每 epoch 存 checkpoint 到 checkpoints/
python evaluate.py   # 评测（在 config.py 改 ckpt_path 指定 checkpoint）
```

> 训练集（`data/train/`，约 59MB）与模型权重（`checkpoints/`，约 1.2GB）未上传。训练数据请从 [couplet-dataset](https://github.com/wb14123/couplet-dataset) 的 Releases 下载放入 `data/train/`；权重需先运行 `train.py` 生成。

## 文件结构

| 文件 | 职责 |
|---|---|
| `encoder_element.py` | 多头注意力、前馈网络、LayerNorm |
| `input_part.py` | 词嵌入、位置编码 |
| `encoder_sublayer.py` | pre-norm 残差连接 |
| `encoder_layer.py` / `encoder.py` | 编码器层 / 编码器 |
| `decoder_layer.py` / `decoder.py` | 解码器层（含交叉注意力）/ 解码器 |
| `output_part.py` | 生成器（d_model -> 词表 logits） |
| `transformer.py` | 组装 EncoderDecoder + `make_model` |
| `train.py` | 训练循环 |
| `evaluate.py` | 解码 + 指标计算 |
| `data_utils.py` | 词表构建 / 数据加载 / mask 构造 |
| `config.py` | 超参数集中管理 |
