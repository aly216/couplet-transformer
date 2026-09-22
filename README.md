# 从零实现 Transformer（中文对联生成）

从零手写《Attention Is All You Need》的 Transformer 序列到序列模型，不依赖 HuggingFace 等高层封装。多头注意力、位置编码、编码器 / 解码器、掩码、LayerNorm、残差连接、Noam 学习率调度全部自行实现，并在「上联 → 下联」对联生成任务上验证模型能正常工作。

## 想证明什么

这个项目的主线是**完整复现 Transformer 并理解每个组件的原理**，对联是验证模型能正常工作的任务载体。重点在架构实现，不在任务本身。

## 整体结构

标准 Encoder-Decoder：编码器 6 层、解码器 6 层、8 头注意力、d_model=512、d_ff=2048。

```
输入 → 词嵌入(×√d_model) + 位置编码
     → Encoder ×6（自注意力 + 前馈，各带残差 + LayerNorm）
     → Decoder ×6（因果自注意力 + 交叉注意力 + 前馈）
     → 线性层 → 词表 logits
```

## 核心组件实现

### 多头注意力

- 4 个线性层做 Q / K / V / O 投影，8 个头，每头 d_k = 512 / 8 = 64；
- `scores = QKᵀ / √d_k`，`masked_fill` 后 softmax、dropout，再乘 V；
- 流程：拆头 → 注意力 → 拼头 → 输出投影。

### 位置编码

- 正弦 sin/cos，用 `register_buffer` 保存（不参与梯度）；
- 选 sin/cos 而非可学习参数：可外推到训练未见过的序列长度，且相对位置可由线性关系表达。

### 前馈网络

- `Linear(512 → 2048) → ReLU → Dropout → Linear(2048 → 512)`。

### 残差连接 + LayerNorm（pre-norm）

- **采用 pre-norm**（先 LayerNorm 再子层）：`x + dropout(sublayer(norm(x)))`；
- 相比原论文的 post-norm，pre-norm 梯度更稳定、更好训练，是 GPT 等现代模型的做法；
- LayerNorm 自实现，`std` 用 `unbiased=False`（除以 N 而非 N-1）。

## 关键设计细节（为什么这么写）

| 细节 | 原因 |
|---|---|
| 注意力除以 √d_k | 防止点积随维度增大而增大，避免 softmax 进入饱和区、梯度消失 |
| 词嵌入 ×√d_model | 让嵌入与位置编码量级一致，相加后信息不被淹没 |
| 三种掩码 | encoder 用 padding mask；decoder 自注意力用因果 mask + padding mask；交叉注意力用 src mask |
| 因果掩码 | 下三角矩阵，保证预测第 t 个词时看不到 t 之后的词（自回归） |
| 标签平滑 0.1 | 防止模型对正确答案过度自信，缓解过拟合 |

## 训练

- Adam(β=0.9, 0.98) + **Noam 学习率调度**（warmup=4000，先线性上升后按 `step^-0.5` 衰减）；
- 梯度裁剪 1.0、teacher forcing、dropout 0.1；
- 词表 9132，5 个 epoch，batch 64（RTX 4060 8GB）。

## 任务与评测

中文对联生成（字符级建模），训练集 77 万对、测试集 4000 对。

| 指标 | 值 |
|---|---|
| 字数对齐率 | 79.53% |
| distinct-1 / distinct-2 | 3.62% / 36.53% |
| BLEU-1 / 2 / 3 / 4 | 15.44 / 6.01 / 2.78 / 1.48 |
| chrF | 3.12 |

> 对联是「一对多」开放生成，BLEU / chrF 假设唯一参考答案、并不适用，故以字数对齐率、distinct 为主指标，BLEU 仅作纵向回归参考。

## 局限

- 尚无 baseline 对照实验与验证集早停；
- generator 与目标 embedding 未做权重绑定（原论文有 tie）；
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
| `output_part.py` | 生成器（d_model → 词表 logits） |
| `transformer.py` | 组装 EncoderDecoder + `make_model` |
| `train.py` | 训练循环（Noam / 标签平滑 / 梯度裁剪） |
| `evaluate.py` | 解码 + 指标计算 + 分层错误分析 |
| `data_utils.py` | 词表构建 / 数据加载 / mask 构造 |
| `config.py` | 超参数集中管理 |
