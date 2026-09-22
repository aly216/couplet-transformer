# 中文对联生成：从零实现 Transformer

Transformer 序列到序列模型，完成中文对联的「上联 → 下联」生成任务。不依赖 HuggingFace 等高层封装，多头注意力、位置编码、编码器/解码器、Noam 学习率调度等核心组件全部自行实现。

## 动机

对联生成是一个「一对多」的开放文本生成任务：同一个上联存在大量合格的下联。选择它作为练手项目，是因为它同时覆盖了 Transformer 的完整训练 / 解码链路，又能暴露 BLEU 等参考式指标在开放生成任务上的失效问题——后者是通用文本生成评测中的常见误区，也是本项目想讲清楚的一个点。

## 模型结构

| 组件 | 配置 |
|---|---|
| 词表 | 9132（9130 字 + `<pad>` + `<unk>`） |
| d_model / d_ff | 512 / 2048 |
| 注意力头 / 层数 | 8 / 6（encoder、decoder 各 6 层） |
| dropout | 0.1 |
| 位置编码 | 正弦 sin/cos，`register_buffer` 保存 |

关键实现细节：

- **pre-norm**：LayerNorm 在子层之前（现代 Transformer 默认，比原论文 post-norm 训练更稳定）；
- **缩放**：注意力得分 `1/√d_k`，词嵌入 `×√d_model`；
- **掩码**：encoder 用 padding mask；decoder 用因果 mask + padding mask；cross-attention 用 src mask；
- **LayerNorm** 自实现，`std` 用 `unbiased=False`（除以 N，而非 N-1）。

## 数据

- 训练集 **770,491** 对 / 测试集 **4,000** 对（标准中文对联数据集）；
- 字符级建模：每个汉字一个 token。

> 训练集（`data/train/`，约 59MB）未包含在仓库中。请从 [couplet-dataset](https://github.com/wb14123/couplet-dataset) 的 Releases 下载 `train/in.txt`、`train/out.txt` 放入 `data/train/`。仓库已包含 `data/vocabs` 与 `data/test/`，可直接评测。

## 训练

- Adam(β=0.9, 0.98) + **Noam 学习率调度**（warmup=4000，先线性上升后按 `step^-0.5` 衰减）；
- **标签平滑 0.1**、梯度裁剪 1.0、teacher forcing；
- 5 个 epoch，batch 64（RTX 4060 8GB）。

## 评测与结果

评测脚本用**贪心解码 + 重复抑制**（`no_repeat_ngram=2`，屏蔽会复现已有 2-gram 的候选字，消除 `月月月月` 这类自增强退化循环，同时保留叠词只出现一次）。

| 指标 | 值 |
|---|---|
| 字数对齐率 | 79.53% |
| distinct-1 / distinct-2 | 3.62% / 36.53% |
| BLEU-1 / 2 / 3 / 4 | 15.44 / 6.01 / 2.78 / 1.48 |
| chrF | 3.12 |

测试集随机抽样示例：

| 上联 | 生成下联 |
|---|---|
| 风铃串串春摇响 | 细雨丝丝柳线垂 |
| 衣上酒痕玫瑰紫 | 花间月色桂香浓 |
| 平生最是相思苦 | 往事无非寂寞愁 |

## 为什么这么选指标

BLEU / chrF 假设「存在唯一参考答案」，适合翻译等答案受限的任务。而对联是**一对多**开放生成：一条上联有无数合格下联，生成结果与那唯一一条参考下联字面不重叠，**不代表质量差**。抽样里 `柳岸春风入酒樽` 这类好对，BLEU 照样接近 0。

因此指标分两层：

- **主指标（任务相关）**：字数对齐率（对联硬性要求上下联等长）、distinct-n（衡量多样性，防止只抄固定模板）；
- **参考指标（回归报警）**：BLEU / chrF 仅作纵向对比的「退化报警器」，不当作质量结论。

## 局限与改进方向

- 尚无 LSTM / Seq2Seq 的 baseline 对照实验；
- 无独立验证集与早停，epoch 数为人工设定；
- 贪心解码对极短（0-5 字）和超长（16+ 字）对联的对齐偏弱；
- generator 与目标 embedding 未做权重绑定；src / tgt embedding 独立训练（非共享）。

## 运行

```bash
python train.py      # 训练，每 epoch 存 checkpoint 到 checkpoints/
python evaluate.py   # 评测（在 config.py 改 ckpt_path 指定 checkpoint）
```

> 模型权重（`checkpoints/`，约 1.2GB）未上传，需先运行 `train.py` 生成。

## 文件结构

| 文件 | 职责 |
|---|---|
| `input_part.py` | 词嵌入（×√d_model）+ 正弦位置编码 |
| `encoder_element.py` | 多头注意力、前馈网络、LayerNorm、`clones` |
| `encoder_sublayer.py` | pre-norm 残差连接 |
| `encoder_layer.py` / `encoder.py` | 编码器层 / 编码器 |
| `decoder_layer.py` / `decoder.py` | 解码器层（含 cross-attention）/ 解码器 |
| `output_part.py` | 生成器（d_model → 词表 logits） |
| `transformer.py` | 组装 EncoderDecoder + `make_model` |
| `train.py` | 训练循环（Noam / 标签平滑 / 梯度裁剪） |
| `evaluate.py` | 解码 + 指标计算 + 分层错误分析 |
| `data_utils.py` | 词表构建 / 数据加载 / mask 构造 |
| `config.py` | 超参数集中管理 |
