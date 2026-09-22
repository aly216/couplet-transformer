"""
数据加载与预处理工具（中文对联生成）

职责：
1. 读 data/vocabs 构建 word2idx / idx2word，追加 <pad> <unk> 两个特殊 token
2. 读 train/test 的上联 in.txt / 下联 out.txt，每行按空格逐字切分，首尾加 <s> </s>
3. 转成 token id，按长度分桶 + padding + 构造 mask，供 DataLoader 使用

特殊 token 约定（与 data/vocabs 前两行一致）：
    <s>   = 0     起始符（vocabs 第 1 行）
    </s>  = 1     结束符（vocabs 第 2 行）
    <pad> = 9130  填充符（build_vocab 动态追加，由返回值给出）
    <unk> = 9131  未知字（build_vocab 动态追加，由返回值给出）
"""
import functools

import torch
from torch.utils.data import Dataset, DataLoader

# 固定特殊 token（data/vocabs 前两行）
SOS_IDX = 0   # <s> 起始符
EOS_IDX = 1   # </s> 结束符


def build_vocab(vocab_path):
    """读 data/vocabs 建 word2idx / idx2word，并追加 <pad> <unk>。

    返回 (word2idx, idx2word, pad_idx, unk_idx)。
    """
    with open(vocab_path, encoding='utf-8') as f:
        tokens = [line.strip() for line in f if line.strip()]

    word2idx = {token: idx for idx, token in enumerate(tokens)}
    pad_idx = len(tokens)
    unk_idx = len(tokens) + 1
    word2idx['<pad>'] = pad_idx
    word2idx['<unk>'] = unk_idx
    idx2word = {idx: token for token, idx in word2idx.items()}

    return word2idx, idx2word, pad_idx, unk_idx


def read_pairs(in_path, out_path, limit=None):
    """读上联/下联文件，返回 [(src_chars, tgt_chars), ...]，每句首尾加 <s> </s>。

    数据文件每行已经按空格逐字分隔（如 "晚 风 摇 树 树 还 挺"），
    所以 split() 即可得到字列表。
    limit 用于只读前 N 对（跑通 pipeline 时先取 1% 数据）。
    """
    pairs = []
    with open(in_path, encoding='utf-8') as fin, open(out_path, encoding='utf-8') as fout:
        for i, (src_line, tgt_line) in enumerate(zip(fin, fout)):
            if limit is not None and i >= limit:
                break
            src = ['<s>'] + src_line.split() + ['</s>']
            tgt = ['<s>'] + tgt_line.split() + ['</s>']
            pairs.append((src, tgt))
    return pairs


def encode(chars, word2idx, unk_idx):
    """字列表 -> token id 列表，未知字映射到 <unk>。"""
    return [word2idx.get(c, unk_idx) for c in chars]


class CoupletDataset(Dataset):
    """把 (src, tgt) 字对预先编码成 token id，供 DataLoader 使用。"""

    def __init__(self, pairs, word2idx, unk_idx):
        self.data = [
            (encode(src, word2idx, unk_idx), encode(tgt, word2idx, unk_idx))
            for src, tgt in pairs
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def subsequent_mask(size):
    """(1, size, size) 下三角 mask，遮挡未来位置（因果注意力）。"""
    return torch.tril(torch.ones(1, size, size))


def collate_fn(batch, pad_idx):
    """把一个 batch 的 (src_ids, tgt_ids) pad 成对齐张量，并构造 mask。

    batch: list[(src_ids, tgt_ids)]
    返回 (src, tgt, src_mask, tgt_mask)：
        src      (batch, src_len)           上联 token id
        tgt      (batch, tgt_len)           下联 token id（含 <s> </s>）
        src_mask (batch, 1, src_len)        源 padding mask，非 pad 处 = 1
        tgt_mask (batch, tgt_len, tgt_len)  目标因果 mask，下三角 = 1
    """
    src_list = [item[0] for item in batch]
    tgt_list = [item[1] for item in batch]

    src_len = max(len(s) for s in src_list)
    tgt_len = max(len(t) for t in tgt_list)

    # pad 到本 batch 最长
    src_pad = [s + [pad_idx] * (src_len - len(s)) for s in src_list]
    tgt_pad = [t + [pad_idx] * (tgt_len - len(t)) for t in tgt_list]

    src = torch.tensor(src_pad, dtype=torch.long)
    tgt = torch.tensor(tgt_pad, dtype=torch.long)

    # 源 padding mask：(batch, 1, src_len)，非 pad 处 = 1
    src_mask = (src != pad_idx).unsqueeze(-2).float()
    # 目标因果 mask：(batch, tgt_len, tgt_len)，下三角 & 非 pad
    tgt_mask = ((tgt != pad_idx).unsqueeze(-2) & subsequent_mask(tgt_len).bool()).float()

    return src, tgt, src_mask, tgt_mask


def get_dataloader(pairs, word2idx, unk_idx, pad_idx, batch_size,
                   shuffle=True, num_workers=0):
    """把字对包装成 DataLoader（按长度分桶交给 collate_fn 内 padding）。"""
    dataset = CoupletDataset(pairs, word2idx, unk_idx)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=functools.partial(collate_fn, pad_idx=pad_idx),
        num_workers=num_workers,
    )


if __name__ == '__main__':
    # —— 冒烟测试：建词表、读数据、取一个 batch，看形状与解码结果 ——
    word2idx, idx2word, pad_idx, unk_idx = build_vocab('data/vocabs')
    print(f'词表大小 = {len(word2idx)}，<pad> = {pad_idx}，<unk> = {unk_idx}')

    pairs = read_pairs('data/train/in.txt', 'data/train/out.txt')
    print(f'训练对数量 = {len(pairs)}')

    # 取前 8 对看形状与解码
    loader = get_dataloader(pairs[:8], word2idx, unk_idx, pad_idx,
                            batch_size=8, shuffle=False)
    src, tgt, src_mask, tgt_mask = next(iter(loader))
    print('src      形状:', tuple(src.shape))
    print('tgt      形状:', tuple(tgt.shape))
    print('src_mask 形状:', tuple(src_mask.shape))
    print('tgt_mask 形状:', tuple(tgt_mask.shape))

    print('第 0 条 上联:', ''.join(idx2word[i] for i in src[0].tolist()))
    print('第 0 条 下联:', ''.join(idx2word[i] for i in tgt[0].tolist()))
