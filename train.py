import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (device, vocab_path, train_in_path, train_out_path,
                    epochs, batch_size, warmup,
                    print_interval_num, plot_interval_num, save_dir, img_dir)
from transformer import make_model, D_MODEL
from data_utils import build_vocab, read_pairs, get_dataloader


class NoamOpt:
    """原版 Transformer 的 Noam 学习率调度。

    lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})
    前 warmup 步线性上升，之后按 step^{-0.5} 缓慢衰减。
    """

    def __init__(self, model_size, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self._warmup = warmup
        self._rate = 0.0
        self.model_size = model_size

    def step(self):
        """先按当前步数更新 lr，再执行一步优化。"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.model_size ** (-0.5) * min(
            step ** (-0.5), step * self._warmup ** (-1.5)
        )


def train_iters(src, tgt, src_mask, tgt_mask, model, optimizer, scheduler, criterion):
    """单步训练：teacher forcing 前向 + 反向 + 更新，返回该步 loss。

    输入 src/tgt 为 (batch, seq_len) 的 token id，mask 由 collate_fn 构造。
    """
    src = src.to(device)
    tgt = tgt.to(device)
    src_mask = src_mask.to(device)
    tgt_mask = tgt_mask.to(device)

    # teacher forcing：decoder 输入 tgt[:-1]，目标是 tgt[1:]
    dec_input = tgt[:, :-1]
    dec_mask = tgt_mask[:, :-1, :-1]   # 因果 mask 同步去掉最后一列

    logits = model(src, dec_input, src_mask, dec_mask)  # (batch, len-1, vocab)
    loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scheduler.step()   # Noam：内部先更新 lr 再 optimizer.step()

    return loss.item()


def train_transformer():
    # —— 数据（全量） ——
    word2idx, idx2word, pad_idx, unk_idx = build_vocab(vocab_path)
    train_pairs = read_pairs(train_in_path, train_out_path)
    print(f'设备：{device}')
    print(f'词表大小：{len(word2idx)}，训练样本：{len(train_pairs)}')
    my_dataloader = get_dataloader(train_pairs, word2idx, unk_idx, pad_idx,
                                   batch_size, shuffle=True)

    # —— 模型 ——
    model = make_model().to(device)

    # —— 优化器 + Noam 调度（原版配方） ——
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamOpt(D_MODEL, warmup, optimizer)

    # —— 损失：标签平滑（创新点） ——
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    plot_loss_list = []   # 每 plot_interval 步的平均 loss，用于画曲线
    for epoch_idx in range(1, epochs + 1):
        model.train()
        print_loss_total = 0.0
        plot_loss_total = 0.0
        start_time = time.time()

        for item, (src, tgt, src_mask, tgt_mask) in enumerate(
                tqdm(my_dataloader, desc=f'epoch {epoch_idx}'), start=1):
            loss = train_iters(src, tgt, src_mask, tgt_mask,
                               model, optimizer, scheduler, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if item % print_interval_num == 0:
                print_loss_avg = print_loss_total / print_interval_num
                print_loss_total = 0.0
                print(f'批次{item}，损失{print_loss_avg:.4f}，'
                      f'lr {scheduler._rate:.2e}，时间{time.time() - start_time:.2f}')

            if item % plot_interval_num == 0:
                plot_loss_avg = plot_loss_total / plot_interval_num
                plot_loss_list.append(plot_loss_avg)
                plot_loss_total = 0.0

        # 每个 epoch 保存一次 checkpoint（含词表，方便推理时解码）
        torch.save({'model': model.state_dict(),
                    'epoch': epoch_idx,
                    'word2idx': word2idx},
                   f'./{save_dir}/epoch{epoch_idx}.pt')

        # 画 loss 曲线
        plt.figure()
        plt.plot(plot_loss_list)
        plt.xlabel(f'step (x{plot_interval_num})')
        plt.ylabel('loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'./{img_dir}/plot_loss_{epoch_idx}.png')
        plt.close()
        print(f'epoch {epoch_idx} 完成，用时 {time.time() - start_time:.1f}s，'
              f'已保存 {save_dir}/epoch{epoch_idx}.pt 与 {img_dir}/plot_loss_{epoch_idx}.png')

    return plot_loss_list


if __name__ == '__main__':
    train_transformer()
