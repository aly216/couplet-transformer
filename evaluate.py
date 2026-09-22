"""
评测脚本：中文对联生成（上联 -> 下联）

指标：字数对齐率、distinct-n（任务相关）+ 字级 BLEU-1~4、chrF（参考对比）
并做抽样展示 + 按上联长度分层错误分析。

用法：直接运行 python evaluate.py 即可。
     要评测哪个 checkpoint，在 config.py 的 ckpt_path 里填（训练完改那一行）。
"""
import math
import random
from collections import Counter

import torch

from config import device, ckpt_path, num_samples, batch_size, max_len, num_show
from transformer import make_model
from data_utils import build_vocab, read_pairs, subsequent_mask, SOS_IDX, EOS_IDX


# ============ 指标实现（字级，自实现便于面试讲解） ============

def ngram_counts(tokens, n):
    """统计 token 序列的 n-gram 计数。"""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(refs, hyps, max_n=4):
    """语料级字级 BLEU。refs/hyps: list[list[str]]，单参考、一一对应。

    BLEU-N = brevity_penalty * (P1 * ... * PN)^(1/N)，Pn 为 clipped n-gram 精确率。
    """
    total_ref_len = sum(len(r) for r in refs)
    total_hyp_len = sum(len(h) for h in hyps)

    precisions = []
    for n in range(1, max_n + 1):
        total_clipped = 0
        total_hyp = 0
        for ref, hyp in zip(refs, hyps):
            ref_ng = ngram_counts(ref, n)
            hyp_ng = ngram_counts(hyp, n)
            total_hyp += sum(hyp_ng.values())
            total_clipped += sum(min(cnt, ref_ng.get(ng, 0)) for ng, cnt in hyp_ng.items())
        precisions.append(total_clipped / total_hyp if total_hyp > 0 else 0.0)

    # 简短惩罚（brevity penalty）
    bp = 1.0 if total_hyp_len >= total_ref_len else (
        math.exp(1 - total_ref_len / total_hyp_len) if total_hyp_len > 0 else 0.0)

    result = {}
    for i in range(max_n):
        if precisions[i] == 0.0:
            for j in range(i, max_n):
                result[f'BLEU-{j + 1}'] = 0.0
            break
        geo = math.exp(sum(math.log(p) for p in precisions[:i + 1]) / (i + 1))
        result[f'BLEU-{i + 1}'] = bp * geo
    return result


def corpus_chrf(refs, hyps, max_n=6):
    """语料级 chrF：字符 n-gram 的 P/R 的 F 值，在 n=1..6 上取平均。"""
    f_scores = []
    for n in range(1, max_n + 1):
        total_match = total_hyp = total_ref = 0
        for ref, hyp in zip(refs, hyps):
            ref_ng = ngram_counts(ref, n)
            hyp_ng = ngram_counts(hyp, n)
            total_hyp += sum(hyp_ng.values())
            total_ref += sum(ref_ng.values())
            total_match += sum(min(cnt, ref_ng.get(ng, 0)) for ng, cnt in hyp_ng.items())
        p = total_match / total_hyp if total_hyp > 0 else 0.0
        r = total_match / total_ref if total_ref > 0 else 0.0
        f_scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    return sum(f_scores) / max_n


def corpus_distinct(hyps, n):
    """distinct-n：生成文本中不重复 n-gram 的比例，衡量输出多样性。

    值越接近 1 说明模型越会"自己组织语言"，而不是只会抄固定模板；
    对联这种开放任务里，它比 BLEU 更能反映生成质量。
    """
    total = 0
    unique = set()
    for hyp in hyps:
        for i in range(len(hyp) - n + 1):
            total += 1
            unique.add(tuple(hyp[i:i + n]))
    return len(unique) / total if total > 0 else 0.0


# ============ 贪心解码 ============

def greedy_decode(model, src, src_mask, max_len, device, no_repeat_ngram=2):
    """批量贪心解码，返回 (batch, seq_len) token id，含开头 <s>。

    编码器只跑一次，之后每步用 model.decode + model.generator 生成下一个字。
    no_repeat_ngram：重复抑制窗口。生成时屏蔽"会构成已出现过 n-gram"的候选字，
    消除 月月月月 / 宏图…宏图 这类自增强循环，同时保留叠词（如 丝丝）只出现一次。
    设 0 可关闭抑制。
    """
    batch = src.size(0)
    memory = model.encode(src, src_mask)
    ys = torch.full((batch, 1), SOS_IDX, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        if finished.all():
            break
        tgt_mask = subsequent_mask(ys.size(1)).to(device)
        out = model.decode(ys, memory, src_mask, tgt_mask)
        logits = model.generator(out[:, -1])       # (batch, vocab) 原始 logits

        # 简单重复抑制：对每个未结束样本，屏蔽会复现已有 n-gram 的候选字
        if no_repeat_ngram > 0:
            n = no_repeat_ngram
            for b in range(batch):
                if finished[b]:
                    continue
                seq = ys[b].tolist()
                if len(seq) < n:
                    continue
                prefix = tuple(seq[-(n - 1):])                      # 最后 n-1 个字
                seen = {tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)}
                for g in seen:
                    if g[:-1] == prefix:                            # (前缀, 某字) 已出现过
                        logits[b, g[-1]] = -float('inf')

        next_token = logits.argmax(dim=-1)
        next_token = next_token.masked_fill(finished, EOS_IDX)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == EOS_IDX)
    return ys


def ids_to_chars(ids, idx2word):
    """token id -> 字列表，跳过 <s>，遇到 </s> 截断。"""
    chars = []
    for i in ids:
        if i == SOS_IDX:
            continue
        if i == EOS_IDX:
            break
        chars.append(idx2word.get(i, '<unk>'))
    return chars


def pad_src_batch(batch_ids, pad_idx):
    """把不等长的 src id 列表 pad 成张量，返回 (src, src_mask)。"""
    max_len = max(len(x) for x in batch_ids)
    padded = [x + [pad_idx] * (max_len - len(x)) for x in batch_ids]
    src = torch.tensor(padded, dtype=torch.long)
    src_mask = (src != pad_idx).unsqueeze(-2).float()
    return src, src_mask


# ============ 主流程 ============

def evaluate():

    word2idx, idx2word, pad_idx, unk_idx = build_vocab('data/vocabs')
    test_pairs = read_pairs('data/test/in.txt', 'data/test/out.txt', limit=num_samples)
    print(f'评测样本：{len(test_pairs)}')

    # 编码 src（上联）
    src_ids = [[word2idx.get(c, unk_idx) for c in s] for s, _ in test_pairs]

    # 加载模型
    model = make_model().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"已加载 checkpoint：{ckpt_path}（epoch {ckpt.get('epoch', '?')}）")

    # 批量贪心解码
    hyps, refs, src_lens = [], [], []
    with torch.no_grad():
        for i in range(0, len(test_pairs), batch_size):
            src_t, src_mask = pad_src_batch(src_ids[i:i + batch_size], pad_idx)
            src_t, src_mask = src_t.to(device), src_mask.to(device)
            out = greedy_decode(model, src_t, src_mask, max_len, device)
            for j in range(out.size(0)):
                hyps.append(ids_to_chars(out[j].tolist(), idx2word))
                refs.append(test_pairs[i + j][1][1:-1])           # 参考下联（去 <s> </s>）
                src_lens.append(len(test_pairs[i + j][0]) - 2)    # 上联字数

    # 指标
    bleu = corpus_bleu(refs, hyps)
    chrf = corpus_chrf(refs, hyps)
    align = sum(1 for s, h in zip(src_lens, hyps) if len(h) == s) / len(hyps)
    distinct1 = corpus_distinct(hyps, 1)
    distinct2 = corpus_distinct(hyps, 2)

    print('\n' + '=' * 60)
    print('评测结果')
    print('=' * 60)
    print(f'  字数对齐率: {align * 100:.2f}%')
    print(f'  distinct-1: {distinct1 * 100:.2f}%')
    print(f'  distinct-2: {distinct2 * 100:.2f}%')
    print()
    print('字级 BLEU:')
    for k in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']:
        print(f'  {k:>7}: {bleu[k] * 100:.2f}')
    print(f'  chrF    : {chrf * 100:.2f}')

    # 抽样展示
    print('\n' + '=' * 60)
    print(f'抽样展示（随机 {num_show} 条）')
    print('=' * 60)
    idxs = random.sample(range(len(test_pairs)), min(num_show, len(test_pairs)))
    for k in idxs:
        print(f"上联    : {''.join(test_pairs[k][0][1:-1])}")
        print(f"参考下联: {''.join(refs[k])}")
        print(f"生成下联: {''.join(hyps[k])}")
        print('-' * 60)


if __name__ == '__main__':
    evaluate()
