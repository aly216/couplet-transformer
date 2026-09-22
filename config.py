import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据路径
vocab_path = "data/vocabs"
train_in_path = "data/train/in.txt"
train_out_path = "data/train/out.txt"

# 训练超参数
epochs = 5                # 训练轮数
batch_size = 64           # 批大小（RTX 4060 8GB 显存可跑）
warmup = 4000             # Noam 学习率 warmup 步数
print_interval_num = 100  # 每多少步打印一次 loss
plot_interval_num = 100   # 每多少步取一次平均 loss 画图（设为 1 即每步原始 loss）

# 保存路径
save_dir = "checkpoints"
img_dir = "img"

# 评测
ckpt_path = "checkpoints/epoch5.pt"   # 要评测的 checkpoint（训练完改这里）
num_samples = None                    # 只评前 N 条，None = 全部 4000
max_len = 40                          # 贪心解码最大长度
num_show = 20                         # 抽样展示条数
