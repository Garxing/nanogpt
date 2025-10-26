# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char' # 模型检查点和输出文件的保存目录。
eval_interval = 250 # 每训练250步，在验证集上评估一次模型性能。
eval_iters = 200 # 每次评估时，使用200个批次的数据计算平均损失。
log_interval = 10 # 每训练10步，在控制台打印一次当前损失等信息。

# 仅在验证集损失得到改善时才保存模型检查点，节省空间。
always_save_checkpoint = False

wandb_log = False # 是否使用Weights & Biases记录训练日志（此处关闭）。
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char' # 指定使用的数据集名称。
gradient_accumulation_steps = 1 # 梯度累积步数，通常用于模拟更大的批处理大小，此处为1即不累积。
batch_size = 64 # 批处理大小，即每次迭代训练所使用的样本数量。
block_size = 256 # 上下文窗口大小，即模型在生成下一个字符时，能看到的之前字符的最大数量

# baby GPT model :)
n_layer = 6 # Transformer神经网络模型的层数。
n_head = 6 # 每层Transformer中注意力机制的头数。
n_embd = 384 # 字符嵌入向量和模型内部隐藏层的维度。
dropout = 0.2 # 随机失活率，一种防止模型过拟合的正则化手段。

learning_rate = 1e-3 # 模型参数更新的步长。对于小型网络，可以设置得稍高一些
max_iters = 5000 # 训练过程的最大迭代步数。
lr_decay_iters = 5000 # 学习率开始衰减的迭代步数，通常与max_iters相同。
min_lr = 1e-4 # 学习率衰减后的最小值，通常是初始学习率的十分之一
beta2 = 0.99 # Adam优化器中的第二个动量参数，由于每次迭代的token数较少，此处调大以使优化更平滑

warmup_iters = 100 # 学习率预热步数，在最初100步内，学习率从0线性增加到设定值，避免模型在训练初期不稳定。

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
