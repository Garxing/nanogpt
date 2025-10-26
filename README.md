## nanoGPT

### 莎士比亚字符级数据集实验

#### 1、配置环境

```bash
# 下载项目
git clone https://github.com/ChennXIao/nanogpt.git
# 配置环境
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

#### 2、莎士比亚字符级数据集

```bash
cd nanogpt
# 下载数据集到data文件夹下面
python data/shakespeare_char/prepare.py
# 开始训练
python train.py config/train_shakespeare_char.py
# 开始测试：生成推理结果
python sample.py --out_dir=out-shakespeare-char
```

#### 3、训练结果

```bash
step 0: train loss 4.2874, val loss 4.2823
iter 0: loss 4.2654, time 13500.54ms, mfu -100.00%
iter 10: loss 3.1462, time 13.86ms, mfu 26.88%
iter 20: loss 2.7321, time 13.74ms, mfu 26.90%
iter 30: loss 2.6184, time 13.75ms, mfu 26.92%
iter 40: loss 2.5757, time 13.75ms, mfu 26.94%
iter 50: loss 2.5249, time 13.75ms, mfu 26.96%
iter 60: loss 2.5143, time 13.77ms, mfu 26.97%
iter 70: loss 2.4947, time 13.77ms, mfu 26.98%
iter 80: loss 2.4935, time 13.76ms, mfu 26.99%
iter 90: loss 2.4691, time 13.76ms, mfu 27.00%
```

#### 4、测试结果

```bash
ANGELO:
And I will be ready to a serious business,
So many times shall their wisdoms. What think'st thou, my face?

AUFIDIUS:
You shall be there, my lord.

CORIOLANUS:
I content that I shall not die nor only.

CORIOLANUS:
The consul consul conspiracts him yet:
The present thereof whom I should that move him when I in hear
An o'erwhelm he is content.

CORIOLANUS:
Mantly forth on; I know not when I would think
The ground.

MENENIUS:
Here is a rose man a gentleman to the archbricted
Make your voic
```

### 红楼梦数据集实验

#### 1、创建数据集

数据集都保存在data下面，所以需要在data下面创建一个hongloumeng_char_local文件夹，然后创建一个input.txt文件存储红楼梦小说的文本内容，然后创建一个prepare.py文件用来进行分词和划分

```bash
cd nanogpt/data
# 创建一个红楼梦文件夹，用于存放数据
mkdir hongloumeng_char_local
# 在红楼梦文件夹里创建两个文件
cd hongloumeng_char_local
# 创建一个input的文本文件，里面放入红楼梦任意章节的内容
touch input.txt
# 创建一个python文件
vi prepare.py
```

在 data/hongloumeng_char_local/prepare.py中写入下面的内容，执行后会在data/hongloumeng_char_local目录中生成三个文件

- ‘train.bin’ ： 训练集
- ‘val.bin’ ：测试集
- ‘meta.pkl’ ： “汉字/字符到整数值的映射关系”

```python
# data/hongloumeng_char_local/prepare.py

"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
```

#### 2、创建训练配置文件

先在config中创建一个train_hongloumeng_char_local.py文件

```bash
cd nanogpt
vi config/train_hongloumeng_char_local.py
```

写入下面的内容

```python
# config/train_hongloumeng_char_local.py

# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-hongloumeng-char-local'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'hongloumeng-char-local'
wandb_run_name = 'mini-gpt'

dataset = 'hongloumeng_char_local'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
```

#### 3、训练和测试

```bash
cd nanogpt
# 数据准备
python data/hongloumeng_char_local/prepare.py
# 训练
python train.py config/train_hongloumeng_char_local.py
# 测试
python sample.py --out_dir=out-hongloumeng-char-local/
```

#### 4、训练结果

```shell
step 0: train loss 8.4504, val loss 8.4506
iter 0: loss 8.4499, time 16598.15ms, mfu -100.00%
iter 10: loss 7.6026, time 29.95ms, mfu 14.21%
iter 20: loss 6.8748, time 30.21ms, mfu 14.19%
iter 30: loss 6.0137, time 13.99ms, mfu 15.82%
iter 40: loss 5.6637, time 30.12ms, mfu 15.65%
... ...
iter 4890: loss 1.4662, time 30.37ms, mfu 16.18%
iter 4900: loss 1.4400, time 15.54ms, mfu 17.30%
iter 4910: loss 1.4483, time 26.49ms, mfu 17.17%
iter 4920: loss 1.4610, time 30.31ms, mfu 16.86%
iter 4930: loss 1.4936, time 25.88ms, mfu 16.82%
iter 4940: loss 1.4505, time 30.22ms, mfu 16.54%
iter 4950: loss 1.4753, time 30.46ms, mfu 16.29%
iter 4960: loss 1.4595, time 30.15ms, mfu 16.07%
iter 4970: loss 1.4530, time 30.21ms, mfu 15.87%
iter 4980: loss 1.4654, time 30.35ms, mfu 15.69%
iter 4990: loss 1.4377, time 30.32ms, mfu 15.52%
```

#### 5、测试结果

```shell
---------------

宝钗笑道：“二爷，你也太和姑娘说话。”宝钗笑道：“你又说了。我今儿又病了，怎么好呢？”宝钗笑道：“走罢！”黛玉笑道：“你别慌我，我告诉你，我也不依。”宝钗道：“可不是，不过是北静王妃，北静王得病，实在没有什么话说。”宝钗笑道：“你们只管说话，我就不用说话了。”宝钗笑道：“你们不必说话，你们索性说说了。”宝钗道：“我自然明说了，你们宝兄弟明日再和这话，我叫你打发出去。”

宝钗笑道：“你们都去。”宝钗笑道：“既这样说，你再要是个话，你也要弄个谜儿。”宝钗笑道：“你只要猜着了。我也猜着了，就把‘五’，那三个韵。”黛玉道：“我到底分上个社长得好。”说着便饮了。宝钗笑道：“这个个社还要‘蘅芜君’。”香菱笑道：“我也古人古怪，‘春’忆‘，‘，鹤’韵’犹觉‘。”宝钗道：“倒是‘蘅芜苑’。”湘云道：“更妙，‘月’忆秋‘，便得很湍’，‘秋’二字。‘杏花忆菊’忆菊’，‘忆菊’二字，其菊’不觉染‘，尽忆‘钗’忆蕉菊’。分菊如钗也‘。忆菊时亦关，所为菊，也觉得不能尽己无韵。“后事有菊想来，故不得有访，不觉菊花不菊却意，菊影花又合而菊有，依菊也难忆菊为枝菊。不既寻忆故事，菊便是《菊看菊花，菊花，菊蕉叶菊花菊花
---------------
```

