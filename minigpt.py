"""
词级 Mini-GPT 训练脚本
使用 tiktoken (GPT-2 tokenizer)

相比字符级的改进：
1. 更高效的编码（1个单词 ≈ 1个token，而不是多个字符）
2. 更大的有效上下文（256 tokens ≈ 150-200 个单词）
3. 更好的语义理解
4. 加入了学习率调度、梯度裁剪等优化
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import math
import time
import os
import urllib.request

# ============================================================================
# 配置
# ============================================================================

class Config:
    # 数据
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    
    # Tokenizer
    tokenizer_name = 'gpt2'  # 使用 GPT-2 的 tokenizer
    
    # 模型
    vocab_size = 50257  # GPT-2 tokenizer 的词表大小
    n_layer = 8         # 增加层数（从 6 到 8）
    n_head = 8          # 增加注意力头（从 6 到 8）
    n_embd = 512        # 增加维度（从 384 到 512）
    dropout = 0.2
    block_size = 256    # 上下文长度（现在是 256 个词，而不是 256 个字符！）
    
    # 训练
    batch_size = 32     # 减小 batch（因为模型更大了）
    learning_rate = 3e-4
    max_iters = 10000   # 增加训练步数
    eval_interval = 500
    eval_iters = 100
    
    # 学习率调度
    warmup_iters = 500
    min_lr = 3e-5
    
    # 系统
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    compile_model = False  # PyTorch 2.0+ 可以开启

config = Config()

print("=" * 80)
print("词级 Mini-GPT 训练")
print("=" * 80)
print(f"\n设备: {config.device}")
print(f"Tokenizer: {config.tokenizer_name}")
print(f"词表大小: {config.vocab_size:,}")
print(f"上下文长度: {config.block_size} tokens")
print(f"模型维度: {config.n_embd}")
print(f"层数: {config.n_layer}")
print(f"注意力头数: {config.n_head}")


# ============================================================================
# 数据准备
# ============================================================================

print("\n" + "=" * 80)
print("第一步：准备数据和 Tokenizer")
print("=" * 80)

# 下载数据
data_path = 'shakespeare.txt'
if not os.path.exists(data_path):
    print(f"下载数据从 {config.data_url}...")
    urllib.request.urlretrieve(config.data_url, data_path)
    print("下载完成！")
else:
    print("数据已存在")

# 读取数据
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"\n数据统计:")
print(f"  总字符数: {len(text):,}")

# 初始化 tiktoken tokenizer
print(f"\n加载 tiktoken tokenizer: {config.tokenizer_name}")
try:
    enc = tiktoken.get_encoding(config.tokenizer_name)
    print("✓ Tokenizer 加载成功！")
except Exception as e:
    print(f"❌ 加载失败: {e}")
    print("尝试安装: pip install tiktoken")
    exit(1)

# 编码整个文本
print("\n编码文本...")
tokens = enc.encode(text)
print(f"✓ 编码完成！")
print(f"  Token 数量: {len(tokens):,}")
print(f"  压缩比: {len(text)/len(tokens):.2f} 字符/token")

# 测试编码/解码
test_text = "Hello, world! How are you?"
test_tokens = enc.encode(test_text)
test_decoded = enc.decode(test_tokens)

print(f"\n编码测试:")
print(f"  原文: {test_text}")
print(f"  Tokens: {test_tokens}")
print(f"  Token 数: {len(test_tokens)}")
print(f"  解码: {test_decoded}")

# 显示一些 token 的文本
print(f"\n前 20 个 tokens 对应的文本:")
for i, token in enumerate(tokens[:20]):
    token_text = enc.decode([token])
    # 处理特殊字符显示
    if token_text == '\n':
        token_text = '\\n'
    elif token_text == ' ':
        token_text = '·'  # 用中点表示空格
    print(f"  {i:2d}. {token:5d} → '{token_text}'")

# 训练/验证集划分
data = torch.tensor(tokens, dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"\n数据划分:")
print(f"  训练集: {len(train_data):,} tokens")
print(f"  验证集: {len(val_data):,} tokens")


# 数据加载器
def get_batch(split):
    """获取一个批次的数据"""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y


# 测试数据加载
xb, yb = get_batch('train')
print(f"\n批次数据:")
print(f"  输入形状: {xb.shape}")
print(f"  标签形状: {yb.shape}")
print(f"  第一个样本的前10个token:")
print(f"    输入 tokens: {xb[0][:10].tolist()}")
print(f"    标签 tokens: {yb[0][:10].tolist()}")
print(f"  解码后:")
print(f"    输入: {enc.decode(xb[0][:10].tolist())}")
print(f"    标签: {enc.decode(yb[0][:10].tolist())}")


# ============================================================================
# 模型定义（和字符级相同，但参数更大）
# ============================================================================

print("\n" + "=" * 80)
print("第二步：定义模型")
print("=" * 80)

class Head(nn.Module):
    """单个注意力头"""
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer 块（Pre-LN）"""
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """GPT 语言模型"""
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config.n_embd, config.n_head) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """生成文本（带 temperature 和 top-k）"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Top-k 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# 创建模型
model = GPTLanguageModel()
model = model.to(config.device)

# 统计参数
total_params = sum(p.numel() for p in model.parameters())
print(f"\n模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")

# 可选：编译模型（PyTorch 2.0+）
if config.compile_model and hasattr(torch, 'compile'):
    print("编译模型...")
    model = torch.compile(model)

# 测试前向传播
xb, yb = get_batch('train')
logits, loss = model(xb, yb)
print(f"\n前向传播测试:")
print(f"  Logits 形状: {logits.shape}")
print(f"  初始 Loss: {loss.item():.4f}")
print(f"  预期初始 Loss: {math.log(config.vocab_size):.4f} (随机猜测)")


# ============================================================================
# 训练
# ============================================================================

print("\n" + "=" * 80)
print("第三步：开始训练")
print("=" * 80)

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


# 学习率调度
def get_lr(iter):
    """Warmup + Cosine Decay"""
    # Warmup
    if iter < config.warmup_iters:
        return config.learning_rate * iter / config.warmup_iters
    # 已完成
    if iter > config.max_iters:
        return config.min_lr
    # Cosine decay
    decay_ratio = (iter - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


@torch.no_grad()
def estimate_loss():
    """估计训练/验证损失"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# 训练循环
print("\n开始训练...")
print(f"配置: {config.max_iters} 步, batch_size={config.batch_size}, lr={config.learning_rate}")
print(f"学习率调度: warmup={config.warmup_iters}, min_lr={config.min_lr}")
start_time = time.time()

train_losses = []
val_losses = []
lrs = []

for iter in range(config.max_iters):
    
    # 评估
    if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        lr = get_lr(iter)
        
        print(f"step {iter:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e} | time {elapsed:.1f}s")
        
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        lrs.append(lr)
    
    # 更新学习率
    lr = get_lr(iter)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # 获取批次
    xb, yb = get_batch('train')
    
    # 前向传播
    logits, loss = model(xb, yb)
    
    # 反向传播
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()

total_time = time.time() - start_time
print(f"\n训练完成！总时间: {total_time/60:.1f} 分钟")


# ============================================================================
# 生成文本
# ============================================================================

print("\n" + "=" * 80)
print("第四步：生成文本")
print("=" * 80)

model.eval()

def generate_text(prompt, max_new_tokens=200, temperature=0.8, top_k=10):
    """生成文本的辅助函数"""
    tokens = enc.encode(prompt)
    context = torch.tensor(tokens, dtype=torch.long, device=config.device).unsqueeze(0)
    generated = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return enc.decode(generated[0].tolist())


# 测试不同的提示
prompts = [
    "ROMEO:",
    "To be or not to be",
    "First Citizen:\n",
    "JULIET:\n"
]

print("\n生成示例（temperature=0.8, top_k=10）:")
for prompt in prompts:
    print(f"\n{'='*70}")
    print(f"提示: '{prompt}'")
    print(f"{'='*70}")
    text = generate_text(prompt, max_new_tokens=150, temperature=0.8, top_k=10)
    print(text)


# 对比不同的 temperature
print("\n\n" + "=" * 80)
print("Temperature 对比")
print("=" * 80)

prompt = "ROMEO:"
temps = [0.5, 0.8, 1.0, 1.2]

for temp in temps:
    print(f"\n{'='*70}")
    print(f"Temperature = {temp}")
    print(f"{'='*70}")
    text = generate_text(prompt, max_new_tokens=100, temperature=temp, top_k=10)
    print(text)


# ============================================================================
# 保存模型
# ============================================================================

print("\n" + "=" * 80)
print("第五步：保存模型")
print("=" * 80)

checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'config': config,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'lrs': lrs,
    'tokenizer': config.tokenizer_name,
}

torch.save(checkpoint, 'word_level_gpt.pt')
print("✓ 模型已保存到 word_level_gpt.pt")


# ============================================================================
# 可视化训练过程
# ============================================================================

print("\n" + "=" * 80)
print("第六步：可视化训练")
print("=" * 80)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Loss 曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
steps = [i * config.eval_interval for i in range(len(train_losses))]
plt.plot(steps, train_losses, label='Train Loss', linewidth=2)
plt.plot(steps, val_losses, label='Val Loss', linewidth=2)
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)

# 学习率曲线
plt.subplot(1, 2, 2)
plt.plot(steps, lrs, linewidth=2, color='green')
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print("✓ 训练曲线已保存到 training_curves.png")


# ============================================================================
# 总结
# ============================================================================

print("\n" + "=" * 80)
print("训练总结")
print("=" * 80)

print(f"\n模型配置:")
print(f"  参数量: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"  层数: {config.n_layer}")
print(f"  注意力头数: {config.n_head}")
print(f"  Embedding 维度: {config.n_embd}")
print(f"  上下文长度: {config.block_size} tokens")
print(f"  词表大小: {config.vocab_size:,}")

print(f"\n训练配置:")
print(f"  训练步数: {config.max_iters}")
print(f"  批大小: {config.batch_size}")
print(f"  初始学习率: {config.learning_rate}")
print(f"  最小学习率: {config.min_lr}")
print(f"  Warmup 步数: {config.warmup_iters}")
print(f"  训练时间: {total_time/60:.1f} 分钟")

print(f"\n最终性能:")
final_losses = estimate_loss()
print(f"  训练集 Loss: {final_losses['train']:.4f}")
print(f"  验证集 Loss: {final_losses['val']:.4f}")

# 计算困惑度
train_perplexity = math.exp(final_losses['train'])
val_perplexity = math.exp(final_losses['val'])
print(f"  训练集困惑度: {train_perplexity:.2f}")
print(f"  验证集困惑度: {val_perplexity:.2f}")

print("\n🎉 词级 GPT 训练完成！")

print("\n相比字符级的改进:")
print("  ✓ 更高效的编码（~4x 压缩）")
print("  ✓ 更长的有效上下文")
print("  ✓ 更好的语义理解")
print("  ✓ 添加了学习率调度")
print("  ✓ 添加了梯度裁剪")
print("  ✓ 添加了 top-k 采样")
print("  ✓ 更大的模型（30M vs 10M）")

print("\n下一步建议:")
print("  1. 查看 training_curves.png 了解训练过程")
print("  2. 用不同的提示词测试生成质量")
print("  3. 如果效果不理想，可以:")
print("     - 训练更多步数（20000+）")
print("     - 在更大数据集上训练（WikiText-2, OpenWebText）")
print("     - 增大模型（12 层，768 维）")
print("  4. 进入下一阶段：多 GPU 训练、混合精度")