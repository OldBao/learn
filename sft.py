"""
Supervised Fine-Tuning (SFT) 完整训练脚本
从预训练模型到对话模型

输入：预训练的 GPT 模型
输出：能遵循指令的对话模型
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import json
import time
import os
import urllib.request
from pathlib import Path

# ============================================================================
# 配置
# ============================================================================

class Config:
    # 模型路径
    pretrained_model_path = 'word_level_gpt.pt'  # 你训练的预训练模型
    
    # SFT 数据
    # 我们用 Alpaca 数据集（52k 高质量指令对）
    alpaca_url = 'https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json'
    data_path = 'alpaca_data.json'
    
    # Tokenizer
    tokenizer_name = 'gpt2'
    
    # 训练
    batch_size = 8        # SFT 通常用更小的 batch
    learning_rate = 5e-5  # 比预训练小 10x
    max_iters = 3000      # SFT 不需要很多步
    eval_interval = 300
    eval_iters = 50
    
    # 特殊 token（用于标记对话结构）
    instruction_start = "\n### Instruction:\n"
    input_start = "\n### Input:\n"
    response_start = "\n### Response:\n"
    
    # 系统
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    save_interval = 1000

config = Config()

print("=" * 80)
print("Supervised Fine-Tuning (SFT) 训练")
print("=" * 80)
print(f"\n设备: {config.device}")
print(f"预训练模型: {config.pretrained_model_path}")
print(f"数据集: Alpaca (52k 指令对)")


# ============================================================================
# 下载和准备数据
# ============================================================================

print("\n" + "=" * 80)
print("第一步：准备 SFT 数据")
print("=" * 80)

# 下载 Alpaca 数据
if not os.path.exists(config.data_path):
    print(f"\n下载 Alpaca 数据集...")
    try:
        urllib.request.urlretrieve(config.alpaca_url, config.data_path)
        print("✓ 下载完成！")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n手动下载方法：")
        print(f"1. 访问: {config.alpaca_url}")
        print(f"2. 保存为: {config.data_path}")
        print("3. 重新运行此脚本")
        exit(1)
else:
    print("✓ 数据已存在")

# 加载数据
print("\n加载数据...")
with open(config.data_path, 'r', encoding='utf-8') as f:
    alpaca_data = json.load(f)

print(f"✓ 加载了 {len(alpaca_data)} 条训练样本")

# 查看数据格式
print("\n数据格式示例:")
sample = alpaca_data[0]
print(f"  Instruction: {sample['instruction'][:60]}...")
print(f"  Input: {sample['input'][:60] if sample['input'] else '(empty)'}...")
print(f"  Output: {sample['output'][:60]}...")


# 初始化 tokenizer
print("\n初始化 tokenizer...")
enc = tiktoken.get_encoding(config.tokenizer_name)
print("✓ Tokenizer 加载成功")


# ============================================================================
# 数据处理：关键！
# ============================================================================

print("\n" + "=" * 80)
print("第二步：数据处理（关键步骤）")
print("=" * 80)

print("""
SFT 的核心：只在"回答"部分计算损失！

格式化对话：
  ### Instruction:
  [用户指令]
  
  ### Input:
  [可选的输入]
  
  ### Response:
  [模型回答]  ← 只在这部分计算损失！
""")


def format_instruction(sample):
    """
    将 Alpaca 样本格式化为训练文本
    """
    instruction = sample['instruction']
    input_text = sample['input']
    output = sample['output']
    
    # 构建完整对话
    if input_text:
        prompt = (
            f"{config.instruction_start}{instruction}"
            f"{config.input_start}{input_text}"
            f"{config.response_start}{output}"
        )
    else:
        prompt = (
            f"{config.instruction_start}{instruction}"
            f"{config.response_start}{output}"
        )
    
    return prompt


def prepare_sft_sample(sample, enc, max_length=512):
    """
    准备 SFT 训练样本
    
    关键：返回 input_ids 和 labels
    labels 中非回答部分标记为 -100（损失函数会忽略）
    """
    # 格式化文本
    full_text = format_instruction(sample)
    
    # 编码
    tokens = enc.encode(full_text)
    
    # 截断（如果太长）
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    
    # 找到 Response 的起始位置
    response_start_text = config.response_start
    response_start_tokens = enc.encode(response_start_text)
    
    # 在 tokens 中找到 response_start 的位置
    response_start_idx = None
    for i in range(len(tokens) - len(response_start_tokens) + 1):
        if tokens[i:i+len(response_start_tokens)] == response_start_tokens:
            response_start_idx = i + len(response_start_tokens)
            break
    
    if response_start_idx is None:
        # 如果找不到（数据格式问题），跳过这个样本
        return None
    
    # 创建 labels
    labels = tokens.copy()
    
    # 关键：Response 之前的部分都标记为 -100
    for i in range(response_start_idx):
        labels[i] = -100
    
    return {
        'input_ids': tokens,
        'labels': labels,
        'length': len(tokens)
    }


# 处理所有数据
print("\n处理训练数据...")
processed_data = []

for i, sample in enumerate(alpaca_data):
    processed = prepare_sft_sample(sample, enc)
    if processed is not None:
        processed_data.append(processed)
    
    if (i + 1) % 10000 == 0:
        print(f"  处理了 {i+1}/{len(alpaca_data)} 个样本...")

print(f"✓ 成功处理 {len(processed_data)} 个样本")

# 显示处理后的样本
print("\n处理后的样本示例:")
sample = processed_data[0]
print(f"  Input IDs 长度: {len(sample['input_ids'])}")
print(f"  Labels 长度: {len(sample['labels'])}")
print(f"\n  完整文本:")
print(f"  {enc.decode(sample['input_ids'])}")
print(f"\n  Labels 中 -100 的位置（这些位置不计算损失）:")
mask_positions = [i for i, label in enumerate(sample['labels']) if label == -100]
print(f"  前 20 个位置的 labels: {sample['labels'][:20]}")
print(f"  总共 {len(mask_positions)} 个位置被 mask（不计算损失）")

# 划分训练/验证集
train_size = int(0.95 * len(processed_data))
train_data = processed_data[:train_size]
val_data = processed_data[train_size:]

print(f"\n数据划分:")
print(f"  训练集: {len(train_data)} 样本")
print(f"  验证集: {len(val_data)} 样本")


# ============================================================================
# 数据加载器
# ============================================================================

def get_batch(split, batch_size=config.batch_size):
    """
    获取一个批次的 SFT 数据
    """
    data = train_data if split == 'train' else val_data
    
    # 随机选择样本
    indices = torch.randint(len(data), (batch_size,))
    
    # 找到最大长度（用于 padding）
    max_len = max(data[i]['length'] for i in indices)
    
    # 准备 batch
    input_ids = []
    labels = []
    
    for idx in indices:
        sample = data[idx]
        
        # Padding
        pad_len = max_len - sample['length']
        
        input_id = sample['input_ids'] + [enc.eot_token] * pad_len
        label = sample['labels'] + [-100] * pad_len  # padding 也不计算损失
        
        input_ids.append(input_id)
        labels.append(label)
    
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=config.device)
    labels = torch.tensor(labels, dtype=torch.long, device=config.device)
    
    return input_ids, labels


# 测试数据加载
print("\n测试数据加载...")
xb, yb = get_batch('train', batch_size=2)
print(f"  Batch input_ids shape: {xb.shape}")
print(f"  Batch labels shape: {yb.shape}")
print(f"  示例：第一个样本的前 20 个 labels: {yb[0][:20].tolist()}")
print(f"  → 注意 -100 的位置（这些是指令部分，不计算损失）")


# ============================================================================
# 加载预训练模型
# ============================================================================

print("\n" + "=" * 80)
print("第三步：加载预训练模型")
print("=" * 80)

try:
    print(f"\n加载模型从 {config.pretrained_model_path}...")
    checkpoint = torch.load(config.pretrained_model_path, map_location=config.device)
    
    # 重新导入模型定义（需要和训练时一致）
    from train_word_level_gpt import GPTLanguageModel, Config as PretrainConfig
    
    pretrain_config = checkpoint['config']
    
    # 创建模型
    model = GPTLanguageModel()
    model.load_state_dict(checkpoint['model'])
    model = model.to(config.device)
    model.train()
    
    print("✓ 模型加载成功！")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
except FileNotFoundError:
    print(f"❌ 找不到预训练模型: {config.pretrained_model_path}")
    print("\n请先运行 train_word_level_gpt.py 完成预训练")
    exit(1)
except Exception as e:
    print(f"❌ 加载模型失败: {e}")
    exit(1)


# ============================================================================
# SFT 训练
# ============================================================================

print("\n" + "=" * 80)
print("第四步：SFT 训练")
print("=" * 80)

# 优化器（学习率比预训练小）
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

print(f"\n训练配置:")
print(f"  学习率: {config.learning_rate} (比预训练小 5-10x)")
print(f"  批大小: {config.batch_size}")
print(f"  训练步数: {config.max_iters}")
print(f"  评估间隔: {config.eval_interval}")


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
print("\n开始 SFT 训练...")
start_time = time.time()

train_losses = []
val_losses = []

for iter in range(config.max_iters):
    
    # 评估
    if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        
        print(f"step {iter:4d} | train {losses['train']:.4f} | val {losses['val']:.4f} | time {elapsed:.1f}s")
        
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
    
    # 获取批次
    xb, yb = get_batch('train')
    
    # 前向传播（关键：labels 中 -100 的位置会被自动忽略）
    logits, loss = model(xb, yb)
    
    # 反向传播
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # 保存检查点
    if (iter + 1) % config.save_interval == 0:
        checkpoint_path = f'sft_checkpoint_{iter+1}.pt'
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iter': iter,
            'config': pretrain_config,
        }, checkpoint_path)
        print(f"  → 保存检查点: {checkpoint_path}")

total_time = time.time() - start_time
print(f"\n✓ SFT 训练完成！总时间: {total_time/60:.1f} 分钟")


# ============================================================================
# 测试对话能力
# ============================================================================

print("\n" + "=" * 80)
print("第五步：测试对话能力")
print("=" * 80)

model.eval()

def generate_response(instruction, input_text="", max_new_tokens=150, temperature=0.7):
    """
    生成回答
    """
    # 构建 prompt
    if input_text:
        prompt = (
            f"{config.instruction_start}{instruction}"
            f"{config.input_start}{input_text}"
            f"{config.response_start}"
        )
    else:
        prompt = (
            f"{config.instruction_start}{instruction}"
            f"{config.response_start}"
        )
    
    # 编码
    tokens = enc.encode(prompt)
    context = torch.tensor(tokens, dtype=torch.long, device=config.device).unsqueeze(0)
    
    # 生成
    generated = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=50)
    
    # 解码
    full_text = enc.decode(generated[0].tolist())
    
    # 提取回答部分
    response_start = full_text.find(config.response_start)
    if response_start != -1:
        response = full_text[response_start + len(config.response_start):].strip()
    else:
        response = full_text
    
    return response


# 测试几个指令
test_instructions = [
    {
        "instruction": "What is the capital of France?",
        "input": ""
    },
    {
        "instruction": "Write a haiku about spring.",
        "input": ""
    },
    {
        "instruction": "Explain what photosynthesis is to a 5-year-old.",
        "input": ""
    },
    {
        "instruction": "Summarize the following text.",
        "input": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet."
    }
]

print("\n生成示例（温度 0.7）:")
for i, test in enumerate(test_instructions):
    print(f"\n{'='*70}")
    print(f"示例 {i+1}")
    print(f"{'='*70}")
    print(f"Instruction: {test['instruction']}")
    if test['input']:
        print(f"Input: {test['input']}")
    print(f"\nResponse:")
    response = generate_response(test['instruction'], test['input'], temperature=0.7)
    print(response)


# ============================================================================
# 保存最终模型
# ============================================================================

print("\n" + "=" * 80)
print("第六步：保存模型")
print("=" * 80)

final_checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'config': pretrain_config,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'sft_config': config,
}

torch.save(final_checkpoint, 'sft_model.pt')
print("✓ SFT 模型已保存到 sft_model.pt")


# ============================================================================
# 对比预训练 vs SFT
# ============================================================================

print("\n" + "=" * 80)
print("第七步：对比预训练 vs SFT")
print("=" * 80)

print("""
预训练模型（你训练的）:
  任务: Next Token Prediction
  行为: 续写文本
  
  输入: "What is the capital"
  输出: "of France? What is the capital of Germany?..."
        ↑ 只是续写，不回答问题

SFT 模型（刚训练的）:
  任务: Instruction Following
  行为: 回答问题
  
  输入: "What is the capital of France?"
  输出: "The capital of France is Paris."
        ↑ 真正回答了问题！

这就是 SFT 的魔力！
""")


# ============================================================================
# 总结
# ============================================================================

print("\n" + "=" * 80)
print("总结")
print("=" * 80)

print(f"""
SFT 训练完成！

配置:
  数据: Alpaca ({len(train_data)} 训练样本)
  训练步数: {config.max_iters}
  学习率: {config.learning_rate}
  训练时间: {total_time/60:.1f} 分钟

最终性能:
  训练集 Loss: {train_losses[-1]:.4f}
  验证集 Loss: {val_losses[-1]:.4f}

关键改进:
  ✓ 模型学会了对话格式
  ✓ 能遵循指令
  ✓ 回答更有针对性

下一步:
  1. 多测试不同的指令，评估质量
  2. 如果效果不理想，可以:
     - 训练更多 epochs
     - 调整学习率
     - 使用更多/更好的数据
  3. 进阶: 尝试 DPO 进一步优化

文件:
  - sft_model.pt: 最终模型
  - sft_checkpoint_*.pt: 中间检查点
""")

print("\n" + "=" * 80)
print("🎉 恭喜！你已经训练出一个对话模型！")
print("=" * 80)