import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 1. 强制使用 CPU 环境（Mac Intel 关键设置） =====
# 设置镜像源（如果环境变量没设，这里作为保底）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 显式指定设备为 CPU（禁用任何 GPU/MPS 尝试）
device = torch.device("cpu")

# ===== 2. 模型参数配置 =====
model_name = "gpt2"  # 推荐测试模型：124M 参数，兼容性好
local_dir = "./models/" + model_name  # 本地缓存目录

try:
    # ===== 3. 加载 Tokenizer =====
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=local_dir
    )
    # 如果 tokenizer 没有 pad_token，设置一个（防止某些模型报错）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ===== 4. 加载 Model（关键配置） =====
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=local_dir,
        device_map="cpu",           # 强制所有模块在 CPU
        torch_dtype=torch.float32,  # Intel CPU 建议用 float32（比 float16 稳定）
        low_cpu_mem_usage=True,     # 减少内存峰值
        trust_remote_code=False     # 除非模型需要自定义代码，否则关闭
    )
    # 将模型移动到 CPU 设备（双重保险）
    model.to(device)

except Exception as e:
    print(f"[错误] 模型加载失败: {e}")
    exit(1)

# ===== 5. 调试信息输出 =====
print(f"✅ 模型加载成功！")
print(f"   模型设备: {model.device}")
print(f"   模型 dtype: {model.dtype}")

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"   总参数量: {total_params / 1e6:.1f} M")

# ===== 6. 简单推理测试（Debug 核心） =====
input_text = "The quick brown fox"
inputs = tokenizer(input_text, return_tensors="pt")

# 将输入数据也移动到 CPU
inputs = {k: v.to(device) for k, v in inputs.items()}

# 使用 no_grad 节省内存
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

print(f"✅ 推理完成！")
print(f"   输入文本: {input_text}")
print(f"   Logits 形状: {logits.shape}")  # 应为 [1, seq_len, vocab_size]

# 可选：查看前几个预测 token
predicted_token_ids = torch.argmax(logits, dim=-1)[0]
predicted_tokens = tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
print(f"   预测续写: {predicted_tokens}")