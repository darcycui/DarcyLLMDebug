import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 1. 强制使用 CPU 环境 =====
device = torch.device("cpu")

# ===== 2. 模型参数配置 - 修改为本地路径 =====
# 指定你手动下载模型文件所在的本地目录
local_model_dir = "./models/gpt2_local"  # 请确保此路径包含所有必需文件

# 可选：验证本地目录是否存在必需文件
required_files = ["config.json"]
# 检查权重文件（支持 .bin 或 .safetensors 格式）
if not os.path.exists(os.path.join(local_model_dir, "pytorch_model.bin")) and \
   not os.path.exists(os.path.join(local_model_dir, "model.safetensors")):
    print("❌ 错误：本地目录中未找到模型权重文件 (pytorch_model.bin 或 model.safetensors)")
    exit(1)

try:
    # ===== 3. 加载 Tokenizer - 从本地 =====
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_dir,           # 关键修改：使用本地路径
        local_files_only=True      # 关键参数：强制本地加载
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer 从本地加载成功")

    # ===== 4. 加载 Model - 从本地，并自动处理格式 =====
    model = AutoModelForCausalLM.from_pretrained(
        local_model_dir,           # 关键修改：使用本地路径
        local_files_only=True,     # 关键参数：强制本地加载
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=False
    )
    model.to(device)
    print("✅ 模型从本地加载成功")

except Exception as e:
    print(f"[错误] 本地模型加载失败: {e}")
    exit(1)

# ===== 5. 后续的调试与推理代码保持不变 =====
# ... (你原文档中第5、6部分的代码可以完全保留)
print(f"✅ 模型加载成功！")
print(f"   模型设备: {model.device}")
print(f"   模型 dtype: {model.dtype}")

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"   总参数量: {total_params / 1e6:.1f} M")

# ===== 6. 简单推理测试 =====
input_text = "The quick brown fox"
inputs = tokenizer(input_text, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

print(f"✅ 推理完成！")
print(f"   输入文本: {input_text}")
print(f"   Logits 形状: {logits.shape}")

# 可选：查看前几个预测 token
predicted_token_ids = torch.argmax(logits, dim=-1)[0]
predicted_tokens = tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
print(f"   预测续写: {predicted_tokens}")