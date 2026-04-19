# test_pytorch.py
import torch
import transformers
import numpy as np

print(f"PyTorch版本: {torch.__version__}")
print(f"Transformers版本: {transformers.__version__}")
print(f"NumPy版本: {np.__version__}")
print(f"PyTorch是否可用: {torch.__version__.startswith('2.')}")