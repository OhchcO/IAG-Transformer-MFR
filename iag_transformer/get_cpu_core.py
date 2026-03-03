import torch
import multiprocessing

# 获取CPU核心数
cpu_count = multiprocessing.cpu_count()
print(f"CPU核心数: {cpu_count}")