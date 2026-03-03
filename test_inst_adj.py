import torch
from data.dataset import CADSynth

# 设置数据集路径（根据你的实际情况修改）
dataset_path = "D:/Code/BrepMFR/dataset"  # 你的数据集路径

# 加载数据集（使用训练集，不进行随机旋转）
dataset = CADSynth(root_dir=dataset_path, split="train", random_rotate=False, num_class=25)

# 获取 DataLoader，batch_size 设为 2 便于观察
loader = dataset.get_dataloader(batch_size=2, shuffle=False, num_workers=0)

# 取第一个 batch
for batch in loader:
    print("Batch keys:", batch.keys())
    if "inst_adj" in batch:
        print("inst_adj shape:", batch["inst_adj"].shape)  # 应为 [2, max_nodes, max_nodes]
        # 打印第一个样本的实例邻接矩阵（只显示前10行/列，避免太大）
        inst_adj_0 = batch["inst_adj"][0]
        print("First sample inst_adj (first 10 rows and columns):")
        print(inst_adj_0[:10, :10])
        # 检查对角线是否全为0（同一面不应与自己相连）
        diag = torch.diag(inst_adj_0)
        print("Diagonal elements (should be all 0):", diag[:10])
        # 检查是否有非0/1的值
        unique_vals = torch.unique(inst_adj_0)
        print("Unique values in inst_adj:", unique_vals)
    else:
        print("inst_adj not found in batch!")
    break  # 只测一个 batch