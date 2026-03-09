import torch
import numpy as np
import os
from data.dataset import CADSynth
from iag_transformer.visual_iag_transformer_model import IAGTransformerModel

# ==================== 配置区域 ====================
# 数据集路径
DATASET_PATH = "D:/Code/BrepMFR/dataset"
# 改进版模型的checkpoint路径（请替换为您的实际路径）
CHECKPOINT_PATH = r"D:\Code\BrepMFR\results\IAG_1pct_lambda001_s\0303\train_epoch100_105138\best.ckpt"
# 是否只跑前N个样本（0表示全跑，但测试集有1万，建议先跑小批量测试）
MAX_SAMPLES = 10000   # 例如先跑500个样本
# 输出结果保存文件
OUTPUT_FILE = "attention_rank.txt"
# 使用的GPU编号（-1表示CPU）
DEVICE_ID = 0
# =================================================

device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() and DEVICE_ID >= 0 else "cpu")

# 加载数据集
dataset = CADSynth(root_dir=DATASET_PATH, split="test", sample_ratio=1.0)
print(f"测试集大小: {len(dataset)}")

# 加载模型
model = IAGTransformerModel.load_from_checkpoint(CHECKPOINT_PATH)
model = model.to(device)
model.eval()
print("模型加载完成")

# 用于存储结果的列表
results = []  # 每个元素为 (sample_id, ratio, sample_index)

# 限制样本数量
num_samples = min(MAX_SAMPLES, len(dataset)) if MAX_SAMPLES > 0 else len(dataset)
print(f"将处理 {num_samples} 个样本...")

for idx in range(num_samples):
    if idx % 50 == 0:
        print(f"处理第 {idx} 个样本...")
    try:
        sample = dataset[idx]
        batch = dataset._collate([sample])
        # 将batch移到设备
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            elif k == 'graph':
                # dgl图需要单独处理
                batch[k] = batch[k].to(device)

        # 前向传播获取注意力权重
        with torch.no_grad():
            # 注意：visual_iag_encoder.py 中的 forward 需要支持 return_attn=True
            # 您之前已经修改了 visual_iag_encoder.py，应该没问题
            inner_states, graph_rep, attn_weights = model.brep_encoder(
                batch,
                last_state_only=True,
                inst_adj=batch.get('inst_adj'),
                return_attn=True
            )

        # 取第一个样本，第一个头，去掉虚拟节点
        attn = attn_weights[0, 1:, 1:].cpu().numpy()  # [num_faces, num_faces]

        # 获取该样本的伪实例标签
        instance_ids = batch['instance_ids'][0].cpu().numpy()  # [max_nodes]
        valid_mask = (instance_ids != -1)  # 有效节点
        valid_ids = instance_ids[valid_mask]
        num_valid = len(valid_ids)

        # 如果有效节点数太少，跳过（例如少于5个面）
        if num_valid < 5:
            continue

        # 截取注意力矩阵到有效节点部分（因为 padding 节点我们不需要）
        attn_valid = attn[:num_valid, :num_valid]

        # 计算实例内和实例间的平均注意力
        intra_weights = []
        inter_weights = []
        for i in range(num_valid):
            for j in range(num_valid):
                if i == j:
                    continue  # 忽略自注意力
                w = attn_valid[i, j]
                if valid_ids[i] == valid_ids[j]:
                    intra_weights.append(w)
                else:
                    inter_weights.append(w)

        if len(intra_weights) == 0 or len(inter_weights) == 0:
            continue  # 避免除零

        avg_intra = np.mean(intra_weights)
        avg_inter = np.mean(inter_weights)
        ratio = avg_intra / (avg_inter + 1e-8)  # 防止分母为0

        # 获取样本的真实ID（文件名中的数字）
        sample_id = sample.data_id  # 假设 PYGGraph 中有 data_id 属性

        results.append((sample_id, ratio, idx))

    except Exception as e:
        print(f"处理样本 {idx} 时出错: {e}")
        continue

# 按比值从高到低排序
results.sort(key=lambda x: x[1], reverse=True)

# 输出结果
print("\n=== 前20个样本 (按实例内/实例间注意力比值排序) ===")
print("排名\t样本ID\t比值\t索引")
with open(OUTPUT_FILE, 'w') as f:
    f.write("rank\tsample_id\tratio\tindex\n")
    for rank, (sid, ratio, idx) in enumerate(results[:20]):
        print(f"{rank+1}\t{sid}\t{ratio:.4f}\t{idx}")
        f.write(f"{rank+1}\t{sid}\t{ratio:.4f}\t{idx}\n")

print(f"\n结果已保存至 {OUTPUT_FILE}")