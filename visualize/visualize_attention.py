import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from data.dataset import CADSynth
from iag_transformer import iag_transformer_model
from iag_transformer.visual_iag_transformer_model import IAGTransformerModel
from models.visual_brepseg_model import BrepSeg


# ==================== 配置区域 ====================
DATASET_PATH = "D:/Code/BrepMFR/dataset"
# 基线和改进版的checkpoint路径（请替换为实际路径）
BASELINE_CHECKPOINT = r"D:\Code\BrepMFR\baseline_result\BrepMFR_1pct\train_epoch100_174225\best.ckpt"
IMPROVED_CHECKPOINT = r"D:\Code\BrepMFR\results\IAG_1pct_lambda001_s\0303\train_epoch100_105138\best.ckpt"
TARGET_ID = 2964  # 筛选出的最佳样本ID
OUTPUT_IMAGE = "attention_comparison2964_0.pdf"  # 输出图像文件名
DEVICE_ID = 0
# =================================================

device = torch.device(f"cuda:{DEVICE_ID}" if torch.cuda.is_available() and DEVICE_ID >= 0 else "cpu")

# 加载数据集
dataset = CADSynth(root_dir=DATASET_PATH, split="test", sample_ratio=1.0)

# 根据ID查找样本索引
sample_idx = None
for i, fn in enumerate(dataset.file_paths):
    file_id = int(fn.stem)  # 假设文件名如 "1.bin"
    if file_id == TARGET_ID:
        sample_idx = i
        break
if sample_idx is None:
    raise ValueError(f"Sample with ID {TARGET_ID} not found.")

sample = dataset[sample_idx]
batch = dataset._collate([sample])
# 将batch移到设备
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        batch[k] = v.to(device)
    elif k == 'graph':
        batch[k] = batch[k].to(device)

# 定义函数加载模型并获取注意力权重
def get_attention_weights(checkpoint_path,Model):
    model = Model.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        if Model == IAGTransformerModel:
            inner_states, graph_rep, attn_weights = model.brep_encoder(
                batch,
                last_state_only=True,
                inst_adj=batch.get('inst_adj'),
                return_attn=True
            )
        elif Model == BrepSeg:
            inner_states, graph_rep, attn_weights = model.brep_encoder(
                batch,
                last_state_only=True,
                return_attn=True
            )
    # attn_weights 形状 [batch, tgt_len, src_len]（平均后的注意力）
    # 取第一个样本，去掉虚拟节点
    attn = attn_weights[0, 1:, 1:].cpu().numpy()
    return attn

# 获取基线和改进版的注意力权重
print("加载基线模型...")
baseline_attn = get_attention_weights(BASELINE_CHECKPOINT, BrepSeg)
print("加载改进版模型...")
improved_attn = get_attention_weights(IMPROVED_CHECKPOINT, IAGTransformerModel)

# 归一化
# def normalize(attn):
#     return (attn - attn.min()) / (attn.max() - attn.min())
#
# bl_norm = normalize(baseline_attn)
# imp_norm = normalize(improved_attn)

vmin = min(baseline_attn.min(), improved_attn.min())
vmax = max(baseline_attn.max(), improved_attn.max())
bl_norm = (baseline_attn - vmin) / (vmax - vmin)
imp_norm = (improved_attn - vmin) / (vmax - vmin)

# 创建图形，并定义3列：左右子图 + 中间/右侧colorbar列
fig = plt.figure(figsize=(12, 5))
gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.1)
# width_ratios: 左图1，右图1，colorbar列占0.05（窄）
# wspace: 列间距

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
cax = fig.add_subplot(gs[2])  # colorbar axes

im1 = ax1.imshow(bl_norm, cmap='viridis', vmin=0, vmax=1)
ax1.set_title('Baseline')
ax1.set_xlabel('Face index')
ax1.set_ylabel('Face index')

im2 = ax2.imshow(imp_norm, cmap='viridis', vmin=0, vmax=1)
ax2.set_title('IAG-Transformer')
ax2.set_xlabel('Face index')
ax2.set_ylabel('Face index')

cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
cbar.set_label('Attention weight')

plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
plt.show()

print(f"热图已保存至 {OUTPUT_IMAGE}")