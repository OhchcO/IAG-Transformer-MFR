from datetime import datetime
import pathlib

import torch
from torch.utils.data import DataLoader
# 导入你的模型类和工具函数
from iag_transformer.iag_transformer_model import IAGTransformerModel
from models.visual_brepseg_model import BrepSeg
from inference_iag import evaluate_self_correction, evaluate_two_stage
from data.dataset import CADSynth

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 25 # 包含背景类 [cite: 82]
    dataset_path = r'D:\Code\BrepMFR\dataset'
    sample_ratio = 0.01
    batch_size = 64
    num_workers = 0

    # 配置日志保存
    log_dir = pathlib.Path("experiment_logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # 定义一个简单的记录函数
    def log_print(message):
        print(message)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")


    # 1. 加载训练好的权重
    # 请替换为你的实际路径
    iag_ckpt = r"D:\Code\BrepMFR\结果对比（新）\IAG_1pct_lambda001_s\0303\train_epoch100_105138\best.ckpt"
    base_ckpt = r"D:\Code\BrepMFR\结果对比（新）\BrepMFR_100pct\train_epoch90_191331\best.ckpt"

    print("Loading models...")
    # 加载改进后的 IAG 模型 [cite: 84]
    iag_model = IAGTransformerModel.load_from_checkpoint(iag_ckpt).to(device)
    # 加载基线模型 (BrepSeg) [cite: 84]
    baseline_model = BrepSeg.load_from_checkpoint(base_ckpt).to(device)

    log_print(f"Time: {datetime.now()}")
    log_print(f"IAG Checkpoint: {iag_ckpt}")
    log_print(f"Baseline Checkpoint: {base_ckpt}")

    # 2. 准备测试数据集
    # 确保你使用的 dataset 包含 'adj' (AAG邻接矩阵) 字段 [cite: 54]
    Dataset = CADSynth
    test_data = Dataset(root_dir=dataset_path, split="test", random_rotate=False, num_class=num_classes,
                        sample_ratio=sample_ratio)
    test_loader = test_data.get_dataloader(
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )


    # 3. 运行实验 1: 自校正 (Self-Correction)
    # 验证模型在没有外部帮助下，通过迭代产生的伪偏置是否能提升性能
    log_print("\n--- Running Experiment 1: Self-Correction ---")
    metrics_1 = evaluate_self_correction(iag_model, test_loader, device)
    log_print(f"Metrics 1: {metrics_1}")

    # 4. 运行实验 2: 级联推理 (Two-Stage Cascade)
    # 验证利用 Baseline 的稳健分类结果作为 IAG 的先验，是否能达到最佳效果
    log_print("\n--- Running Experiment 2: Two-Stage Cascade ---")
    metrics_2 = evaluate_two_stage(baseline_model, iag_model, test_loader, device)
    log_print(f"Metrics 2: {metrics_2}")

if __name__ == "__main__":
    main()