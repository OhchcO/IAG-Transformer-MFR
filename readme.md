# IAG-Transformer

IAG-Transformer: An Instance-Aware Graph Network for Few-Shot Machining Feature Recognition

（IAG-Transformer：面向小样本加工特征识别的实例感知图网络）

## 简介

IAG-Transformer 是一种专为零件加工特征识别设计的图 Transformer 架构，其核心创新在于将实例感知注意力偏置引入自注意力机制，使模型能够显式建模同一加工特征实例内面与面之间的语义关联。与现有方法仅依赖拓扑和几何信息不同，IAG-Transformer 通过自动生成的伪实例标签，引导注意力聚焦于同实例的面，从而在小样本场景下实现显著性能提升。

## 核心贡献

- **实例感知注意力偏置**：在 Transformer 的多头注意力中引入可学习的偏置项，使同一实例内的面获得更高的注意力权重，增强特征紧凑性。
- **可学习缩放因子**：引入可训练标量 \( s \)，允许模型自适应调整实例先验的强度，提升对不同数据分布的鲁棒性。
- **无监督伪实例标签**：利用 B-rep 图的拓扑连通性和面类别信息自动生成伪实例标签，无需人工标注，即可获得实例级监督信号。
- **小样本下的卓越性能**：在 CADSynth 数据集上，仅用 1% 训练数据即可将准确率从 44.08% 提升至 49.01%（绝对增益 +4.93%），5% 数据下提升 2.02%，充分验证了方法的有效性。

IAG-Transformer 不仅继承了 BrepMFR 的图建模能力，更通过实例感知机制为加工特征识别提供了全新的思路，尤其适用于工业场景中标注数据稀缺的情况。

## 依赖
```bash
Python 3.8+
PyTorch 1.10+
PyTorch Lightning 1.6+
DGL 0.8+
fairseq
numpy, scipy, tqdm
```


## 安装

```bash
git clone https://github.com/yourusername/IAG-Transformer.git
cd IAG-Transformer
pip install -r requirements.txt
```

## 数据准备

请参考 BrepMFR 准备 CADSynth 数据集，确保目录结构如下：
```txt
dataset/
├── train.txt
├── val.txt
├── test.txt
└── *.bin
```

## 训练与验证
此处以使用 1% 数据训练，启用可学习缩放因子 \( s \)，初始化标准差 λ=0.1，训练20轮为例
```bash
python train_iag.py train \
    --dataset_path /path/to/dataset \
    --max_epochs 20 \
    --batch_size 64 \
    --sample_ratio 0.01 \
    --used_inst_scale \
    --inst_init_std 0.1 \
    --experiment_name IAG_1pct_lambda0.1_s
```

## 测试
```bash
python train_iag.py test \
    --dataset_path /path/to/dataset \
    --checkpoint ./results/XXXXX/XXXX/XXX/best.ckpt \
    --batch_size 64 \
    --sample_ratio 0.01
```

## 日志
```bash
tensorboard --logdir /results/XXXXX/XXXX/XXX/
```

## 许可证

