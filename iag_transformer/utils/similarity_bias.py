# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceSimilarityBias(nn.Module):
    """
    无监督面-面相似度偏置（基于原始几何特征，无标签泄露）
    核心：计算所有面之间的余弦相似度，作为可学习权重的偏置融入Attention
    """

    def __init__(self, dim_node=256, alpha=0.1):
        super(InstanceSimilarityBias, self).__init__()
        # 可学习权重：控制相似度偏置的贡献度，初始0.1（可自动调优）
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.dim_node = dim_node  # 面特征维度（原模型是256）

    def forward(self, node_feat):
        """
        前向传播：计算面-面余弦相似度偏置
        :param node_feat: 所有面的原始几何特征，shape=[N, 256]（N=总面数）
        :return: 相似度偏置矩阵，shape=[N, N]
        """
        # 步骤1：L2归一化特征（避免面积/法向量等尺度影响相似度）
        node_feat_norm = F.normalize(node_feat, p=2, dim=-1)

        # 步骤2：计算面-面余弦相似度矩阵（核心1行）
        # sim_matrix[i,j] = 面i和面j的余弦相似度（范围[-1,1]）
        sim_matrix = torch.mm(node_feat_norm, node_feat_norm.T)

        # 步骤3：用可学习权重缩放偏置（控制强度）
        sim_bias = self.alpha * sim_matrix

        return sim_bias