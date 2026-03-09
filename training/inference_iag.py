import torch
import numpy as np
import pathlib


def compute_metrics(preds_list, labels_list):
    """
    独立指标计算函数
    :param preds_list: 包含所有 batch 预测结果的列表 (List of np.array)
    :param labels_list: 包含所有 batch 真值结果的列表 (List of np.array)
    :param num_classes: 类别数量 (通常为 25)
    """
    num_classes = 25
    # 将列表拼接为完整的 Numpy 数组
    preds_np = np.concatenate(preds_list)
    labels_np = np.concatenate(labels_list)

    results = {}

    # 1. 总体准确率 (Overall Accuracy)
    per_face_comp = (preds_np == labels_np).astype(np.int64)
    oa = np.mean(per_face_comp)
    results['OA'] = oa
    print(f"Overall Accuracy: {oa:.5f}")

    # 2. 每类平均准确率 (Mean Class Accuracy)
    per_class_acc = []
    for i in range(0, num_classes):
        class_pos = np.where(labels_np == i)
        if len(class_pos[0]) > 0:
            class_i_preds = preds_np[class_pos]
            class_i_label = labels_np[class_pos]
            acc_i = np.mean((class_i_preds == class_i_label).astype(np.int64))
            per_class_acc.append(acc_i)
            # print(f"Class_{i}_Acc: {acc_i:.5f}") # 可选打印

    mAcc = np.mean(per_class_acc)
    results['mAcc'] = mAcc
    print(f"Mean Class Accuracy: {mAcc:.5f}")

    # 3. 平均交并比 (mIoU)
    per_class_iou = []
    for i in range(0, num_classes):
        label_pos = np.where(labels_np == i)
        pred_pos = np.where(preds_np == i)

        if len(label_pos[0]) > 0:
            # 交集：预测和标签都是 i
            intersection = np.sum((preds_np[label_pos] == i).astype(np.int64))
            # 并集：标签是 i 或 预测是 i
            union = np.sum((labels_np == i).astype(np.int64)) + \
                    np.sum((preds_np == i).astype(np.int64)) - intersection

            iou = intersection / (union + 1e-12)
            per_class_iou.append(iou)

    mIoU = np.mean(per_class_iou)
    results['mIoU'] = mIoU
    print(f"mIoU: {mIoU:.5f}")

    return results


@torch.no_grad()
def generate_pseudo_inst_adj(batch, preds, device):
    """
    根据预测结果和 AAG 拓扑生成伪实例矩阵 [cite: 48, 75]
    """
    n_graph, max_n_node = batch["padding_mask"].size()[:2]
    node_pos = torch.where(batch["padding_mask"] == False)

    pred_matrix = -1 * torch.ones((n_graph, max_n_node), device=device, dtype=torch.long)
    pred_matrix[node_pos] = preds

    new_inst_adjs = []
    for i in range(n_graph):
        p_labels = pred_matrix[i]
        adj = batch["inst_adj"][i].to(device)  # 原始属性邻接图拓扑 [cite: 54]

        # 逻辑：类别相同 且 属于加工特征 (0-23) 且 拓扑邻接 [cite: 75, 82]
        class_match = (p_labels.unsqueeze(0) == p_labels.unsqueeze(1))
        is_feature = (p_labels < 24).unsqueeze(0) & (p_labels < 24).unsqueeze(1)

        M = class_match & is_feature & adj.to(torch.bool)
        M.fill_diagonal_(0)

        new_inst_adjs.append(M.long())
    return torch.stack(new_inst_adjs)


def run_forward(model, batch, inst_adj=None):
    """
    通用前向传播函数
    :param model: iag_model 或 baseline_model
    :param batch: 数据批次
    :param inst_adj: 实例矩阵，如果是基线模型则忽略此参数
    """
    # 判断模型是否为 IAG-Transformer (即是否需要 inst_adj)
    # 逻辑：如果传入了 inst_adj 且模型支持该参数
    if inst_adj is not None:
        node_emb, graph_emb = model.brep_encoder(batch, last_state_only=True, inst_adj=inst_adj)
    else:
        # 基线模型 (BrepSeg) 的调用，不传 inst_adj
        node_emb, graph_emb = model.brep_encoder(batch, last_state_only=True)

    # 后续分类逻辑（两者完全一致）[cite: 77]
    node_emb = node_emb[0].permute(1, 0, 2)[:, 1:, :]
    padding_mask = batch["padding_mask"]
    node_pos = torch.where(padding_mask == False)
    node_z = node_emb[node_pos]

    num_nodes_per_graph = torch.sum((~padding_mask).long(), dim=-1)
    graph_z = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)

    z = model.attention([node_z, graph_z])
    node_seg = model.classifier(z)

    return node_seg


# ================= 实验 1: 自校正 (Self-Correction) =================
def evaluate_self_correction(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Step 1: 零偏置预测
        zero_adj = torch.zeros_like(batch['inst_adj'])
        out_init = run_forward(model, batch, zero_adj)
        preds_init = torch.argmax(out_init, dim=-1)

        # Step 2: 生成伪偏置并精修
        m_self = generate_pseudo_inst_adj(batch, preds_init, device)
        out_final = run_forward(model, batch, m_self)

        preds = torch.argmax(out_final, dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch["label_feature"].cpu().numpy())

    return compute_metrics(all_preds, all_labels)


# ================= 实验 2: 级联推理 (Two-Stage) =================
def evaluate_two_stage(baseline_model, iag_model, dataloader, device):
    baseline_model.eval()
    iag_model.eval()
    all_preds, all_labels = [], []

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Step 1: 使用 Baseline 获得初始预测 [cite: 84]
        with torch.no_grad():
            # 假设 Baseline 输出直接为 [total_nodes, num_classes]
            out_base = run_forward(baseline_model, batch)
            preds_base = torch.argmax(out_base, dim=-1)

        # Step 2: 转化为 IAG 的先验矩阵 [cite: 68, 75]
        m_base = generate_pseudo_inst_adj(batch, preds_base, device)

        # Step 3: IAG 模型精修 [cite: 48, 63]
        out_final = run_forward(iag_model, batch, m_base)

        preds = torch.argmax(out_final, dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch["label_feature"].cpu().numpy())

    return compute_metrics(all_preds, all_labels)