"""【新增文件】坑边界辅助监督与轻量 P3 残差分支；原 YOLO/KAN 文件未修改、未删除。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# 【整理前代码保留】 KAN_ROOT = Path(r'D:\Projects\[Seg]\ultralytics-main\sim2real_kan_adapter')
# 【整理修改 CODE-PACKAGE-02】使用本副本中的模块与资源。
KAN_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(KAN_ROOT))
from kan_adapter import P3ClassifierWithKAN  # noqa: E402
from ultralytics.nn.tasks import SemanticSegmentationModel  # noqa: E402


class CraterEdgeBranch(nn.Module):
    """【新增】局部边缘特征同时送往边界监督与主分类器。"""

    def __init__(self, channels: int = 64, hidden: int = 16) -> None:
        super().__init__()
        self.feature = nn.Sequential(nn.Conv2d(channels, hidden, 3, padding=1), nn.SiLU())
        self.edge_head = nn.Conv2d(hidden, 1, 1)
        self.project = nn.Conv2d(hidden, channels, 1)
        # 【新增】初始残差为零，保证加载后未训练时等价于原 KAN 模型。
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature(x)
        return self.project(features), self.edge_head(features)


class P3ClassifierWithEdge(nn.Module):
    """【新增】原 P3+KAN 特征添加边缘残差，再调用原五类分类器。"""

    def __init__(self, kan_classifier: P3ClassifierWithKAN) -> None:
        super().__init__()
        self.kan_classifier = kan_classifier
        self.edge = CraterEdgeBranch()
        self.last_edge_logits = None  # 当前批次辅助输出；计算损失后立即清空。

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 【原 KAN 代码，仅供对照；原文件仍完整保留】
        # return self.original_classifier(x + self.adapter(x))
        # 【修改】先计算原 KAN 特征，再用新增边缘分支产生一项残差。
        z = x + self.kan_classifier.adapter(x)
        residual, edge_logits = self.edge(z)
        # 验证器先调用前向、再计算验证损失；推理时仅保留不带计算图的输出。
        self.last_edge_logits = edge_logits if self.training else edge_logits.detach()
        return self.kan_classifier.original_classifier(z + residual)  # 【修改】原分类器接收融合后的特征。


def crater_boundary(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """【新增】从增强后的类别标签构造 4 邻接坑边界，忽略 255 及其直接邻域。"""
    if mask.ndim != 3:
        raise ValueError('语义标签需为 [B,H,W]')
    valid = mask != 255
    known = valid.clone()
    crater = mask == 1
    edge = torch.zeros_like(crater)
    # 对每组上下/左右相邻像素：只有两侧均已标注时才生成边界。
    pairs = [(slice(None, -1), slice(None), slice(1, None), slice(None)),
             (slice(None), slice(None, -1), slice(None), slice(1, None))]
    for ay, ax, by, bx in pairs:
        va, vb = valid[:, ay, ax], valid[:, by, bx]
        different = (crater[:, ay, ax] != crater[:, by, bx]) & va & vb
        edge[:, ay, ax] |= different
        edge[:, by, bx] |= different
        known[:, ay, ax] &= vb
        known[:, by, bx] &= va
    return edge & known, known


def crater_edge_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """【新增】计算边界 BCE；动态正类权重封顶 20，无正样本时正常监督非边界。"""
    edge, valid = crater_boundary(mask)
    logits = F.interpolate(logits.float(), size=mask.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
    positive = edge.sum().to(dtype=logits.dtype)
    negative = (valid & ~edge).sum().to(dtype=logits.dtype)
    weight = (negative / positive.clamp_min(1)).clamp(1, 20)
    loss = F.binary_cross_entropy_with_logits(logits, edge.to(logits.dtype), reduction='none', pos_weight=weight)
    return (loss * valid).sum() / valid.sum().clamp_min(1)


class EdgeSemanticModel(SemanticSegmentationModel):
    """【新增】保留原语义损失并添加 0.1 倍坑边界损失。"""

    def loss(self, batch, preds=None):
        # 【原损失代码，保留并继续执行】
        total, items = super().loss(batch, preds)
        # 【新增】获取本批次边界预测，并将边界损失加到原语义损失上。
        classifier = self.model[-1].classifier
        edge_logits = classifier.last_edge_logits
        classifier.last_edge_logits = None  # 避免长时间持有计算图。
        if edge_logits is None:
            raise RuntimeError('缺少当前批次的边界输出；请使用 edge_only_mode')
        edge = crater_edge_loss(edge_logits, batch['semantic_mask'].to(edge_logits.device))
        items = dict(items)
        items['crater_edge_loss'] = edge.detach()
        return total + 0.1 * edge * batch['img'].shape[0], items


def attach_edge(model: nn.Module) -> None:
    """只包装 KAN 主分类器；类型扩展提供训练损失，不修改源模型实现。"""
    if not isinstance(model, SemanticSegmentationModel):
        raise TypeError('需要原语义分割模型')
    head = model.model[-1]
    if not isinstance(head.classifier, P3ClassifierWithKAN):
        raise TypeError('需要尚未接入边缘分支的 KAN 最佳模型')
    # 【原 KAN 主路径】head.classifier(x) = classifier(x + adapter(x))
    head.classifier = P3ClassifierWithEdge(head.classifier)
    model.__class__ = EdgeSemanticModel  # 【新增】仅当前实例使用扩展损失。
    model.criterion = None


def edge_only_mode(model: nn.Module) -> None:
    """【新增】仅训练新增边缘分支；原模型、KAN、BN 统计和 KAN 网格保持固定。"""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    head = model.model[-1]
    head.training = True
    head.classifier.training = True  # 保存当前批次边界输出，不递归切换冻结子模块。
    head.classifier.edge.train()
    for p in head.classifier.edge.parameters():
        p.requires_grad_(True)
