"""【新增】基于官方 pykan 的 P3 残差适配器；原 MLP 源码保留不变。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
# 【新增】仅从实验目录读取官方源码和局部依赖，不修改用户原 Python 环境。
sys.path.insert(0, str(ROOT / 'dependencies'))
sys.path.insert(0, str(ROOT / 'official_pykan'))
from kan.KANLayer import KANLayer  # noqa: E402  # 作者官方源码，未修改。


class P3KANAdapter(nn.Module):
    """【修改】逐空间位置处理通道：64→16→KAN(16→16)→64。"""

    def __init__(self, channels: int = 64, hidden: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act1 = nn.SiLU()
        # 【原 MLP 源代码，保留注释】
        # self.fc2 = nn.Conv2d(hidden, hidden, kernel_size=1)
        # self.act2 = nn.SiLU()
        # 【新增 KAN】官方 KANLayer；网格为作者默认 [-1,1]、5 个区间、三次样条。
        # 这是初始实验设置，网格是否覆盖仿真训练特征仍需专门检查。
        self.kan = KANLayer(in_dim=hidden, out_dim=hidden, num=5, k=3,
                            grid_range=[-1, 1], save_plot_data=False)
        self.fc3 = nn.Conv2d(hidden, channels, kernel_size=1)
        # 【新增】零初始化输出，保证初始残差为零。
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.fc1(x))
        # 【原 MLP 源代码，保留注释】
        # x = self.act2(self.fc2(x))
        # 【修改】官方 KAN 输入 [样本数, 通道]，对空间位置共享同一组参数。
        batch, channels, height, width = x.shape
        positions = x.permute(0, 2, 3, 1).reshape(-1, channels)
        transformed = self.kan(positions)[0]
        x = transformed.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        return self.fc3(x)


class P3ClassifierWithKAN(nn.Module):
    """【新增】保留原分类器，先将 KAN 分支输出与 P3 特征做残差相加。"""

    def __init__(self, classifier: nn.Sequential) -> None:
        super().__init__()
        self.original_classifier = classifier
        channels = classifier[0].conv.in_channels
        if channels != 64:
            raise ValueError(f'期望 P3 为 64 通道，实际为 {channels}')
        # 【原 MLP 源代码，保留注释】
        # self.adapter = MultiLayerMLPAdapter(channels=64, hidden=16)
        self.adapter = P3KANAdapter(channels=64, hidden=16)  # 【修改】

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_classifier(x + self.adapter(x))


def attach_kan_adapter(model: nn.Module) -> None:
    """【新增】在 P3 主分类器前接入，不修改 P4 辅助分支。"""
    head = model.model[-1]
    if not isinstance(head.classifier, nn.Sequential):
        raise ValueError('主分类器结构不符合预期，或已经接入适配器')
    # 【原 MLP 源代码，保留注释】
    # head.classifier = P3ClassifierWithMLP(head.classifier)
    head.classifier = P3ClassifierWithKAN(head.classifier)


def kan_only_mode(model: nn.Module) -> None:
    """【修改】冻结原模型和 BN；保留官方 KAN 的固定网格与固定掩码。"""
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    head = model.model[-1]
    head.training = True  # 训练时返回主输出与 P4 辅助输出；原 BN 不切回训练模式。
    adapter = head.classifier.adapter
    adapter.train()
    # 【原 MLP 源代码，保留注释】
    # for param in adapter.parameters():
    #     param.requires_grad_(True)
    # 【修改】官方 KAN 的 grid、mask 原本固定，不能一律设为可训练。
    for name, param in adapter.named_parameters():
        param.requires_grad_(name not in {'kan.grid', 'kan.mask'})
