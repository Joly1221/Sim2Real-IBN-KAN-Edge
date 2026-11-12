"""【新增文件 IBN-01】仅替换 model[2].cv1 的归一化，保留原卷积和 SiLU。

IBN 的分通道实现参考 XingangPan/IBN-Net（ECCV 2018）。
官方实现：https://github.com/XingangPan/IBN-Net/blob/master/ibnnet/modules.py
许可见 THIRD_PARTY_NOTICES.md；这里增加原 BN 参数迁移和 Ultralytics 融合兼容。
"""
from __future__ import annotations

from runtime import torch, NAMES
from torch import nn
from ultralytics.nn.modules import Conv, C3k2


class IBN(nn.Module):
    """【新增】前半通道 IN，后半通道 BN；IN 推理时不积累跨图运行统计。"""

    def __init__(self, channels: int, eps: float = .001, momentum: float = .03):
        super().__init__()
        if channels < 2 or channels % 2:
            raise ValueError('本实验限定偶数通道，IN/BN 各占 50%')
        self.half = channels // 2
        # 【适配】沿用原 BN 的 eps/momentum；仿真预训练 BN 的 eps 为 0.001。
        self.IN = nn.InstanceNorm2d(self.half, eps=eps, affine=True, track_running_stats=False)
        self.BN = nn.BatchNorm2d(self.half, eps=eps, momentum=momentum)

    @classmethod
    def from_bn(cls, source: nn.BatchNorm2d) -> 'IBN':
        """【新增】复制仿射参数；BN 后半复制运行统计；IN 使用逐图统计。"""
        if not source.affine or not source.track_running_stats:
            raise ValueError('要求原 BN 含仿射参数和运行统计')
        result = cls(source.num_features, source.eps, source.momentum)
        result.to(device=source.weight.device, dtype=source.weight.dtype)
        n = result.half
        with torch.no_grad():
            result.IN.weight.copy_(source.weight[:n])
            result.IN.bias.copy_(source.bias[:n])
            result.BN.weight.copy_(source.weight[n:])
            result.BN.bias.copy_(source.bias[n:])
            result.BN.running_mean.copy_(source.running_mean[n:])
            result.BN.running_var.copy_(source.running_var[n:])
            result.BN.num_batches_tracked.copy_(source.num_batches_tracked)
        result.train(source.training)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 【新增】只按 C 维分组，高宽不变；拼接顺序和原通道顺序一致。
        first, second = x.split(self.half, dim=1)
        return torch.cat((self.IN(first.contiguous()), self.BN(second.contiguous())), dim=1)


class ConvWithIBN(nn.Module):
    """【新增】Conv→IBN→SiLU；独立类型避免被错误执行 Conv+BN 融合。

    原 Conv 源码完整保留。本包装层的 conv/act 直接继承原实例；仅 bn 类型改变。
    不能继承 Conv：原 fuse() 会将 Conv.bn 一律当作 BatchNorm2d 处理。
    """

    def __init__(self, original: Conv):
        super().__init__()
        self.conv = original.conv
        self.act = original.act
        # 【原结构，注释保留供对照】self.bn = original.bn
        self.bn = IBN.from_bn(original.bn)  # 【修改 IBN-01】BN(32) → IN(16) + BN(16)
        self.train(original.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 【原 Conv.forward，注释保留】return self.act(self.bn(self.conv(x)))
        # 【修改说明】调用顺序不变；现在 self.bn 执行 IBN 的分组归一化。
        return self.act(self.bn(self.conv(x)))


def assert_ibn(model: nn.Module) -> None:
    """【新增】防止载入旧适配器模型、重建时丢失 IBN 或改错层。"""
    if [model.names[i] for i in range(5)] != NAMES:
        raise RuntimeError('模型的五类顺序不一致')
    if not isinstance(model.model[2], C3k2) or not isinstance(model.model[2].cv1, ConvWithIBN):
        raise RuntimeError('model[2].cv1 必须是本实验 ConvWithIBN')
    found = [name for name, m in model.named_modules() if isinstance(m, IBN)]
    if found != ['model.2.cv1.bn'] or model.model[2].cv1.bn.half != 16:
        raise RuntimeError(f'IBN 数量、位置或通道不符：{found}')


def attach_ibn(model: nn.Module) -> None:
    """【新增】仅对原始仿真模型做一次替换，不叠加 KAN/边缘分支。"""
    block = model.model[2]
    if not isinstance(block, C3k2) or type(block.cv1) is not Conv:
        raise TypeError('需要尚未融合、尚未改造的原始 model[2].cv1')
    original = block.cv1
    if not isinstance(original.bn, nn.BatchNorm2d) or original.conv.in_channels != 32 or original.conv.out_channels != 32:
        raise ValueError('只支持当前 YOLO26n-sem 的 32→32 浅层投影卷积')
    if not isinstance(model.model[-1].classifier, nn.Sequential):
        raise ValueError('请从原始仿真 best.pt 出发，不使用 KAN/edge 权重')
    # 【原代码保留】block.cv1 = original  # 原 1×1 Conv + BN(32) + SiLU
    block.cv1 = ConvWithIBN(original)  # 【修改 IBN-02】仅替换这个卷积容器中的归一化
    model.ibn_config = {'location': 'model.2.cv1.bn', 'channels': 32, 'in_channels': 16,
                        'bn_channels': 16, 'training_scope': 'backbone_neck_head',
                        'source_only': True}
    model.criterion = None
    assert_ibn(model)
