"""【新增 IBN-KE-01】在已训练 IBN 的 P3 后接一层 KAN，再接原边缘分支。

复用已有 kan_adapter.py、edge_branch.py；不修改或删除它们的源码。
这里仅组合现有模块，并提供 KAN 与边缘分支一起训练的冻结模式。
"""
from __future__ import annotations

import sys
from pathlib import Path
from runtime import ROOT, YOLO, torch, sha256, protected_hashes
from ibn_backbone import assert_ibn
from torch import nn

# 【整理前代码保留】 KAN_ROOT = Path(r'D:\Projects\[Seg]\ultralytics-main\sim2real_kan_adapter')
# 【整理前代码保留】 EDGE_ROOT = Path(r'D:\Projects\Sim2real\imp\edge')
# 【整理修改 CODE-PACKAGE-02】使用本副本中的模块与资源。
KAN_ROOT = ROOT
EDGE_ROOT = ROOT
sys.path.insert(0, str(KAN_ROOT))
sys.path.insert(0, str(EDGE_ROOT))
from kan_adapter import P3ClassifierWithKAN, attach_kan_adapter  # noqa: E402
from edge_branch import P3ClassifierWithEdge, EdgeSemanticModel, attach_edge  # noqa: E402

# 【整理前代码保留】 BASE_IBN = Path(r'D:\Projects\Sim2real\imp\IBN\runs\ibn_sim_1024\weights\best.pt')
# 【整理修改 CODE-PACKAGE-02】使用本副本中的模块与资源。
BASE_IBN = ROOT / 'runs/ibn_sim_1024/weights/best.pt'
INITIAL = ROOT / 'weights/ibn_kan_edge_init.pt'
KAN_PREFIX = 'model.17.classifier.kan_classifier.adapter.'
EDGE_PREFIX = 'model.17.classifier.edge.'


def is_branch_parameter(name: str) -> bool:
    """【新增】固定 KAN 网格和掩码；只开放 KAN 可学习参数及边缘分支。"""
    return ((name.startswith(KAN_PREFIX) and not name.endswith(('.kan.grid', '.kan.mask')))
            or name.startswith(EDGE_PREFIX))


def assert_combo(model: nn.Module) -> None:
    assert_ibn(model)
    head = model.model[-1].classifier
    if not isinstance(model, EdgeSemanticModel) or not isinstance(head, P3ClassifierWithEdge):
        raise TypeError('需要同时包含 IBN、单层 KAN、边缘分支的完整模型')
    if not isinstance(head.kan_classifier, P3ClassifierWithKAN):
        raise TypeError('边缘分支前缺少单层 KAN')
    layers = [n for n, m in model.named_modules() if type(m).__name__ == 'KANLayer']
    if layers != ['model.17.classifier.kan_classifier.adapter.kan']:
        raise RuntimeError(f'应恰好包含一层 KAN：{layers}')
    edges = [n for n, m in model.named_modules() if type(m).__name__ == 'CraterEdgeBranch']
    if edges != ['model.17.classifier.edge']:
        raise RuntimeError(f'边缘分支位置不正确：{edges}')


def attach_combo(model: nn.Module) -> None:
    """【新增】与旧 KAN+edge 版本位置一致，原 IBN 主模型参数原样保留。"""
    assert_ibn(model)
    if not isinstance(model.model[-1].classifier, nn.Sequential):
        raise ValueError('需要尚未添加适配器的 IBN best.pt，避免重复叠加')
    # 【原 IBN 路径，注释保留】logits = original_classifier(P3)
    attach_kan_adapter(model)  # 【新增】z = P3 + KAN_adapter(P3)
    attach_edge(model)         # 【新增】logits = original_classifier(z + edge_residual(z))
    # 【说明】两项新增模块均采用原实现的零残差初始化，不移植旧骨干上的已训练分支权重。
    model.ibn_kan_edge_config = {'placement': 'P3 -> KAN residual -> edge residual -> original classifier',
                                'kan_layers': 1, 'edge_loss_weight': 0.1,
                                'trainable_scope': 'kan_and_edge', 'ibn_base_frozen': True,
                                'branch_initialization': 'fresh_zero_residual', 'source_only': True}
    assert_combo(model)


def branches_only_mode(model: nn.Module) -> None:
    """【新增】同时训练 KAN+edge；冻结 IBN、骨干、颈部、分类头及 BN 统计。"""
    assert_combo(model)
    model.eval()
    model.requires_grad_(False)
    head = model.model[-1]
    head.training = True  # 只启用原语义头的训练输出，不递归打开原 BN。
    head.classifier.training = True  # 边缘 logits 保留计算图以计算边界损失。
    head.classifier.kan_classifier.adapter.train()
    head.classifier.edge.train()
    # 【旧 edge_only_mode，注释保留】只开放 classifier.edge，KAN 冻结。
    # 【修改 IBN-KE-02】此次 KAN 也是新初始化，因此 KAN 和 edge 都需要训练。
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(is_branch_parameter(name))


def frozen_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if not is_branch_parameter(name)}


def check_frozen(model: nn.Module, reference: dict[str, torch.Tensor]) -> None:
    for name, value in reference.items():
        if not torch.equal(model.state_dict()[name].detach().cpu(), value):
            raise RuntimeError(f'冻结参数或缓冲区变化：{name}')
    if any(p.requires_grad != is_branch_parameter(n) for n, p in model.named_parameters()):
        raise RuntimeError('可训练参数范围不正确')
    if any(m.training for m in model.modules() if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d))):
        raise RuntimeError('原 IN/BN 应保持 eval 模式')


def source_hashes() -> dict[str, str]:
    hashes = protected_hashes()
# 【整理前代码保留】     paths = [BASE_IBN, KAN_ROOT / 'kan_adapter.py', EDGE_ROOT / 'edge_branch.py',
# 【整理前代码保留】              KAN_ROOT / 'runs/kan_sim_1024/weights/best.pt',
# 【整理前代码保留】              EDGE_ROOT / 'runs/edge_sim_1024-2/weights/best.pt',
# 【整理前代码保留】              Path(r'D:\Projects\Sim2real\imp\IBN') / 'ibn_backbone.py',
# 【整理前代码保留】              Path(r'D:\Projects\Sim2real\imp\IBN') / 'train_ibn.py']
# 【整理修改 CODE-PACKAGE-02】使用本副本中的模块与资源。
    paths = [BASE_IBN, ROOT / 'kan_adapter.py', ROOT / 'edge_branch.py',
             ROOT / 'ibn_backbone.py', ROOT / 'train_ibn.py', ROOT / 'ibn_kan_edge.py']
    hashes.update({str(p): sha256(p) for p in paths})
    return hashes
