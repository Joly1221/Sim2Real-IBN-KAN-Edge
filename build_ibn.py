"""【新增文件 IBN-02】生成独立 IBN 初始权重；不覆盖原 best.pt，不启动训练。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from runtime import ROOT, BASE, INITIAL, YOLO, torch, protected_hashes, sha256
from ibn_backbone import attach_ibn


def build(base: Path = BASE, output: Path = INITIAL) -> dict:
    if output.exists():
        raise FileExistsError(f'文件已存在，拒绝覆盖：{output}')
    before_hashes = protected_hashes()
    wrapper = YOLO(str(base), task='semantic')
    model = wrapper.model.cpu().float().eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    old_parameters = sum(p.numel() for p in model.parameters())
    attach_ibn(model)
    # 【新增】后续采用骨干、颈部、分类头联合仿真训练。
    model.requires_grad_(True)
    after = model.state_dict()
    unchanged = [k for k in before if not k.startswith('model.2.cv1.bn.')]
    assert all(k in after and torch.equal(before[k], after[k]) for k in unchanged)
    assert sum(p.numel() for p in model.parameters()) == old_parameters
    # 【修改】不沿用原模型的旧训练分数/优化器；此文件只表示 IBN 尚未训练的初始化。
    checkpoint = dict(wrapper.ckpt)
    checkpoint.update(model=deepcopy(model), ema=None, optimizer=None, scaler=None,
                      updates=None, epoch=-1, best_fitness=None, train_metrics={}, train_results=None,
                      date=datetime.now(timezone.utc).isoformat(),
                      ibn_metadata={**model.ibn_config, 'initialization_only': True,
                                    'source_weights': str(base), 'source_sha256': sha256(base)})
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    assert protected_hashes() == before_hashes
    report = {'output': str(output), 'source': str(base), 'source_sha256': sha256(base),
              'output_sha256': sha256(output), 'parameters_before': old_parameters,
              'parameters_after': sum(p.numel() for p in model.parameters()),
              'unchanged_non_target_tensors': len(unchanged), 'protected_files': before_hashes,
              'protected_files_unchanged': True, 'initialization_only': True,
              'output_equivalence_to_original_expected': False,
              'replacement': 'model.2.cv1.bn: BN(32) -> IN(16)+BN(16)'}
    (output.parent / 'build_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--output', type=Path, default=INITIAL)
    options = parser.parse_args()
    build(options.base, options.output)
