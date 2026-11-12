"""【新增 IBN-KE-03】从已训练 IBN best.pt 构建完整组合模型，不覆盖原权重。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from ibn_kan_edge import BASE_IBN, INITIAL, YOLO, torch, attach_combo, branches_only_mode, source_hashes
from runtime import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE_IBN)
    parser.add_argument('--output', type=Path, default=INITIAL)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'拒绝覆盖已存在的权重：{args.output}')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    protected = source_hashes()
    loaded = YOLO(str(args.base), task='semantic')
    model = loaded.model.cpu().float().eval()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    sample = torch.rand(1, 3, 256, 256)
    with torch.inference_mode():
        expected = model(sample)
    attach_combo(model)
    model.eval()
    for name, value in before.items():
        target = name.replace('model.17.classifier.', 'model.17.classifier.kan_classifier.original_classifier.', 1)
        if not torch.equal(value, model.state_dict()[target]):
            raise RuntimeError(f'原 IBN 参数迁移不一致：{name}')
    with torch.inference_mode():
        actual = model(sample)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    branches_only_mode(model)
    trainable = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    model.eval()
    model.model[-1].classifier.last_edge_logits = None
    model.criterion = None
    checkpoint = dict(loaded.ckpt)
    checkpoint.update(model=deepcopy(model), ema=None, optimizer=None, scaler=None, updates=None,
                      epoch=-1, best_fitness=None, stopper=None, train_metrics={}, train_results=None,
                      date=datetime.now().astimezone().isoformat(),
                      ibn_kan_edge_metadata={**model.ibn_kan_edge_config,
                                             'base': str(args.base), 'base_sha256': sha256(args.base),
                                             'new_branches_trained': False})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    assert source_hashes() == protected
    report = {'base': str(args.base), 'base_sha256': sha256(args.base), 'output': str(args.output),
              'output_sha256': sha256(args.output), 'preserved_base_tensors': len(before),
              'initial_output_exactly_equal_to_ibn': True, 'parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameter_count': sum(trainable.values()), 'trainable_parameters': trainable,
              'source_files_unchanged': True, 'protected_files': protected,
              'base_ibn_previously_trained': True, 'new_branches_trained': False}
    (args.output.parent / 'ibn_kan_edge_build_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('protected_files', 'trainable_parameters')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
