"""【新增 IBN-KE-04】固定 IBN 主模型，只用仿真数据联合训练新增 KAN 和边缘分支。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from ibn_kan_edge import (INITIAL, YOLO, torch, assert_combo, branches_only_mode,
                          frozen_state, check_frozen, is_branch_parameter)
from runtime import ROOT
from train_ibn import simulation_data
from ultralytics.models.yolo.semantic.train import SemanticSegmentationTrainer


class IBNKANEdgeTrainer(SemanticSegmentationTrainer):
    """【新增】防止基础训练器重新开放冻结层；优化器和 EMA 都限制到新增分支。"""

    def _build_train_pipeline(self):
        branches_only_mode(self.model)
        result = super()._build_train_pipeline()
        expected = {id(p) for p in self.model.parameters() if p.requires_grad}
        actual = [p for group in self.optimizer.param_groups for p in group['params']]
        if not expected or {id(p) for p in actual} != expected or len(actual) != len(expected):
            raise RuntimeError('优化器必须且仅包含 KAN 与 edge 的可训练参数')
        self.check_frozen()
        return result

    def build_optimizer(self, model, name='AdamW', lr=.0001, momentum=.9, decay=.0005, iterations=1):
        selected = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        return torch.optim.AdamW([
            {'params': [p for n, p in selected if not n.endswith('bias')], 'weight_decay': decay, 'param_group': 'weight'},
            {'params': [p for n, p in selected if n.endswith('bias')], 'weight_decay': 0., 'param_group': 'bias'},
        ], lr=lr, betas=(momentum, .999))

    def _model_train(self):
        # 【原入口，注释保留】self.model.train()  # 会重新更新原 BN 统计。
        branches_only_mode(self.model)
        self.check_frozen()

    def check_frozen(self):
        check_frozen(self.model, self.frozen_reference)

    def optimizer_step(self):
        super().optimizer_step()
        # 【新增】EMA 的浮点平均也可能产生舍入漂移；冻结部分直接复制原值。
        with torch.no_grad():
            live = self.model.state_dict()
            for name, value in self.ema.ema.state_dict().items():
                if name in self.frozen_reference:
                    value.copy_(live[name])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    parser.add_argument('--weights', type=Path, default=INITIAL)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', default=None)
    parser.add_argument('--name', default=None)
    args = parser.parse_args()
    if min(args.epochs, args.batch) < 1 or args.imgsz < 64 or args.lr <= 0 or args.patience < 0:
        parser.error('训练参数超出有效范围')
    if args.device and ',' in args.device:
        parser.error('本脚本仅验证了单 GPU 或 CPU')
    torch.set_num_threads(1)
    model = YOLO(str(args.weights), task='semantic').model.cpu().float()
    assert_combo(model)
    branches_only_mode(model)
    reference = frozen_state(model)
    initial = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    data = simulation_data(args.smoke)
    settings = dict(model=str(args.weights), task='semantic', data=str(data),
                    epochs=1 if args.smoke else args.epochs, imgsz=256 if args.smoke else args.imgsz,
                    batch=2 if args.smoke else args.batch, device=args.device or ('cpu' if args.smoke else '0'),
                    project=str(ROOT / 'runs'), name=args.name or ('ibn_kan_edge_smoke' if args.smoke else 'ibn_kan_edge_sim_1024'),
                    exist_ok=False, optimizer='AdamW', lr0=args.lr, momentum=.9, weight_decay=.0005,
                    warmup_bias_lr=0., warmup_epochs=0. if args.smoke else 3., nbs=2 if args.smoke else 64,
                    patience=args.patience, workers=0, amp=False, cache=False, plots=False,
                    freeze=None, pretrained=True, seed=42, deterministic=True, mosaic=0., close_mosaic=0)
    trainer = IBNKANEdgeTrainer(overrides=settings)
    # 【原常规入口，注释保留】YOLO(str(args.weights)).train(...)
    trainer.model = model  # 【新增】沿用完整结构，避免按原 YAML 重建丢失 IBN/KAN/edge。
    trainer.frozen_reference = reference
    trainer.add_callback('on_fit_epoch_end', lambda t: t.check_frozen())
    progress = {'batch': 0, 'last_write': 0.}

    def save_progress(t, stage: str, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - progress['last_write'] < 10:
            return
        state = {'stage': stage, 'epoch': t.epoch + 1, 'epochs': t.epochs,
                 'batch': progress['batch'], 'batches_per_epoch': len(t.train_loader),
                 'updated_local_time': time.strftime('%Y-%m-%d %H:%M:%S')}
        target = t.save_dir / 'progress.json'
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(target)
        progress['last_write'] = now

    def epoch_start(t) -> None:
        progress['batch'] = 0
        save_progress(t, 'training', True)

    def batch_end(t) -> None:
        progress['batch'] += 1
        save_progress(t, 'training')

    trainer.add_callback('on_train_epoch_start', epoch_start)
    trainer.add_callback('on_train_batch_end', batch_end)
    trainer.add_callback('on_train_epoch_end', lambda t: save_progress(t, 'validating', True))
    trainer.train()
    trainer.check_frozen()
    state = trainer.model.state_dict()
    changed = [n for n, p in initial.items() if not torch.equal(p, state[n].detach().cpu())]
    if not any('.adapter.' in n for n in changed) or not any('.classifier.edge.' in n for n in changed):
        raise RuntimeError('KAN 或边缘分支没有更新')
    reloaded = YOLO(str(trainer.best), task='semantic').model.cpu().float().eval()
    assert_combo(reloaded)
    # 保存格式是原框架的 FP16；比较时先作同样的存储精度转换。
    for name, tensor in reference.items():
        stored = tensor.half().float() if tensor.is_floating_point() else tensor
        if not torch.equal(stored, reloaded.state_dict()[name]):
            raise RuntimeError(f'保存重载后冻结部分不一致：{name}')
    with torch.inference_mode():
        pred = reloaded(torch.rand(1, 3, 256, 256))
    assert pred.shape == (1, 5, 32, 32) and torch.isfinite(pred).all()
    audit = {'only_simulation': True, 'smoke_only': args.smoke, 'data': str(data),
             'train_images': len(trainer.train_loader.dataset), 'val_images': len(trainer.test_loader.dataset),
             'frozen_ibn_backbone_neck_classifier_and_bn_unchanged': True,
             'kan_grid_and_mask_frozen': True, 'trainable_scope': 'kan_and_edge',
             'changed_branch_parameters': changed, 'best_checkpoint': str(trainer.best),
             'checkpoint_reload_and_forward_ok': True, 'checkpoint_frozen_tensors_match_fp16_storage': True}
    (trainer.save_dir / 'ibn_kan_edge_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    save_progress(trainer, 'completed', True)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
