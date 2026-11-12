"""【新增文件 IBN-03】只用仿真数据联合训练骨干、颈部、分类头；默认不启动。"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from runtime import ROOT, INITIAL, SIM, NAMES, YOLO, torch
from ibn_backbone import assert_ibn
from ultralytics.models.yolo.semantic.train import SemanticSegmentationTrainer
from ultralytics.data.dataset import DATASET_CACHE_VERSION
from ultralytics.data.utils import get_hash, img2label_paths, load_dataset_cache_file


class IBNTrainer(SemanticSegmentationTrainer):
    """【新增】复用原训练流程，只检查结构和全模型参数是否进入优化器。"""

    def _build_train_pipeline(self):
        assert_ibn(self.model)
        result = super()._build_train_pipeline()
        expected = {id(p) for p in self.model.parameters()}
        actual = [p for g in self.optimizer.param_groups for p in g['params']]
        if len(actual) != len({id(p) for p in actual}) or {id(p) for p in actual} != expected:
            raise RuntimeError('优化器未且仅包含完整模型参数，或参数重复')
        if not all(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('本实验需要骨干、颈部、分类头均参与训练')
        # 【新增】IN/BN 的仿射参数与原训练器一致，不施加 weight decay。
        norm_ids = {id(p) for p in self.model.model[2].cv1.bn.parameters()}
        if any(g['weight_decay'] != 0 and any(id(p) in norm_ids for p in g['params']) for g in self.optimizer.param_groups):
            raise RuntimeError('IBN 仿射参数意外被施加 weight decay')
        return result


def simulation_data(smoke: bool) -> Path:
    """【新增】正式训练读取已有仿真数据；小检查仅复制 8 张训练、4 张验证图。"""
    if not smoke:
        # 原数据集缓存必须有效；不在原数据目录重建缓存。
        for split in ('train', 'val'):
            images = sorted(str(p) for p in (SIM / 'images' / split).glob('*.png'))
            masks = img2label_paths(images, label_dir='masks', suffix='.png')
            expected = get_hash(images + masks + ['label_mapping:{}', 'nc:5'])
            cache = load_dataset_cache_file(SIM / 'masks' / f'{split}.cache')
            if cache['version'] != DATASET_CACHE_VERSION or cache['hash'] != expected:
                raise RuntimeError(f'{split} 原仿真缓存不匹配；未向源数据写入，请检查数据')
        return SIM / 'data.yaml'
    destination = ROOT / 'smoke_data'
    for split, count in (('train', 8), ('val', 4)):
        images = sorted((SIM / 'images' / split).glob('*.png'))[:count]
        if len(images) != count:
            raise RuntimeError(f'{split} 仿真图像不足')
        for kind in ('images', 'masks'):
            folder = destination / kind / split
            folder.mkdir(parents=True, exist_ok=True)
            for image in images:
                target = folder / image.name
                if not target.exists():
                    shutil.copy2(SIM / kind / split / image.name, target)
    config = destination / 'data.yaml'
    config.write_text(f'path: {destination.as_posix()}\ntrain: images/train\nval: images/val\n'
                      'masks_dir: masks\nnames:\n' + ''.join(f'  {i}: {n}\n' for i, n in enumerate(NAMES)), encoding='utf-8')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true', help='8张仿真训练图/4张验证图，256输入，1轮流程检查')
    mode.add_argument('--train', action='store_true', help='启动正式仿真训练')
    parser.add_argument('--weights', type=Path, default=INITIAL)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default=None, help='正式训练默认0，小检查默认cpu')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--name', default=None)
    args = parser.parse_args()
    if min(args.epochs, args.batch) < 1 or args.imgsz < 64 or args.lr <= 0 or args.patience < 0:
        parser.error('检查 epochs、batch、imgsz、lr、patience 的范围')
    if args.device and ',' in args.device:
        parser.error('此独立脚本当前验证范围为单 GPU 或 CPU')
    torch.set_num_threads(1)
    data = simulation_data(args.smoke)
    # 【原常规入口，注释保留】YOLO(str(args.weights)).train(data=str(data), ...)
    # 【修改 IBN-03】直接将完整模型交给原训练器，避免按原 YAML 重建时把 IBN 丢掉。
    model = YOLO(str(args.weights), task='semantic').model.cpu().float()
    assert_ibn(model)
    model.requires_grad_(True)
    keys = ('model.0.conv.weight', 'model.2.cv1.bn.IN.weight',
            'model.2.cv1.bn.BN.weight', 'model.16.cv1.conv.weight',
            'model.17.classifier.1.weight')
    initial = {k: model.state_dict()[k].detach().clone() for k in keys}
    settings = dict(model=str(args.weights), data=str(data), task='semantic',
                    epochs=1 if args.smoke else args.epochs, imgsz=256 if args.smoke else args.imgsz,
                    batch=2 if args.smoke else args.batch, device=args.device or ('cpu' if args.smoke else '0'),
                    workers=0, project=str(ROOT / 'runs'), name=args.name or ('smoke' if args.smoke else 'ibn_sim_1024'),
                    exist_ok=False, optimizer='AdamW', lr0=args.lr, momentum=.9, weight_decay=.0005,
                    warmup_bias_lr=0.0, warmup_epochs=0.0 if args.smoke else 3.0,
                    nbs=2 if args.smoke else 64, patience=args.patience, seed=42, deterministic=True,
                    freeze=None, amp=False, cache=False, plots=False, save_period=-1,
                    mosaic=0.0, close_mosaic=0, pretrained=True)
    trainer = IBNTrainer(overrides=settings)
    trainer.model = model
    trainer.train()
    reloaded = YOLO(str(trainer.best), task='semantic').model.cpu().float().eval()
    assert_ibn(reloaded)
    with torch.inference_mode():
        prediction = reloaded(torch.rand(1, 3, 256, 256))
    if prediction.shape != (1, 5, 32, 32) or not torch.isfinite(prediction).all():
        raise RuntimeError('训练权重重载后的输出异常')
    changed = {k: not torch.equal(v, trainer.model.state_dict()[k].detach().cpu()) for k, v in initial.items()}
    if not all(changed.values()):
        raise RuntimeError(f'预期参与训练的关键权重没有更新：{changed}')
    report = dict(only_simulation=True, training_scope='backbone_neck_head', data=str(data),
                  train_images=len(trainer.train_loader.dataset), val_images=len(trainer.test_loader.dataset),
                  changed_key_parameters=changed, best_checkpoint=str(trainer.best),
                  last_checkpoint=str(trainer.last), reload_and_forward_ok=True,
                  smoke_only=args.smoke)
    (trainer.save_dir / 'ibn_training_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
