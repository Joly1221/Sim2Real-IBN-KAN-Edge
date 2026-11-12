"""【新增文件 IBN-05】CPU 验证迁移、形状、梯度、融合、保存重载；不做正式训练。"""
from __future__ import annotations

from copy import deepcopy
import json

from runtime import ROOT, BASE, INITIAL, SIM, YOLO, torch, protected_hashes
from ibn_backbone import IBN, ConvWithIBN, assert_ibn


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    hashes = protected_hashes()
    original = YOLO(str(BASE), task='semantic').model.cpu().float().eval()
    model = YOLO(str(INITIAL), task='semantic').model.cpu().float().eval()
    assert_ibn(model)
    old = original.model[2].cv1.bn
    new = model.model[2].cv1.bn
    pairs = [(new.IN.weight, old.weight[:16]), (new.IN.bias, old.bias[:16]),
             (new.BN.weight, old.weight[16:]), (new.BN.bias, old.bias[16:]),
             (new.BN.running_mean, old.running_mean[16:]), (new.BN.running_var, old.running_var[16:])]
    assert all(torch.equal(a, b) for a, b in pairs)
    original_state = original.state_dict()
    current = model.state_dict()
    for name, tensor in original_state.items():
        if not name.startswith('model.2.cv1.bn.'):
            assert torch.equal(tensor, current[name]), name
    assert new.IN.running_mean is None and new.IN.running_var is None
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in original.parameters())
    shapes = {}
    handles = []
    for name, module in [('projection', model.model[2].cv1.conv), ('IN', new.IN), ('BN', new.BN),
                         ('IBN', new), ('shallow_C3k2', model.model[2]), ('P3', model.model[16])]:
        handles.append(module.register_forward_hook(lambda m, x, y, key=name: shapes.update({key: list(y.shape)})))
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 1024, 1024))
    for handle in handles:
        handle.remove()
    shapes['segmentation_logits'] = list(output.shape)
    expected_shapes = {'projection': [1, 32, 256, 256], 'IN': [1, 16, 256, 256],
                       'BN': [1, 16, 256, 256], 'IBN': [1, 32, 256, 256],
                       'shallow_C3k2': [1, 64, 256, 256], 'P3': [1, 64, 128, 128],
                       'segmentation_logits': [1, 5, 128, 128]}
    assert shapes == expected_shapes and torch.isfinite(output).all(), shapes
    print('1024 张量形状检查通过', shapes, flush=True)

    # 【新增】检查正常推理时 BN 不被更新，IN 不积累运行统计。
    stats = {k: v.clone() for k, v in new.BN.state_dict().items()}
    samples = torch.rand(2, 32, 32, 32)
    with torch.inference_mode():
        alone = new(samples[:1])
        together = new(samples)[:1]
    torch.testing.assert_close(alone, together, rtol=1e-5, atol=1e-5)
    assert all(torch.equal(v, new.BN.state_dict()[k]) for k, v in stats.items())

    # 【新增】标准自动融合只合并普通 Conv/BN，IBN 容器应完整保留。
    fused = deepcopy(model).fuse(verbose=False, imgsz=256)
    assert_ibn(fused)
    x = torch.rand(1, 3, 256, 256)
    with torch.inference_mode():
        reference, actual = model(x), fused(x)
    torch.testing.assert_close(reference, actual, rtol=2e-4, atol=3e-4)
    fuse_max_difference = float((reference - actual).abs().max())
    print('自动融合检查通过，最大差值', fuse_max_difference, flush=True)

    # 【新增】用一次合成标签反向检查，确认 IN、骨干、颈部、分类头都可更新。
    model.train().requires_grad_(True)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    labels = torch.zeros(2, 256, 256, dtype=torch.long)
    labels[:, :48] = 4
    labels[:, 80:160, 64:192] = 1
    labels[:, 180:220, 32:64] = 2
    labels[:, 48:64] = 3
    loss, items = model.loss({'img': torch.rand(2, 3, 256, 256), 'semantic_mask': labels})
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    keys = ('model.0.conv.weight', 'model.2.cv1.bn.IN.weight', 'model.2.cv1.bn.IN.bias',
            'model.2.cv1.bn.BN.weight', 'model.16.cv1.conv.weight', 'model.17.classifier.1.weight')
    parameters = dict(model.named_parameters())
    gradients = {}
    for key in keys:
        grad = parameters[key].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, key
        gradients[key] = float(grad.abs().sum())
    optimizer.step()
    assert all(not torch.equal(before[k], parameters[k]) for k in keys)
    assert new.BN.num_batches_tracked > stats['num_batches_tracked']
    print('损失和骨干/IBN/颈部/分类头梯度检查通过', flush=True)

    # 【新增】完整公共预测入口实际执行一次（仿真图），覆盖 AutoBackend 融合和后处理。
    source = next(iter(sorted((SIM / 'images/val').glob('*.png'))))
    predictor = YOLO(str(INITIAL), task='semantic')
    prediction = predictor.predict(str(source), imgsz=256, device='cpu', save=False, verbose=False)[0]
    assert prediction.semantic_mask is not None
    assert tuple(prediction.semantic_mask.data.shape[-2:]) == prediction.orig_shape
    assert protected_hashes() == hashes
    audit = {'device': 'cpu', 'gpu_used': False, 'only_synthetic_and_simulation_inputs': True,
             'shape_checks': shapes, 'parameter_and_buffer_transfer_ok': True,
             'untouched_original_tensors_equal': True, 'parameter_count': sum(p.numel() for p in model.parameters()),
             'eval_statistics_unchanged': True, 'fuse_max_difference': fuse_max_difference,
             'key_gradient_l1': gradients, 'semantic_loss': {k: float(v) for k, v in items.items()},
             'checkpoint_reload_ok': True, 'standard_predict_ok': True,
             'protected_source_files_unchanged': True, 'protected_files': hashes,
             'formal_training_performed': False}
    destination = ROOT / 'checks'
    destination.mkdir(exist_ok=True)
    (destination / 'verification.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
