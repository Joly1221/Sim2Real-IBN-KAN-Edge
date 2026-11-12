"""【新增 IBN-KE-05】CPU 检查组合结构、1024 张量、零残差、冻结、梯度和公共预测。"""
from __future__ import annotations

from copy import deepcopy
import json
from ibn_kan_edge import (BASE_IBN, INITIAL, YOLO, torch, assert_combo, branches_only_mode,
                          frozen_state, check_frozen, source_hashes)
from runtime import ROOT, SIM, sha256


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(42)
    hashes = source_hashes()
    initial_sha = sha256(INITIAL)
    base = YOLO(str(BASE_IBN), task='semantic').model.cpu().float().eval()
    combo = YOLO(str(INITIAL), task='semantic').model.cpu().float().eval()
    assert_combo(combo)
    classifier = combo.model[-1].classifier
    shapes = {}
    hooks = []
    targets = {'P3': combo.model[16], 'kan_projection': classifier.kan_classifier.adapter.fc1,
               'KANLayer': classifier.kan_classifier.adapter.kan,
               'kan_residual': classifier.kan_classifier.adapter.fc3,
               'edge_features': classifier.edge.feature, 'edge_logits': classifier.edge.edge_head,
               'edge_residual': classifier.edge.project}
    for name, module in targets.items():
        hooks.append(module.register_forward_hook(lambda m, x, y, key=name:
                      shapes.update({key: list((y[0] if isinstance(y, tuple) else y).shape)})))
    hooks.append(classifier.kan_classifier.original_classifier.register_forward_pre_hook(
        lambda m, x: shapes.update({'original_classifier_input': list(x[0].shape)})))
    x = torch.rand(1, 3, 1024, 1024)
    with torch.inference_mode():
        expected, actual = base(x), combo(x)
    for hook in hooks:
        hook.remove()
    shapes['logits'] = list(actual.shape)
    wanted = {'P3': [1,64,128,128], 'kan_projection': [1,16,128,128], 'KANLayer': [16384,16],
              'kan_residual': [1,64,128,128], 'edge_features': [1,16,128,128],
              'edge_logits': [1,1,128,128], 'edge_residual': [1,64,128,128],
              'original_classifier_input': [1,64,128,128], 'logits': [1,5,128,128]}
    assert shapes == wanted, shapes
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    classifier.last_edge_logits = None
    fused = deepcopy(combo).fuse(verbose=False, imgsz=256)
    assert_combo(fused)
    with torch.inference_mode():
        small = torch.rand(1,3,256,256)
        unfused_y, fused_y = combo(small), fused(small)
    torch.testing.assert_close(unfused_y, fused_y, rtol=3e-4, atol=5e-4)
    print('1024 结构、零残差输出和融合检查通过', flush=True)

    branches_only_mode(combo)
    reference = frozen_state(combo)
    optimizer = torch.optim.AdamW([p for p in combo.parameters() if p.requires_grad], lr=1e-4)
    trainable_initial = {n: p.detach().clone() for n,p in combo.named_parameters() if p.requires_grad}
    masks = torch.zeros(2,256,256,dtype=torch.long)
    masks[:,:48] = 4
    masks[:,64:180,64:192] = 1
    masks[:,190:230,30:70] = 2
    masks[:,:4,:] = 255
    batch = {'img': torch.rand(2,3,256,256), 'semantic_mask': masks}
    gradients = {}
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, losses = combo.loss(batch)
        assert torch.isfinite(loss).all() and 'crater_edge_loss' in losses
        loss.sum().backward()
        for name, parameter in combo.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                gradients[name] = float(parameter.grad.abs().sum())
            else:
                assert parameter.grad is None, name
        optimizer.step()
        check_frozen(combo, reference)
    selected = ['model.17.classifier.kan_classifier.adapter.kan.coef',
                'model.17.classifier.kan_classifier.adapter.fc1.weight',
                'model.17.classifier.kan_classifier.adapter.fc3.weight',
                'model.17.classifier.edge.feature.0.weight',
                'model.17.classifier.edge.edge_head.weight', 'model.17.classifier.edge.project.weight']
    assert all(gradients[name] > 0 for name in selected)
    changed = [n for n, p in trainable_initial.items() if not torch.equal(p, combo.state_dict()[n])]
    assert all(name in changed for name in selected)
    print('KAN、边缘损失与边缘残差梯度通过；IBN/原网络/BN/网格均未变化', flush=True)
    predictor = YOLO(str(INITIAL), task='semantic')
    image = sorted((SIM/'images/val').glob('*.png'))[0]
    result = predictor.predict(str(image), imgsz=256, device='cpu', save=False, verbose=False)[0]
    assert tuple(result.semantic_mask.data.shape) == result.orig_shape
    assert source_hashes() == hashes and sha256(INITIAL) == initial_sha
    report = {'device': 'cpu', 'formal_training_performed': False, 'synthetic_gradient_steps': 3,
              'shapes': shapes, 'initial_output_exactly_equal_to_trained_ibn': True,
              'fuse_max_difference': float((unfused_y-fused_y).abs().max()),
              'frozen_state_unchanged': True, 'new_branch_gradients': {n: gradients[n] for n in selected},
              'semantic_and_edge_losses': {k: float(v) for k,v in losses.items()},
              'public_predict_and_reload_ok': True, 'source_files_and_initial_weight_unchanged': True}
    destination = ROOT/'checks/ibn_kan_edge_verification.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
