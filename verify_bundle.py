"""【新增 CODE-PACKAGE-03】验证整理副本的模块来源、完整模型、冻结和前向。"""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
import ibn_kan_edge
from ibn_kan_edge import YOLO, torch, assert_combo, branches_only_mode, frozen_state, check_frozen, source_hashes
from runtime import sha256


def main() -> None:
    torch.set_num_threads(1)
    # 【新增】导入全部当前顶层入口；main guard 防止导入时启动训练或预测。
    imported=[]
    for path in sorted(ROOT.glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
        if path.stem!=Path(__file__).stem:
            module=importlib.import_module(path.stem)
            assert Path(module.__file__).resolve().is_relative_to(ROOT),str(module.__file__)
            imported.append(path.name)
    origins={}
    for name in ['runtime','ibn_backbone','ibn_kan_edge','kan_adapter','edge_branch','ultralytics','kan','sklearn','joblib','threadpoolctl']:
        module=importlib.import_module(name)
        origin=Path(module.__file__).resolve()
        assert origin.is_relative_to(ROOT),(name,str(origin))
        origins[name]=str(origin)
    weights=ROOT/'runs/ibn_kan_edge_real_train_1024_resume/weights/last.pt'
    digest_before=sha256(weights)
    assert digest_before=='5eba98606a50a81b9de7ac4f99446ac07a37c8274d0bad013b0360c18cfe7c31'
    wrapper=YOLO(str(weights),task='semantic')
    model=wrapper.model.cpu().float().eval()
    assert_combo(model)
    assert wrapper.ckpt['epoch']==29
    shapes={}
    handle=model.model[16].register_forward_hook(lambda m,x,y:shapes.update(P3=list(y.shape)))
    torch.manual_seed(20261112)
    with torch.inference_mode():
        output=model(torch.rand(1,3,1024,1024))
    handle.remove()
    shapes['logits']=list(output.shape)
    assert shapes=={'P3':[1,64,128,128],'logits':[1,5,128,128]}
    assert torch.isfinite(output).all()
    output_hash=hashlib.sha256(output.numpy().tobytes()).hexdigest()
    expected=json.loads((ROOT/'metadata/original_forward_reference.json').read_text(encoding='utf-8'))
    assert output_hash==expected['output_sha256'],'整理前后前向输出不同'
    branches_only_mode(model)
    fixed=frozen_state(model)
    check_frozen(model,fixed)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==15025
    from train_ibn_kan_edge import IBNKANEdgeTrainer
    optimizer=IBNKANEdgeTrainer.build_optimizer(None,model)
    selected={id(p) for p in model.parameters() if p.requires_grad}
    assert selected=={id(p) for group in optimizer.param_groups for p in group['params']}
    # 只核对参数集合；不做反向传播或 optimizer.step，不修改模型权重。
    protected=source_hashes()
    assert all(Path(path).resolve().is_relative_to(ROOT) for path in protected)
    assert sha256(weights)==digest_before
    manifest=json.loads((ROOT/'COPY_MANIFEST.json').read_text(encoding='utf-8'))
    for record in manifest['files']:
        assert sha256(ROOT/record['copy'])==record['copied_sha256'],record['copy']
    manifest['verification']='passed; see PACKAGE_VERIFICATION.json'
    (ROOT/'COPY_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    report={'passed':True,'code_root':str(ROOT),'imported_entry_points':imported,'module_origins':origins,
            'copied_file_hashes_verified':len(manifest['files']),'full_checkpoint_sha256':digest_before,
            'checkpoint_epoch':30,'shapes':shapes,'output_sha256':output_hash,
            'exact_forward_match_with_original':True,'trainable_parameters':15025,'frozen_state_tensors':len(fixed),
            'optimizer_contains_only_branches':True,'no_training_performed':True,
            'old_code_directory_required_for_import':False,'original_source_files_unchanged_during_copy':manifest['source_files_unchanged'],
            'data_not_copied':True,'python_environment_not_copied':True}
    (ROOT/'PACKAGE_VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
