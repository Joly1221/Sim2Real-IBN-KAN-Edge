"""【新增 IBN-REAL-06】回归检查 Windows 进度文件占用与第 9 轮检查点恢复。"""
from __future__ import annotations

from copy import deepcopy
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import warnings

from training_progress import write_progress
from train_ibn_kan_edge_real import restore_training_state, RealTrainOnlyTrainer
# 【整理前代码保留】 from ibn_kan_edge import YOLO, torch, INITIAL, branches_only_mode, frozen_state, check_frozen
# 【整理修改 CODE-PACKAGE-02】使用本副本中的模块与资源。
from ibn_kan_edge import YOLO, torch, INITIAL, branches_only_mode, frozen_state, check_frozen
from runtime import ROOT
from ultralytics.utils.torch_utils import ModelEMA


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='ibn_progress_regression_') as directory:
        target = Path(directory)/'progress.json'
        target.write_text('{"epoch":9}',encoding='utf-8')
        kernel = ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
                                      wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        # Windows 读句柄允许读写，但不允许删除/替换，实际重现原来的占用错误。
        handle = kernel.CreateFileW(str(target),0x80000000,3,None,3,128,None)
        assert handle != ctypes.c_void_p(-1).value, ctypes.get_last_error()
        try:
            assert write_progress(target,{'epoch':10}) is False
            assert json.loads(target.read_text())['epoch']==9
            backup = json.loads((target.parent/'progress_events.jsonl').read_text(encoding='utf-8'))
            assert backup['epoch']==10 and backup['progress_file_error']
        finally:
            assert kernel.CloseHandle(handle)
        assert write_progress(target,{'epoch':11}) is True
        assert json.loads(target.read_text())['epoch']==11
        assert not list(target.parent.glob('*.tmp'))

# 【整理前代码保留】     path = Path(r'D:\Projects\Sim2real\imp\IBN\runs\ibn_kan_edge_real_train_1024_v2\weights\last.pt')
# 【整理修改 CODE-PACKAGE-02】使用本副本中的模块与资源。
    path = ROOT/'runs/ibn_kan_edge_real_train_1024_v2/weights/last.pt'
    loaded = YOLO(str(path),task='semantic')
    model = loaded.model.cpu().float()
    branches_only_mode(model)
    reference = frozen_state(YOLO(str(INITIAL),task='semantic').model.cpu().float())
    check_frozen(model,reference)
    optimizer = RealTrainOnlyTrainer.build_optimizer(None,model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda x:(1-x/30)*.99+.01)
    dummy = SimpleNamespace(epochs=30,optimizer=optimizer,scheduler=scheduler,
                            scaler=torch.amp.GradScaler('cpu',enabled=False),ema=ModelEMA(model))
    record = restore_training_state(dummy,loaded.ckpt)
    assert dummy.start_epoch==9 and dummy.ema.updates==342
    assert len(optimizer.state)==13
    assert all(int(state['step'])==342 for state in optimizer.state.values())
    assert all(abs(group['lr']-7.36e-5)<1e-12 for group in optimizer.param_groups)
    assert all(torch.equal(value,dummy.ema.ema.state_dict()[name]) for name,value in reference.items())
    assert sum(p.numel() for group in optimizer.param_groups for p in group['params'])==15025
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        scheduler.step()
    assert all(abs(group['lr']-7.03e-5)<1e-12 for group in optimizer.param_groups)
    # 新格式保存训练中的模型时，也应选择同一个恢复轮次与优化器计划。
    new_format = {**loaded.ckpt,'resume_model':deepcopy(model)}
    assert restore_training_state(dummy,new_format)['model_restore_basis']=='live_model'
    print(json.dumps({'windows_file_lock_reproduced':True,'training_not_interrupted':True,
                      'progress_recovers_after_unlock':True,'optimizer_parameter_count':15025,
                      'resume_verified':record,'next_epoch_lr':7.03e-5},indent=2))


if __name__=='__main__':
    main()
