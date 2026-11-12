"""【新增 IBN-REAL-02】真实 train 全量训练 KAN+edge；其余冻结，无验证、无早停。"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
import yaml
from ibn_kan_edge import INITIAL, YOLO, torch, assert_combo, branches_only_mode, frozen_state, source_hashes
from runtime import ROOT, sha256
from train_ibn_kan_edge import IBNKANEdgeTrainer
from prepare_ibn_real_train import DESTINATION, write_yaml
from ultralytics import __version__
from training_progress import write_progress, append_event, replace_with_retry


def restore_training_state(trainer, checkpoint: dict) -> dict:
    """【新增 IBN-REAL-05】恢复优化器、EMA 计数、轮次与原 30 轮学习率计划。"""
    completed = int(checkpoint['epoch']) + 1
    if not 0 < completed < trainer.epochs:
        raise ValueError('检查点轮次必须小于目标总轮次')
    if checkpoint.get('optimizer') is None:
        raise ValueError('检查点没有优化器状态，不能按此入口恢复')
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    if checkpoint.get('scaler'):
        trainer.scaler.load_state_dict(checkpoint['scaler'])
    trainer.ema.ema.load_state_dict(checkpoint['model'].float().state_dict())
    trainer.ema.updates = int(checkpoint.get('updates') or 0)
    trainer.start_epoch = completed
    trainer.scheduler.last_epoch = completed - 1
    record = {'completed_epochs_before_resume': completed, 'next_epoch': completed + 1,
              'total_epochs': trainer.epochs, 'optimizer_state_restored': True,
              'ema_updates_restored': trainer.ema.updates,
              'optimizer_lrs_at_restore': [group['lr'] for group in trainer.optimizer.param_groups],
              'model_restore_basis': 'live_model' if checkpoint.get('resume_model') is not None else 'saved_ema_model',
              'bitwise_identical_continuation_claimed': False}
    return record


class RealTrainOnlyTrainer(IBNKANEdgeTrainer):
    """【新增】复用已验证的冻结/优化器/EMA，仅替换数据与训练后处理。"""

    def _setup_train(self):
        result = super()._setup_train()
        # 【原框架】metrics 初始化为验证指标的零值；本实验不生成伪验证指标。
        self.metrics = {}
        self.fitness = None
        self.best_fitness = None
        self.validation_calls_skipped = 0
        if getattr(self, 'resume_bundle', None) is not None:
            record = restore_training_state(self, self.resume_bundle)
            self.check_frozen()
            (self.save_dir/'resume_restore.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            print('RESUME '+json.dumps(record),flush=True)
        print('REAL TRAIN ONLY: 15025 trainable parameters; no validation, no early stopping.',flush=True)
        return result

    def validate(self):
        # 【原调用，注释保留】return super().validate()
        # val=False 时原框架仍会在最后一轮调用 validate；这里明确跳过。
        self.validation_calls_skipped += 1
        return {}, None

    def final_eval(self):
        # 【原调用，注释保留】super().final_eval()  # 原框架默认再次评价 best.pt。
        print('Validation remains disabled. Saved complete model: '+str(self.last),flush=True)

    def save_metrics(self, metrics):
        # 【原入口，注释保留】super().save_metrics(metrics)
        # 【修改 IBN-REAL-03】框架首批次会再次自动插入 val/* 零值。
        # 只写真实训练损失和学习率，避免伪验证列和最后一轮 CSV 列数变化。
        training_only = {key: value for key, value in metrics.items()
                         if key.startswith('train/') or key.startswith('lr/')}
        return super().save_metrics(training_only)

    def save_model(self):
        """【新增】只保存最后一轮完整模型；无验证时不生成名为 best 的权重。"""
        self.check_frozen()
        snapshot = deepcopy(self.ema.ema).cpu().float().eval()
        snapshot.criterion = None
        snapshot.model[-1].classifier.last_edge_logits = None
        # 【新增】以后检查点同时保存实时训练模型，避免只剩 EMA 模型用于恢复。
        resume_model = deepcopy(snapshot)
        resume_model.load_state_dict(self.model.state_dict())
        if any(not torch.isfinite(v).all() for v in snapshot.state_dict().values() if v.is_floating_point()):
            raise RuntimeError('模型包含非有限值，停止保存')
        # 【原框架采用 FP16】此独立训练入口保存 FP32，冻结网格与底座可逐位核对。
        checkpoint = {'epoch':self.epoch,'model':snapshot,'resume_model':resume_model,'ema':None,'updates':self.ema.updates,
                      'optimizer':self.optimizer.state_dict(),'scaler':self.scaler.state_dict(),
                      'best_fitness':None,'train_args':vars(self.args),'train_results':self.read_results_csv(),
                      'train_metrics':self.label_loss_items(self.tloss),'version':__version__,
                      'date':datetime.now().astimezone().isoformat(),
                      'real_adapter_training':{'selection':'last_epoch_ema','validation_enabled':False,
                                               'test_used':False,'scope':'kan_and_edge','source_only':False}}
        temporary = self.last.with_suffix('.tmp')
        torch.save(checkpoint,temporary)
        # 【原代码保留】temporary.replace(self.last)
        replace_with_retry(temporary,self.last)
        return True


def verify_data(data_yaml: Path) -> tuple[Path,dict]:
    config = yaml.safe_load(data_yaml.read_text(encoding='utf-8'))
    data_root = Path(config['path'])
    provenance = json.loads((data_root/'provenance.json').read_text(encoding='utf-8'))
    assert provenance['all_images_used_for_training'] and not provenance['validation_enabled']
    assert not provenance['test_used'] and config['train'] == config['val'] == 'images/train'
    assert sha256(Path(provenance['source'])/'annotations.json') == provenance['annotation_sha256']
    for row in provenance['records']:
        assert sha256(data_root/'images/train'/row['file']) == row['sha256']
        assert sha256(data_root/'masks/train'/(Path(row['file']).stem+'.png')) == row['mask_sha256']
    return data_root,provenance


def smoke_data(data_root: Path, provenance: dict) -> Path:
    """【新增】仅从 train 拷贝 8 张做一轮流程检查，不涉及验证或测试数据。"""
    target = ROOT/'real_adapter_smoke_data'
    if target.exists():
        raise FileExistsError(f'小样本目录已存在：{target}')
    # 先选含坑与岩石标注的图像，剩余按原顺序补足，不依据模型预测挑选。
    chosen = []
    for class_id in [1,2,4]:
        row = next(r for r in provenance['records'] if r['pixels_by_class'][class_id]>0)
        if row not in chosen:
            chosen.append(row)
    for row in provenance['records']:
        if len(chosen)>=8:
            break
        if row not in chosen:
            chosen.append(row)
    for kind in ['images','masks']:
        (target/kind/'train').mkdir(parents=True)
        for row in chosen:
            name = row['file'] if kind=='images' else Path(row['file']).stem+'.png'
            shutil.copy2(data_root/kind/'train'/name,target/kind/'train'/name)
    return write_yaml(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--train',action='store_true')
    mode.add_argument('--smoke',action='store_true')
    parser.add_argument('--data',type=Path,default=DESTINATION/'data.yaml')
    parser.add_argument('--weights',type=Path,default=INITIAL)
    parser.add_argument('--resume-from',type=Path,default=None,
                        help='从本入口生成的 last.pt 恢复；--epochs 仍表示累计目标总轮次')
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--imgsz',type=int,default=1024)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--device',default=None)
    parser.add_argument('--name',default=None)
    args = parser.parse_args()
    if min(args.epochs,args.batch)<1 or args.imgsz<64 or args.lr<=0 or (args.device and ',' in args.device):
        parser.error('请使用有效参数和单 GPU 或 CPU')
    if args.resume_from and args.smoke:
        parser.error('恢复训练不能同时使用 --smoke')
    torch.set_num_threads(1)
    protected = source_hashes()
    start_weights = args.resume_from or args.weights
    initial_sha = sha256(start_weights)
    data_root,provenance = verify_data(args.data)
    data = smoke_data(data_root,provenance) if args.smoke else args.data
    loaded = YOLO(str(start_weights),task='semantic')
    resume_bundle = loaded.ckpt if args.resume_from else None
    model = loaded.model.cpu().float()
    if resume_bundle is not None:
        previous = resume_bundle.get('train_args',{})
        for key,current in [('epochs',args.epochs),('imgsz',args.imgsz),('batch',args.batch),('lr0',args.lr),('data',str(args.data))]:
            if previous.get(key)!=current:
                raise ValueError(f'为保留原训练计划，恢复时 {key} 必须仍为 {previous.get(key)}')
        if resume_bundle.get('real_adapter_training',{}).get('scope')!='kan_and_edge':
            raise ValueError('只接受本真实适配器训练入口保存的检查点')
        if resume_bundle.get('resume_model') is not None:
            model = resume_bundle['resume_model'].cpu().float()
    assert_combo(model)
    # 【修改】此实验明确使用真实数据，更新模型元数据，避免误标为零样本模型。
    model.ibn_config = {**model.ibn_config,'source_only':False,'training_scope':'frozen_ibn_base'}
    model.ibn_kan_edge_config = {**model.ibn_kan_edge_config,'source_only':False,
                                'real_supervised_finetuning':True,'selection':'last_epoch_ema'}
    branches_only_mode(model)
    frozen = frozen_state(model)
    original_frozen = frozen_state(YOLO(str(INITIAL),task='semantic').model.cpu().float())
    assert frozen.keys()==original_frozen.keys()
    assert all(torch.equal(value,original_frozen[name]) for name,value in frozen.items())
    initial = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    assert sum(t.numel() for t in initial.values())==15025
    settings = dict(model=str(start_weights),data=str(data),task='semantic',
                    epochs=1 if args.smoke else args.epochs,imgsz=256 if args.smoke else args.imgsz,
                    batch=2 if args.smoke else args.batch,device=args.device or ('cpu' if args.smoke else '0'),
                    project=str(ROOT/'runs'),name=args.name or ('ibn_kan_edge_real_smoke' if args.smoke else 'ibn_kan_edge_real_train_1024'),
                    exist_ok=False,optimizer='AdamW',lr0=args.lr,momentum=.9,weight_decay=.0005,
                    nbs=2 if args.smoke else args.batch,warmup_epochs=0. if args.smoke else 1.,warmup_bias_lr=0.,
                    val=False,patience=0,freeze=None,amp=False,workers=0,cache=False,plots=False,
                    seed=42,deterministic=True,mosaic=0.,close_mosaic=0,pretrained=True)
    trainer = RealTrainOnlyTrainer(overrides=settings)
    trainer.model = model
    trainer.frozen_reference = frozen
    trainer.resume_bundle = resume_bundle
    progress = {'batch':0,'last':0.}

    def update(t,stage: str,force: bool=False) -> None:
        now = time.monotonic()
        if not force and now-progress['last']<10:
            return
        state = {'stage':stage,'epoch':t.epoch+1,'epochs':t.epochs,'batch':progress['batch'],
                 'batches_per_epoch':len(t.train_loader),'train_images':len(t.train_loader.dataset),
                 'trainable_parameters':15025,'validation_enabled':False,'test_used':False,
                 'updated_local_time':time.strftime('%Y-%m-%d %H:%M:%S')}
        path = t.save_dir/'progress.json'
        # 【原代码保留】temp.write_text(...); temp.replace(path)
        write_progress(path,state)  # 【修改】进度文件占用时仅记录告警，不终止训练。
        progress['last']=now

    def epoch_start(t) -> None:
        progress['batch']=0
        update(t,'training',True)

    def batch_end(t) -> None:
        progress['batch']+=1
        update(t,'training')

    def epoch_end(t) -> None:
        t.check_frozen()
        record = {'epoch':t.epoch+1,'frozen_state_unchanged':True,'frozen_state_tensors':len(frozen),
                  'trainable_parameter_count':sum(p.numel() for p in t.model.parameters() if p.requires_grad),
                  'validation_enabled':False}
        append_event(t.save_dir/'freeze_checks.jsonl',record)
        update(t,'epoch_saved',True)

    trainer.add_callback('on_train_epoch_start',epoch_start)
    trainer.add_callback('on_train_batch_end',batch_end)
    trainer.add_callback('on_fit_epoch_end',epoch_end)
    try:
        trainer.train()
        trainer.check_frozen()
        live = trainer.model.state_dict()
        changed = [n for n,p in initial.items() if not torch.equal(p,live[n].detach().cpu())]
        assert any('.adapter.' in n for n in changed) and any('.classifier.edge.' in n for n in changed)
        reloaded = YOLO(str(trainer.last),task='semantic').model.cpu().float().eval()
        assert_combo(reloaded)
        assert all(torch.equal(t,reloaded.state_dict()[n]) for n,t in frozen.items())
        with torch.inference_mode():
            prediction = reloaded(torch.rand(1,3,256,256))
        assert prediction.shape==(1,5,32,32) and torch.isfinite(prediction).all()
        assert not trainer.best.exists()
        assert source_hashes()==protected and sha256(start_weights)==initial_sha
        report = {'training_domain':'real_supervised','only_provided_train_used':True,'smoke_only':args.smoke,
                  'source':provenance['source'],'data':str(data),'train_images':len(trainer.train_loader.dataset),
                  'validation_enabled':False,'test_used':False,'selection':'last_epoch_ema',
                  'checkpoint':str(trainer.last),'checkpoint_sha256':sha256(trainer.last),
                  'frozen_ibn_backbone_neck_classifier_bn_grid_unchanged':True,
                  'changed_branch_parameters':changed,'trainable_parameter_count':15025,
                  'checkpoint_reload_and_forward_ok':True,'source_files_unchanged':True}
        report.update(resumed_from=str(args.resume_from) if args.resume_from else None,
                      completed_epochs_before_resume=int(resume_bundle['epoch'])+1 if resume_bundle else 0,
                      total_completed_epochs=int(trainer.epoch)+1,
                      model_restore_basis=('live_model' if resume_bundle.get('resume_model') is not None else 'saved_ema_model') if resume_bundle else 'initial_model')
        (trainer.save_dir/'real_adapter_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        update(trainer,'completed',True)
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    except Exception:
        if hasattr(trainer,'train_loader') and hasattr(trainer,'epoch'):
            update(trainer,'failed',True)
        raise


if __name__=='__main__':
    main()
