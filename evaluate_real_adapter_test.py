"""【新增 IBN-REAL-EVAL-01】完整 IBN+KAN+edge 真实微调模型的固定测试。

复用原评价的标签栅格化、调色板与统计函数；原评价程序、权重和标注不修改。
同批预测仿真 IBN 基线，避免使用不同子集的汇总百分比进行比较。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import html
import json
from pathlib import Path
import time

from ibn_kan_edge import YOLO, torch, assert_combo, BASE_IBN
from ibn_backbone import assert_ibn
# 【整理前代码保留】 from runtime import ROOT, SIM, NAMES
# 【整理修改 CODE-PACKAGE-02】使用本副本中的模块与资源。
from runtime import ROOT, SIM, NAMES, PALETTE
from evaluate_kaggle_test import (CLASSES, MODEL_LABELS, OLD_HASHES, sha256, dump, rasterize,
                                  palette_mask, overlay, class_metrics, np, Image, ImageDraw,
                                  ImageFont, plt)

LABELS = {'ibn_baseline':'Simulation-trained IBN', 'finetuned':'IBN + KAN + edge; real fine-tuned'}
ZH = ['月壤','撞击坑','岩石','山体','天空']


def add_area_diagnostics(payload, rows):
    """【新增】检查类别占比漂移；未知区域的预测不能直接当作误检真值。"""
    diagnostics={}
    for key in payload['models']:
        all_pixels=np.array([json.loads(row[key+'_all_image_pixels_by_class']) for row in rows],dtype=np.int64).sum(axis=0)
        annotated=np.asarray(payload['models'][key]['matrix_rows_truth_columns_prediction']).sum(axis=0)
        unknown=all_pixels-annotated
        assert (unknown>=0).all()
        diagnostics[key]={'predicted_pixels_by_class':all_pixels.tolist(),
                          'predicted_percentage_by_class':(all_pixels/all_pixels.sum()*100).tolist(),
                          'unknown_region_prediction_percentage_by_class':(unknown/unknown.sum()*100).tolist()}
    payload['prediction_area_diagnostics']=diagnostics


def add_five_class_metrics(payload):
    """【新增 IBN-REAL-EVAL-03】展示五类；无真值类别不伪造精度或纳入平均。"""
    for model in payload['models'].values():
        matrix=np.asarray(model['matrix_rows_truth_columns_prediction'],dtype=np.int64)
        truth_rows={entry['model_class_id']:index for index,entry in enumerate(model['per_class'])}
        entries=[]
        for class_id,name in enumerate(NAMES):
            row=truth_rows.get(class_id)
            support=int(matrix[row].sum()) if row is not None else 0
            predicted=int(matrix[:,class_id].sum())
            tp=int(matrix[row,class_id]) if support else None
            fp=predicted-tp if support else None
            fn=support-tp if support else None
            entries.append({'model_class_id':class_id,'class':name,'class_zh':ZH[class_id],
                            'annotated_pixels':support,'predicted_pixels_within_annotated_regions':predicted,
                            'true_positive':tp,'false_positive':fp,'false_negative':fn,
                            'iou':tp/(tp+fp+fn) if support else None,
                            'precision':tp/predicted if support and predicted else None,
                            'recall':tp/support if support else None,
                            'status':'annotated_regions_only' if support else 'no_ground_truth_annotation'})
        model['five_class_metrics']=entries
        scored=[entry['iou'] for entry in entries if entry['iou'] is not None]
        model['annotated_region_miou']=float(np.mean(scored)) if scored else None
        model['miou_class_ids']=[entry['model_class_id'] for entry in entries if entry['iou'] is not None]
        model['full_image_five_class_miou']=None
    payload['additional_metrics_scope']='IoU, precision and recall on annotated pixels only; unsupported classes excluded from mean'


def make_report(out, payload, rows, colors):
    """【新增】报告真实训练设定，明确区别于零样本实验及最佳验证权重。"""
    before = np.array([x['annotated_pixel_recall'] for x in payload['models']['ibn_baseline']['per_class']])*100
    after = np.array([x['annotated_pixel_recall'] for x in payload['models']['finetuned']['per_class']])*100
    matrix = np.asarray(payload['models']['finetuned']['matrix_rows_truth_columns_prediction'])
    percentages = matrix/matrix.sum(axis=1,keepdims=True)*100
    figures = out/'figures'
    figures.mkdir(exist_ok=True)  # 【新增】可用保存的计数重新生成报告，不重复推理。
    area=payload['prediction_area_diagnostics']['finetuned']
    area_percent=area['predicted_percentage_by_class']
    unknown_percent=area['unknown_region_prediction_percentage_by_class']
    caution=(f'坑召回升高伴随明显的类别预测偏移：全部图像像素的 {area_percent[1]:.2f}% 被判为坑，'
             f'仅 {area_percent[0]:.2f}% 被判为月壤；{percentages[1,1]:.2f}% 的岩石标注像素被误分为坑。'
             f'未标注区域中 {unknown_percent[1]:.2f}% 被判为坑。未知区域不能直接全部算作误检，'
             '但结合四联图中大面积地面被覆盖为坑的现象，不能将高召回解释为坑识别问题已经解决。')
    fig, ax = plt.subplots(figsize=(8,4.7),layout='constrained')
    for offset, values, label, color in [(-.18,before,LABELS['ibn_baseline'],'#899bb0'),(.18,after,LABELS['finetuned'],'#287c87')]:
        bars = ax.bar(np.arange(3)+offset,values,.35,label=label,color=color)
        ax.bar_label(bars,fmt='%.2f',padding=3)
    ax.set(xticks=range(3),xticklabels=['Crater','Rock','Sky'],ylim=(0,115),
           ylabel='Recall on annotated pixels (%)',title=f"Same {payload['images']} test images; FP32, input 1024")
    ax.legend(fontsize=8,loc='upper left')
    for ext in ['png','pdf']:
        fig.savefig(figures/f'annotated_recall.{ext}',dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,3.8),layout='constrained')
    heat = ax.imshow(percentages,vmin=0,vmax=100,cmap='Blues',aspect='auto')
    ax.set(xticks=range(5),xticklabels=MODEL_LABELS,yticks=range(3),yticklabels=['Crater','Rock','Sky'],
           xlabel='Predicted class',ylabel='Annotated class',title='Full fine-tuned model: prediction direction (%)')
    for y in range(3):
        for x in range(5):
            ax.text(x,y,f'{percentages[y,x]:.2f}',ha='center',va='center',color='white' if percentages[y,x]>55 else '#162838')
    fig.colorbar(heat,ax=ax,label='Row percentage')
    for ext in ['png','pdf']:
        fig.savefig(figures/f'prediction_directions.{ext}',dpi=170)
    plt.close(fig)
    table = ['| 已标注类别 | 仿真 IBN | 完整微调模型 | 差值（百分点） | 有效标注像素 |','|---|---:|---:|---:|---:|']
    html_rows = []
    for i, row in enumerate(payload['models']['finetuned']['per_class']):
        table.append(f"| {row['class_zh']} | {before[i]:.4f}% | {after[i]:.4f}% | {after[i]-before[i]:+.4f} | {row['annotated_pixels']:,} |")
        html_rows.append(f"<tr><td>{row['class_zh']}</td><td>{before[i]:.2f}%</td><td>{after[i]:.2f}%</td><td>{after[i]-before[i]:+.2f}</td></tr>")
    # 【新增】原三类召回对照保留，另列五类 IoU/精确率/召回率及缺标状态。
    add_five_class_metrics(payload)
    five_table=['| 类别 | 有效标注像素 | IoU | 精确率 | 召回率 |', '|---|---:|---:|---:|---:|']
    five_html=[]
    percentage=lambda value:'无法计算（无真值）' if value is None else f'{value*100:.2f}%'
    for entry in payload['models']['finetuned']['five_class_metrics']:
        cells=[entry['class_zh'],f"{entry['annotated_pixels']:,}",percentage(entry['iou']),percentage(entry['precision']),percentage(entry['recall'])]
        five_table.append('| '+' | '.join(cells)+' |')
        five_html.append('<tr>'+''.join(f'<td>{html.escape(cell)}</td>' for cell in cells)+'</tr>')
    current_miou=payload['models']['finetuned']['annotated_region_miou']*100
    baseline_miou=payload['models']['ibn_baseline']['annotated_region_miou']*100
    metric_note=(f'已标注区域的三类 mIoU：完整模型 {current_miou:.2f}%，仿真 IBN {baseline_miou:.2f}%。'
                 '五类全图 mIoU：无法计算。月壤、山体没有有效真值标注，“无法计算”不表示 0% 或预测中没有该类。'
                 '表中精确率、召回率和 IoU 都只统计已标注区域，未标注区域的假阳性不计入，不能代表全图表现。')
    export_rows=[{'model':key,**entry} for key,model in payload['models'].items() for entry in model['five_class_metrics']]
    with (out/'five_class_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(export_rows[0]))
        writer.writeheader()
        writer.writerows(export_rows)
    limitations = (
        # 【原口径保留说明】最初仅展示三类像素召回；现新增同一区域内的 IoU 与精确率。
        '所有指标只统计已标注区域；未知和跨类别重叠区域不参与评价。'
        '这些不是全图指标或实例召回率，不能由高召回断言误检少。'
        f"与 150 张训练图像的文件 SHA-256 重合数为 {payload['train_test_overlap']['matching_images']}。"
        '文件不重复不代表场景独立，未核验相邻视角或同场景关系。'
        f"其中 {payload['previous_200_overlap']['matching_images']} 张曾用于此前评价，因此不是全新独立盲测。"
    )
    report = ['# IBN＋KAN＋边缘分支完整模型测试','',
              f"测试数据：`{payload['source']}`，{payload['images']} 张图、{payload['annotations']} 个多边形。",
              f"完整权重：`{payload['models']['finetuned']['weights']}`。",
              '权重为真实 train 的 150 张图训练至累计第 30 轮后的完整 EMA 模型；不是独立适配器文件。',
              '本次只推理，不更新任何模型参数。测试结果未用于选择轮次或阈值。',
              '问题：该完整微调模型在这 46 张图上的已标注区域召回率如何，相比仿真 IBN 底座有何变化？','',
              '## 五类指标','',*five_table,'',metric_note,'',
              '计算方式：IoU = TP/(TP+FP+FN)，精确率 = TP/(TP+FP)，召回率 = TP/(TP+FN)。'
              'TP、FP、FN 均限制在有效标注区域；mIoU 只平均有真值的坑、岩石、天空三类。','',
              '## 召回率对照与预测偏移','',caution,'',*table,'',
              f'坑像素召回变化 {after[0]-before[0]:+.4f} 个百分点；岩石变化 {after[1]-before[1]:+.4f}；天空变化 {after[2]-before[2]:+.4f}。',
              '该比较同时包含新增模块和真实标签训练的影响，不能把差值单独归因于 KAN、边缘损失或 IBN。',
              '一种待验证的解释是：真实训练集只对坑、岩石、天空提供正类监督，未标注月壤被正确忽略，却缺少将这些区域约束为月壤的监督。'
              '适配器改变特征后仍可能压低月壤输出；冻结原分类器并不保证类别分布不变。需要补充可靠背景标签或源域保持约束的对照实验来验证，不能把未知像素直接全改成月壤。',
              '当前模型属于有监督真实域适配，不能和源域训练模型一起作为纯零样本结果。','',
              '## 当前模型的坑像素去向','',
              *[f'- {name}：{value:.4f}%' for name,value in zip(ZH,percentages[0])],'',
              '## 评价范围','',limitations,
              '两模型均在本次以相同 CPU、FP32、1024、无增强进行预测。每模型只有一个训练结果；只作描述性比较。',
              '模型仍输出五类，但这里没有月壤和山体的完整真值，不能评价它们的召回率。',
              '标注映射为 Craters→1、Rocks→2、Sky→4；同类多边形取并集，跨类重叠及未标注像素为255。',
              f"累计训练从第 9 轮 EMA 检查点恢复，保留优化器；不声称与不中断训练逐位相同。",'',
              '## 输出','',
              '`index.html` 为浏览页；`comparison/` 为原图、真值、仿真 IBN、完整微调模型四联图。',
              '`finetuned/pred_masks/` 为完整模型类别索引掩膜；`truth_masks/` 为评价使用的真值副本。',
              '`per_image.csv`、`metrics_annotated_pixels.json` 保存逐图和汇总计数；`evaluation_verification.json` 保存核验。',
              '`five_class_metrics.csv` 保存两模型五类指标（小数比例）；无真值类别的指标留空，status 标明原因。']
    (out/'analysis-report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    (out/'stats-appendix.md').write_text(
        '# 统计说明\n\n每个模型一个训练结果，46 张相同测试图像。主指标按类别汇总所有有效标注像素；'
        '效应量为召回差值（百分点）。没有独立场景分组或重复训练；像素非独立。'
        '未计算跨训练的标准差、置信区间或显著性检验，也不作显著性结论。没有多重检验。'
        '缺少某类标注的单图召回留空，不计为零。\n\n'+metric_note+'\n'
        'IoU=TP/(TP+FP+FN)，precision=TP/(TP+FP)，recall=TP/(TP+FN)。'
        'JSON/CSV 中的五类指标使用 0～1 比例；无真值类别为 null/空值，不纳入三类 mIoU。\n',encoding='utf-8')
    (out/'figure-catalog.md').write_text(
        '# 图表解释\n\n'+caution+'\n\n'
        f'- annotated_recall.png/pdf：比较同一子集的三类召回。坑变化 {after[0]-before[0]:+.4f}、岩石变化 {after[1]-before[1]:+.4f}、天空变化 {after[2]-before[2]:+.4f} 个百分点。'
        '图中无误差条；观察方向和量级，判断本子集的漏分是否缓解，不能推断统计显著或模块因果。\n'
        f'- prediction_directions.png/pdf：每行是一个真值类别内的五类预测比例。坑→月壤 {percentages[0,0]:.4f}%，坑→岩石 {percentages[0,2]:.4f}%。'
        '用于定位剩余混淆；不能据此评价未知区域的误检。\n'
        '- comparison/：四列依次为原图、已标注真值、仿真 IBN、完整微调模型。'
        '统一调色板，未标注真值保留原图；检查空间形态和可疑的大面积预测，图像不能替代完整负例标注。\n',encoding='utf-8')
    cards=[]
    for row in rows:
        stem=Path(row['image']).stem
        recall=row['finetuned_Craters_recall']
        detail='无有效坑标注' if recall=='' else f'坑标注像素召回 {recall*100:.2f}%'
        cards.append(f'<article data-name="{html.escape(row["image"],quote=True)}"><h3>{html.escape(row["image"])}</h3><p>{detail}</p>'
                     f'<a href="comparison/{stem}.jpg"><img loading="lazy" src="comparison/{stem}.jpg"></a>'
                     f'<p><a href="finetuned/pred_masks/{stem}.png">完整模型掩膜</a> · <a href="truth_masks/{stem}.png">评价真值</a></p></article>')
    legend=''.join(f'<span style="margin-right:18px"><i style="background:rgb({int(c[0])},{int(c[1])},{int(c[2])})"></i>{n}</span>' for n,c in zip(ZH,colors))
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>完整 IBN＋KAN＋边缘分支 · 真实测试</title><style>
body{{margin:0;background:#f3f5f8;color:#203247;font:16px/1.6 'Microsoft YaHei',sans-serif}}main{{max-width:1440px;margin:auto;padding:28px}}
article,.note{{background:white;padding:18px;border-radius:12px;margin:22px 0}}article img{{width:100%}}a{{color:#186275}}
table{{border-collapse:collapse;background:white}}td,th{{padding:12px 18px;border-bottom:1px solid #ddd;text-align:left}}.table-wrap{{overflow-x:auto}}
.figs{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.figs img{{width:100%}}i{{display:inline-block;width:16px;height:16px;margin-right:6px}}
input{{padding:12px;width:min(80%,500px)}}@media(max-width:700px){{.figs{{grid-template-columns:1fr}}main{{padding:12px}}td,th{{padding:6px}}}}
</style><main><h1>完整 IBN＋KAN＋边缘分支 · {payload['images']} 张真实测试</h1>
<p>150 张真实图像微调 · 第 30 轮完整 last.pt · 输入 1024 · 无测试时训练</p>
<div class="note"><strong>解读重点：</strong>{caution}</div>
<div class="note">{limitations}</div><h2>完整模型五类指标 · 仅已标注区域</h2>
<div class="table-wrap"><table id="five-class-metrics"><tr><th>类别</th><th>有效标注像素</th><th>IoU</th><th>精确率</th><th>召回率</th></tr>{''.join(five_html)}</table></div>
<p><strong>{metric_note}</strong></p><p><a href="five_class_metrics.csv">下载五类指标</a></p>
<h2>已标注区域像素召回率对照</h2>
<table><tr><th>类别</th><th>仿真 IBN</th><th>完整微调模型</th><th>差值 / 百分点</th></tr>{''.join(html_rows)}</table>
<p>差异同时包含新增模块和真实监督训练的影响，不能单独归因于某个模块。</p>
<p><a href="analysis-report.md">详细报告</a> · <a href="metrics_annotated_pixels.json">完整指标</a> · <a href="per_image.csv">逐图统计</a></p>
<div class="figs"><img src="figures/annotated_recall.png"><img src="figures/prediction_directions.png"></div>
<h2>逐图检查</h2><p>从左到右：原图 → 已标注真值 → 仿真 IBN → 完整微调模型</p><p>{legend}</p>
<input id="filter" aria-label="文件名筛选" placeholder="输入文件名筛选">{''.join(cards)}
<script>document.getElementById('filter').addEventListener('input',function(){{const q=this.value.toLowerCase();document.querySelectorAll('article').forEach(a=>a.hidden=!a.dataset.name.toLowerCase().includes(q));}});</script></main></html>'''
    (out/'index.html').write_text(page,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path(r'D:\Projects\Sim2real\data\real_data\kaggle_dataset\test\test'))
    parser.add_argument('--weights',type=Path,default=ROOT/'runs/ibn_kan_edge_real_train_1024_resume/weights/last.pt')
    parser.add_argument('--out',type=Path,default=ROOT/'kaggle_full_model_test_20261112/output')
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    annotation=args.source/'annotations.json'
    data=json.loads(annotation.read_text(encoding='utf-8'))
    assert {x['id']:x['name'] for x in data['categories']}=={k:v[0] for k,v in CLASSES.items()}
    ids={r['id'] for r in data['images']}
    assert len(ids)==len(data['images'])==46
    grouped=defaultdict(list)
    for item in data['annotations']:
        assert item['image_id'] in ids
        grouped[item['image_id']].append(item)
    paths=[args.source/'images'/Path(row['file_name']).name for row in data['images']]
    assert len({p.stem for p in paths})==len(paths)
    assert set(paths)=={p for p in (args.source/'images').iterdir() if p.is_file()}
    source_hashes={str(p):sha256(p) for p in paths}
# 【整理前代码保留】     provenance_path=ROOT/'real_adapter_train_data/provenance.json'
# 【整理修改 CODE-PACKAGE-02】使用本副本中的模块与资源。
    provenance_path=ROOT/'metadata/real_train_provenance.json'
    provenance=json.loads(provenance_path.read_text(encoding='utf-8'))
    train_hashes={sha256(Path(r['source'])) for r in provenance['records']}
    assert train_hashes=={r['sha256'] for r in provenance['records']}
    matching=[str(p) for p in paths if source_hashes[str(p)] in train_hashes]
    assert not matching, '测试图像与训练图像存在文件哈希重合，需要先检查划分'
    old_hashes=set(json.loads(OLD_HASHES.read_text(encoding='utf-8')).values())
    protected={str(p):sha256(p) for p in [annotation,args.weights,BASE_IBN,provenance_path,Path(provenance['source'])/'annotations.json']}
    audit=json.loads((args.weights.parent.parent/'real_adapter_audit.json').read_text(encoding='utf-8'))
    assert audit['checkpoint_sha256']==protected[str(args.weights)] and audit['total_completed_epochs']==30
    args.out.mkdir(parents=True,exist_ok=False)
    for folder in ['truth_masks','truth_overlay','comparison','ibn_baseline/pred_masks','ibn_baseline/pred_overlay','finetuned/pred_masks','finetuned/pred_overlay']:
        (args.out/folder).mkdir(parents=True)
    palette=Image.open(PALETTE).getpalette()
    assert palette is not None and len(palette)==768
    colors=np.array(palette[:15],dtype=np.uint8).reshape(5,3)
    palette[255*3:255*3+3]=[100,100,100]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    models={'ibn_baseline':YOLO(str(BASE_IBN),task='semantic'),'finetuned':YOLO(str(args.weights),task='semantic')}
    assert_ibn(models['ibn_baseline'].model)
    assert_combo(models['finetuned'].model)
    assert models['finetuned'].ckpt['epoch']==29
    for model in models.values():
        assert [model.names[i] for i in range(5)]==NAMES
        model.model.eval().requires_grad_(False)
    matrices={key:np.zeros((3,5),dtype=np.int64) for key in models}
    rows=[]
    started=time.perf_counter()
    for i,(record,path) in enumerate(zip(data['images'],paths),1):
        original=Image.open(path).convert('RGB')
        w,h=original.size
        assert (w,h)==(record['width'],record['height'])
        truth,ambiguous,clipped=rasterize(grouped[record['id']],w,h)
        row={'image':path.name,'image_id':record['id'],'annotated_pixels':int((truth!=255).sum()),'ambiguous_pixels':ambiguous,'clipped_polygons':clipped}
        palette_mask(truth,palette,args.out/'truth_masks'/(path.stem+'.png'))
        truth_view=overlay(original,truth,colors)
        truth_view.save(args.out/'truth_overlay'/(path.stem+'.jpg'),quality=92)
        panels=[('Image',original),('Annotated truth; others unknown',truth_view)]
        for key,model in models.items():
            result=model.predict(str(path),imgsz=1024,device=args.device,half=False,augment=False,save=False,verbose=False)[0].cpu()
            pred=result.semantic_mask.data.numpy().astype(np.uint8)
            assert pred.shape==(h,w) and np.isin(pred,np.arange(5)).all()
            for category,(name,class_id,_) in CLASSES.items():
                hist=np.bincount(pred[truth==class_id],minlength=5)
                matrices[key][category]+=hist
                row[name+'_truth_pixels']=int(hist.sum())
                row[key+'_'+name+'_correct_pixels']=int(hist[class_id])
                row[key+'_'+name+'_recall']=float(hist[class_id]/hist.sum()) if hist.sum() else ''
                for label,count in zip(NAMES,hist):
                    row[key+'_'+name+'_to_'+label]=int(count)
            row[key+'_all_image_pixels_by_class']=json.dumps(np.bincount(pred.ravel(),minlength=5).tolist())
            palette_mask(pred,palette,args.out/key/'pred_masks'/(path.stem+'.png'))
            pred_view=overlay(original,pred,colors)
            pred_view.save(args.out/key/'pred_overlay'/(path.stem+'.jpg'),quality=92)
            panels.append((LABELS[key],pred_view))
        card=Image.new('RGB',(2400,640),'#182634')
        draw=ImageDraw.Draw(card)
        for j,(title,panel) in enumerate(panels):
            panel.thumbnail((600,600),Image.Resampling.LANCZOS)
            card.paste(panel,(600*j,40))
            draw.text((600*j+12,9),title,fill='white',font=font)
        card.save(args.out/'comparison'/(path.stem+'.jpg'),quality=93)
        rows.append(row)
        if i==1 or i%10==0 or i==len(paths):
            print(f'{i}/{len(paths)} completed; elapsed {time.perf_counter()-started:.1f}s',flush=True)
    with (args.out/'per_image.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload={'created_at':datetime.now().isoformat(),'source':str(args.source),'images':len(paths),'annotations':len(data['annotations']),
             'annotation_sha256':protected[str(annotation)],'imgsz':1024,'device':args.device,'precision':'FP32',
             'training_performed_during_evaluation':False,'training_domain':'real_supervised','selection':'fixed_last_epoch_30_ema',
             'ignore_id':255,'category_mapping':{v[0]:v[1] for v in CLASSES.values()},
             'metric':'micro-aggregated per-class annotated-pixel recall',
             'valid_annotated_pixels':sum(r['annotated_pixels'] for r in rows),'ambiguous_pixels':sum(r['ambiguous_pixels'] for r in rows),
             'train_test_overlap':{'matching_images':len(matching),'training_images':provenance['images'],'method':'file SHA-256'},
             'previous_200_overlap':{'matching_images':sum(v in old_hashes for v in source_hashes.values())},
             'inference_and_artifact_seconds':time.perf_counter()-started,'models':{}}
    for key,weight in [('ibn_baseline',BASE_IBN),('finetuned',args.weights)]:
        payload['models'][key]={'weights':str(weight),'sha256':protected[str(weight)],'per_class':class_metrics(matrices[key],rows,key),
                               'matrix_rows_truth_columns_prediction':matrices[key].tolist()}
    assert np.array_equal(matrices['ibn_baseline'].sum(axis=1),matrices['finetuned'].sum(axis=1))
    assert matrices['finetuned'].sum()==payload['valid_annotated_pixels']
    assert all(sha256(Path(p))==v for p,v in {**protected,**source_hashes}.items())
    payload['source_images_annotations_weights_unchanged']=True
    add_area_diagnostics(payload,rows)
    add_five_class_metrics(payload)
    dump(args.out/'source_hashes.json',source_hashes)
    dump(args.out/'metrics_annotated_pixels.json',payload)
    make_report(args.out,payload,rows,colors)
    print(json.dumps(payload,ensure_ascii=False,indent=2),flush=True)
    print('DONE '+str(args.out),flush=True)


if __name__=='__main__':
    main()
