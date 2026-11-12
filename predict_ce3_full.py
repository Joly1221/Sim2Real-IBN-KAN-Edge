"""【新增 IBN-CE3-01】完整 IBN+KAN+edge 权重预测 CE3，保留所有旧程序。"""
from __future__ import annotations

import csv
from datetime import datetime
import html
import json
from pathlib import Path
import re

from ibn_kan_edge import YOLO, torch, assert_combo
# 【整理前代码保留】 from runtime import ROOT, SIM, NAMES
# 【整理修改 CODE-PACKAGE-02】使用本副本中的模块与资源。
from runtime import ROOT, SIM, NAMES, PALETTE
from evaluate_kaggle_test import sha256, dump, palette_mask, overlay, np, Image, ImageDraw, ImageFont

SOURCE=Path(r'D:\Projects\Sim2real\data\real_data\CE3')
WEIGHTS=ROOT/'runs/ibn_kan_edge_real_train_1024_resume/weights/last.pt'
OUT=ROOT/'ce3_full_model_20261112/output'
ZH=['月壤','撞击坑','岩石','山体','天空']


def main():
    torch.set_num_threads(1)
    images=sorted(p for p in SOURCE.rglob('*') if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'})
    assert len(images)==2 and len({p.stem for p in images})==2
    protected={str(p):sha256(p) for p in [WEIGHTS,*images]}
    audit=json.loads((WEIGHTS.parent.parent/'real_adapter_audit.json').read_text(encoding='utf-8'))
    assert protected[str(WEIGHTS)]==audit['checkpoint_sha256'] and audit['total_completed_epochs']==30
    model=YOLO(str(WEIGHTS),task='semantic')
    assert_combo(model.model)
    assert model.ckpt['epoch']==29
    assert [model.names[i] for i in range(5)]==NAMES
    model.model.eval().requires_grad_(False)
    OUT.mkdir(parents=True,exist_ok=False)
    for folder in ['original','pred_masks','pred_overlay','comparison']:
        (OUT/folder).mkdir()
    palette=Image.open(PALETTE).getpalette()
    assert palette is not None and len(palette)==768
    colors=np.array(palette[:15],dtype=np.uint8).reshape(5,3)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    records=[]
    sheets=[]
    for i,path in enumerate(images,1):
        with Image.open(path) as im:
            im.verify()
        original=Image.open(path).convert('RGB')
        result=model.predict(str(path),imgsz=1024,device='cpu',augment=False,save=False,verbose=False)[0].cpu()
        prediction=result.semantic_mask.data.numpy().astype(np.uint8)
        assert prediction.shape==(original.height,original.width) and np.isin(prediction,np.arange(5)).all()
        filename=path.stem+'.png'
        mask_path=OUT/'pred_masks'/filename
        palette_mask(prediction,palette,mask_path)
        assert np.array_equal(prediction,np.asarray(Image.open(mask_path)))
        original.save(OUT/'original'/filename)
        view=overlay(original,prediction,colors)
        view.save(OUT/'pred_overlay'/filename)
        sheet=Image.new('RGB',(1800,760),'#182634')
        draw=ImageDraw.Draw(sheet)
        for j,(title,panel) in enumerate([('CE3 原图',original),('完整 IBN＋KAN＋边缘分支：预测',view)]):
            scaled=panel.copy()
            scaled.thumbnail((900,650),Image.Resampling.LANCZOS)
            sheet.paste(scaled,(j*900+(900-scaled.width)//2,48+(650-scaled.height)//2))
            draw.text((j*900+14,10),title,font=font,fill='white')
        for j,(name,color) in enumerate(zip(ZH,colors)):
            x=14+j*135
            draw.rectangle((x,716,x+20,736),fill=tuple(map(int,color)))
            draw.text((x+28,710),name,font=font,fill='white')
        draw.text((900,710),'没有真值标注，无法计算精度',font=font,fill='white')
        sheet.save(OUT/'comparison'/filename)
        sheets.append(sheet)
        counts=np.bincount(prediction.ravel(),minlength=5)
        records.append({'image':path.name,'source_sha256':protected[str(path)],'width':original.width,'height':original.height,
                        'pixels':dict(zip(NAMES,map(int,counts))),'fractions':dict(zip(NAMES,map(float,counts/prediction.size)))})
        print(f'{i}/{len(images)} completed: {path.name}',flush=True)
    overview=Image.new('RGB',(1800,760*len(sheets)),'#182634')
    for i,sheet in enumerate(sheets):
        overview.paste(sheet,(0,760*i))
    overview.save(OUT/'overview.jpg',quality=94)
    assert all(sha256(Path(p))==value for p,value in protected.items())
    manifest={'completed_at':datetime.now().astimezone().isoformat(),'state':'complete','source':str(SOURCE),
              'weights':str(WEIGHTS),'weights_sha256':protected[str(WEIGHTS)],'complete_model_components':['YOLO26n-sem','IBN','single KAN adapter','edge branch'],
              'checkpoint_epoch':30,'checkpoint_selection':'last_epoch_ema','model_training_domain':'real_supervised',
              'imgsz':1024,'device':'cpu','precision':'FP32','cpu_threads':1,'class_names':NAMES,'palette_rgb':colors.tolist(),
              'training_performed':False,'ground_truth_metrics':None,'source_images_weights_unchanged':True,'images':records}
    dump(OUT/'manifest.json',manifest)
    with (OUT/'prediction_fractions.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['image',*NAMES])
        writer.writeheader()
        for record in records:
            writer.writerow({'image':record['image'],**record['fractions']})
    table_rows=''.join('<tr><td>'+name+'</td>'+''.join(f'<td>{record["fractions"][key]*100:.2f}%</td>' for record in records)+'</tr>' for name,key in zip(ZH,NAMES))
    legend=''.join(f'<span><i style="background:rgb({int(c[0])},{int(c[1])},{int(c[2])})"></i>{name}</span>' for name,c in zip(ZH,colors))
    cards=''.join(f'<article><h2>图 {i} · {html.escape(record["image"])}</h2><a href="comparison/{Path(record["image"]).stem}.png"><img src="comparison/{Path(record["image"]).stem}.png"></a>'
                  f'<p><a href="original/{Path(record["image"]).stem}.png">原图</a> · <a href="pred_overlay/{Path(record["image"]).stem}.png">预测叠加图</a> · '
                  f'<a href="pred_masks/{Path(record["image"]).stem}.png">类别索引掩膜</a></p></article>' for i,record in enumerate(records,1))
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CE3 · 完整 IBN＋KAN＋边缘分支预测</title><style>body{{background:#f2f5f8;color:#203247;font:16px/1.7 'Microsoft YaHei',sans-serif;margin:0}}main{{max-width:1440px;margin:auto;padding:24px}}article,.note{{background:white;padding:18px;margin:20px 0;border-radius:12px}}article img{{width:100%}}article h2{{font-size:18px;overflow-wrap:anywhere}}a{{color:#17647e}}table{{border-collapse:collapse;background:white}}td,th{{padding:10px 22px;border-bottom:1px solid #ddd}}span{{margin-right:20px}}i{{display:inline-block;width:16px;height:16px;margin-right:7px}}</style>
<main><h1>CE3 · 完整 IBN＋KAN＋边缘分支预测</h1><p>2 张图 · 第 30 轮完整 last.pt · 输入 1024 · CPU FP32</p>
<div class="note">目录没有真值标注，五类 IoU、精确率、召回率及 mIoU 均无法计算。下面仅显示预测像素占比，不是精度。</div>
<p>{legend}</p><h2>五类预测像素占比（不是精度）</h2><table><tr><th>类别</th><th>图 1</th><th>图 2</th></tr>{table_rows}</table>
<p><a href="README.md">预测说明</a> · <a href="manifest.json">权重与配置</a> · <a href="prediction_fractions.csv">预测占比 CSV</a> · <a href="overview.jpg">两图总览</a></p>{cards}</main></html>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')
    note=f'# CE3 完整模型预测\n\n权重：`{WEIGHTS}`\n\n完整模型包括 YOLO26n-sem、IBN、单层 KAN 适配器与边缘分支。真实 train 微调至累计第30轮。\n\n2张图，1024输入，CPU FP32，只预测、不训练。目录没有真值标注，不能计算精度或mIoU。预测占比不是精度。\n\n`index.html` 为浏览页；`comparison` 为对照图；`pred_masks` 为0～4类索引PNG，依次为月壤、撞击坑、岩石、山体、天空。调色板与此前IBN测试一致。源图和权重哈希已核对未变。\n'
    (OUT/'README.md').write_text(note,encoding='utf-8')
    refs=re.findall(r'(?:href|src)="([^"]+)"',page)
    assert all((OUT/ref).is_file() for ref in refs)
    dump(OUT/'verification.json',{'complete_model_checked':True,'masks_same_size_as_original':True,'masks_saved_and_reloaded_equal':True,
                                'class_ids_valid':True,'links_checked':len(refs),'all_links_exist':True,'source_and_weights_unchanged':True})
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
