"""【新增 IBN-EVAL-01】真实子集评价：固定模型，未标注像素忽略，不训练。

沿用此前 Apollo 评价的多边形栅格化、类别映射和已标注像素召回率口径。
同时评价原始仿真模型，保存预测掩膜、对照图、逐图统计及报告。
原训练/预测程序和真值文件均不修改。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import time

# 【整理前代码保留】 from runtime import ROOT, BASE, NAMES, SIM, YOLO, torch
# 【整理修改 CODE-PACKAGE-02】使用本副本中的模块与资源。
from runtime import ROOT, BASE, NAMES, SIM, YOLO, torch, PALETTE
from ibn_backbone import assert_ibn
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = {0: ('Craters', 1, '撞击坑'), 1: ('Rocks', 2, '岩石'), 2: ('Sky', 4, '天空')}
MODEL_LABELS = ['Regolith', 'Crater', 'Rock', 'Mountain', 'Sky']
# 【整理前代码保留】 OLD_HASHES = Path(r'D:\Projects\[Seg]\ultralytics-main\sim2real_conv_adapter\apollo_test_20261110\output\source_hashes.json')
# 【整理修改 CODE-PACKAGE-02】使用本副本中的模块与资源。
OLD_HASHES = ROOT / 'metadata/apollo_200_source_hashes.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def rasterize(annotations, width, height):
    """【新增】按类别求多边形并集，跨类别重叠与未标注区域赋 255。"""
    planes = np.zeros((3, height, width), dtype=np.uint8)
    clipped = 0
    for annotation in annotations:
        category = annotation['category_id']
        assert category in CLASSES
        assert not annotation.get('iscrowd', 0) and not annotation.get('ignore', 0)
        assert isinstance(annotation['segmentation'], list) and annotation['segmentation']
        for polygon in annotation['segmentation']:
            points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
            assert len(points) >= 3 and np.isfinite(points).all()
            if ((points[:, 0] < 0) | (points[:, 0] > width - 1) |
                    (points[:, 1] < 0) | (points[:, 1] > height - 1)).any():
                clipped += 1
            points[:, 0] = points[:, 0].clip(0, width - 1)
            points[:, 1] = points[:, 1].clip(0, height - 1)
            cv2.fillPoly(planes[category], [np.rint(points).astype(np.int32)], 1)
    count = planes.sum(axis=0)
    truth = np.full((height, width), 255, dtype=np.uint8)
    for category, (_, model_id, _) in CLASSES.items():
        truth[(planes[category] > 0) & (count == 1)] = model_id
    return truth, int((count > 1).sum()), clipped


def palette_mask(mask, palette, path):
    output = Image.fromarray(mask).convert('P')
    output.putpalette(palette)
    output.save(path)


def overlay(original, mask, colors):
    pixels = np.asarray(original).astype(np.float32).copy()
    valid = mask != 255
    pixels[valid] = .6 * pixels[valid] + .4 * colors[mask[valid]]
    return Image.fromarray(pixels.clip(0, 255).astype(np.uint8))


def class_metrics(matrix, records, model_key):
    result = []
    for category, (name, class_id, chinese) in CLASSES.items():
        row = matrix[category]
        support = int(row.sum())
        result.append({'class': name, 'class_zh': chinese, 'model_class_id': class_id,
                       'annotated_pixels': support, 'correct_pixels': int(row[class_id]),
                       'annotated_pixel_recall': float(row[class_id] / support) if support else None,
                       'images_with_annotations': sum(r[name + '_truth_pixels'] > 0 for r in records),
                       'prediction_pixels_by_class': dict(zip(NAMES, map(int, row)))})
    return result


def reports(out, payload, records, colors):
    """【新增】只做本子集上的描述性比较，避免用像素伪造独立样本。"""
    figure_dir = out / 'figures'
    figure_dir.mkdir(exist_ok=True)  # 【新增】允许仅从保存的统计结果重新生成报告。
    before = np.array([c['annotated_pixel_recall'] for c in payload['models']['baseline']['per_class']]) * 100
    after = np.array([c['annotated_pixel_recall'] for c in payload['models']['ibn']['per_class']]) * 100
    figure, ax = plt.subplots(figsize=(8, 4.6), layout='constrained')
    x = np.arange(3)
    for offset, values, label, color in [(-.18, before, 'Original simulation model', '#8496ad'),
                                         (.18, after, 'Simulation-trained IBN', '#ce792f')]:
        bars = ax.bar(x + offset, values, .35, label=label, color=color)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=10)
    ax.set_xticks(x, ['Crater', 'Rock', 'Sky'])
    ax.set_ylabel('Recall on annotated pixels (%)')
    ax.set_ylim(0, 106)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title(f"Same {payload['images']} images; unannotated pixels excluded")
    figure.savefig(figure_dir / 'annotated_recall.png', dpi=170)
    figure.savefig(figure_dir / 'annotated_recall.pdf')
    plt.close(figure)
    matrix = np.asarray(payload['models']['ibn']['matrix_rows_truth_columns_prediction'])
    percentages = matrix / matrix.sum(axis=1, keepdims=True) * 100
    figure, ax = plt.subplots(figsize=(8, 3.8), layout='constrained')
    heat = ax.imshow(percentages, vmin=0, vmax=100, cmap='Blues', aspect='auto')
    ax.set_xticks(range(5), MODEL_LABELS)
    ax.set_yticks(range(3), ['Crater', 'Rock', 'Sky'])
    ax.set_xlabel('IBN predicted class')
    ax.set_ylabel('Annotated class')
    ax.set_title('Prediction direction within annotated regions (%)')
    for y in range(3):
        for x in range(5):
            ax.text(x, y, f'{percentages[y, x]:.2f}', ha='center', va='center',
                    color='white' if percentages[y, x] > 55 else '#162838')
    figure.colorbar(heat, ax=ax, label='Row percentage')
    figure.savefig(figure_dir / 'prediction_directions.png', dpi=170)
    figure.savefig(figure_dir / 'prediction_directions.pdf')
    plt.close(figure)
    table = ['| 已标注类别 | 原始仿真模型 | IBN | 差值（百分点） | 标注像素数 |',
             '|---|---:|---:|---:|---:|']
    for i, row in enumerate(payload['models']['ibn']['per_class']):
        table.append(f"| {row['class_zh']} | {before[i]:.4f}% | {after[i]:.4f}% | {after[i]-before[i]:+.4f} | {row['annotated_pixels']:,} |")
    crater = percentages[0]
    report = [
        '# IBN 真实图像测试', '',
        f"数据：`{payload['source']}`，{payload['images']} 张图、{payload['annotations']} 个多边形。",
        '问题：仿真训练后的 IBN 模型在本子集上的已标注像素召回率如何，与原始仿真模型相比有何变化？',
        '两个模型都固定参数、输入尺寸 1024；本次没有训练或调整阈值。', '',
        '## 已标注区域的像素召回率', '', *table, '',
        f'本子集的撞击坑召回率由 {before[0]:.4f}% 变为 {after[0]:.4f}%，变化 {after[0]-before[0]:+.4f} 个百分点；'
        f'岩石变化 {after[1]-before[1]:+.4f} 个百分点，天空变化 {after[2]-before[2]:+.4f} 个百分点。',
        f'IBN 将 {crater[0]:.4f}% 的已标注坑像素预测为月壤。当前结果没有显示坑识别问题得到解决。', '',
        '每类召回率 = 该类已标注区域中预测正确的像素数 / 该类有效标注像素数。',
        '未标注区域视为未知，不自动视为月壤。跨类别重叠像素排除；同类别重叠取并集。',
        '**这些指标不是全图 IoU、精确率、总体 mIoU，也不是撞击坑实例召回率。**', '',
        '## IBN 的坑像素误分方向', '',
        *[f'- {name}：{value:.4f}%' for name, value in zip(['月壤','撞击坑','岩石','山体','天空'], crater)], '',
        '## 数据与比较边界', '',
        f"本次 {payload['previous_200_overlap']['matching_images']}/{payload['images']} 张图像的 SHA-256 与此前评价过的 200 张 Apollo 图像相同；这不是全新的独立盲测集。",
        '这是其中一个子集。本表重新计算了原始模型在同一子集的结果，不能直接用此前 200 张总体百分比作差。',
        '每个模型只有一个训练结果，图像可能来自关联场景；这里只报告描述性结果，不作统计显著性或跨场景推广判断。',
        'IBN 同时经历了全模型继续训练，原始模型没有这段继续训练；二者差异不能全部归因于 IBN 结构。',
        f"有效标注像素：{payload['valid_annotated_pixels']:,}；跨类别重叠排除：{payload['ambiguous_pixels']:,}。", '',
        '## 文件', '',
        '- `index.html`：交互筛选的图像对照页。',
        '- `comparison/`：原图、真值、原模型、IBN 四联图。',
        '- `ibn/pred_masks/` 与 `baseline/pred_masks/`：0～4 索引预测掩膜。',
        '- `truth_masks/`：映射到五类体系后的标注掩膜，255 为未知或重叠。原始标注文件未修改。',
        '- `per_image.csv`：逐图类别召回及误分计数。',
        '- `metrics_annotated_pixels.json`：汇总指标、权重哈希、评价设置。',
        '- `figures/`：类别召回图与误分方向图。', '',
        f"IBN 权重：`{payload['models']['ibn']['weights']}`。",
        f"原始权重：`{payload['models']['baseline']['weights']}`。",
    ]
    (out / 'analysis-report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    (out / 'stats-appendix.md').write_text(
        '# 统计说明\n\n两个模型各一次训练结果。主指标为逐类别汇总的已标注像素召回率，'
        '效应量为同一子集上的召回率差值（百分点）。各像素不独立；缺少独立场景分组和重复训练。'
        '不把像素或相邻图像当作独立实验重复，因此未计算均值±标准差、置信区间或显著性检验，'
        '也不提出显著性结论。没有执行多重假设检验，不涉及多重比较校正。'
        '逐图指标保存在 per_image.csv，缺少该类标注的图像留空而非计为 0。\n', encoding='utf-8')
    (out / 'figure-catalog.md').write_text(
        '# 图表说明\n\n'
        f'- `annotated_recall.png/pdf`：坑召回变化 {after[0]-before[0]:+.4f}、岩石变化 {after[1]-before[1]:+.4f}、天空变化 {after[2]-before[2]:+.4f} 个百分点。'
        '对比同一子集的原模型与 IBN 类别召回。数字为像素汇总百分比，'
        '没有误差条。观察变化方向与量级，以判断本子集是否出现改善；不能据此判定统计显著或结构因果。\n'
        f'- `prediction_directions.png/pdf`：IBN 将 {crater[0]:.4f}% 的坑标注像素预测成月壤，漏分问题仍在。'
        '图示 IBN 在各类标注内的五类预测比例。每行合计 100%，'
        '观察撞击坑被分为月壤和岩石的比例，以定位剩余误分问题。不能反推未标注区域的假阳性率。\n'
        '- `comparison/`：逐图定性检查，四列依次为原图、已标注真值、原模型、IBN。'
        '真值中未标注区域保留原图，不代表月壤。三种掩膜统一用仿真五类调色板。\n', encoding='utf-8')
    rows_html = ''.join(f"<tr><td>{r['class_zh']}</td><td>{before[i]:.2f}%</td><td>{after[i]:.2f}%</td><td>{after[i]-before[i]:+.2f}</td></tr>"
                        for i, r in enumerate(payload['models']['ibn']['per_class']))
    cards = []
    for row in records:
        name = row['image']
        stem = Path(name).stem
        recall = row['ibn_Craters_recall']
        description = '无有效坑标注' if recall == '' else f'坑像素召回 {recall*100:.2f}%'
        cards.append(f'<article data-name="{html.escape(name, quote=True)}"><h3>{html.escape(name)}</h3><p>{description}</p>'
                     f'<a href="comparison/{stem}.jpg"><img loading="lazy" src="comparison/{stem}.jpg" alt="原图、真值、原模型和 IBN 对照"></a>'
                     f'<p><a href="ibn/pred_masks/{stem}.png">IBN 掩膜</a> · <a href="truth_masks/{stem}.png">评价用真值</a></p></article>')
    legend = ''.join(f'<span><i style="background:rgb({int(color[0])},{int(color[1])},{int(color[2])})"></i>{name}</span>'
                     for name, color in zip(['月壤','撞击坑','岩石','山体','天空'], colors))
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IBN 真实图像测试 · {payload['images']} 张</title><style>
body{{margin:0;background:#f3f5f8;color:#203247;font:16px/1.65 'Microsoft YaHei',sans-serif}}main{{max-width:1440px;margin:auto;padding:32px}}
h1{{margin:0;font-size:30px}}h2{{margin:26px 0 12px}}p{{margin:10px 0}}a{{color:#1c6094}}.note{{background:#fff3db;padding:16px;border-radius:10px}}
table{{border-collapse:collapse;background:white;width:100%;max-width:780px}}th,td{{padding:12px 18px;border-bottom:1px solid #dce3eb;text-align:left}}
.figs{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:24px}}.figs img{{width:100%;background:white}}
article{{background:white;padding:18px;border-radius:12px;margin:24px 0}}article img{{width:100%;height:auto}}article h3{{font-size:17px;overflow-wrap:anywhere;margin:0}}
input{{padding:12px;width:min(90%,500px);border:1px solid #a6b8c8;border-radius:8px}}.legend span{{display:inline-flex;align-items:center;margin-right:18px}}i{{display:inline-block;width:17px;height:17px;margin-right:7px}}
@media(max-width:700px){{main{{padding:16px}}.figs{{grid-template-columns:1fr}}th,td{{padding:8px}}}}
</style><main><h1>IBN 真实图像测试 · {payload['images']} 张</h1><p>仿真训练后的最佳权重 · 1024 输入 · 固定参数预测</p>
<div class="note">指标只统计已标注区域的像素召回率，未标注区域为未知。{payload['previous_200_overlap']['matching_images']} 张图与此前的 200 张评价数据重合，因此不是新的独立盲测。</div>
<h2>同一批图像上的结果</h2><table><tr><th>类别</th><th>原模型</th><th>IBN</th><th>差值 / 百分点</th></tr>{rows_html}</table>
<p>两者的差异包含全模型继续训练的影响，不能全部归因于 IBN 结构。</p>
<p><a href="analysis-report.md">详细报告</a> · <a href="metrics_annotated_pixels.json">完整指标</a> · <a href="per_image.csv">逐图统计</a></p>
<div class="figs"><img src="figures/annotated_recall.png" alt="类别召回率对比"><img src="figures/prediction_directions.png" alt="IBN 误分方向"></div>
<h2>逐图检查</h2><p>从左至右：原图 → 已标注真值 → 原始仿真模型 → IBN。点击图片查看大图。</p><p class="legend">{legend}</p>
<input id="filter" aria-label="按文件名筛选" placeholder="输入文件名筛选">{''.join(cards)}
<script>document.getElementById('filter').addEventListener('input',function(){{const q=this.value.toLowerCase();document.querySelectorAll('article').forEach(a=>a.hidden=!a.dataset.name.toLowerCase().includes(q));}});</script></main></html>'''
    (out / 'index.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(r'D:\Projects\Sim2real\data\real_data\kaggle_dataset\test\test'))
    parser.add_argument('--weights', type=Path, default=ROOT / 'runs/ibn_sim_1024/weights/best.pt')
    parser.add_argument('--out', type=Path, default=ROOT / 'kaggle_test_20261112/output')
    parser.add_argument('--device', default='0')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    annotation_path = args.source / 'annotations.json'
    data = json.loads(annotation_path.read_text(encoding='utf-8'))
    assert {x['id']: x['name'] for x in data['categories']} == {k: v[0] for k, v in CLASSES.items()}
    image_ids = {x['id'] for x in data['images']}
    assert len(image_ids) == len(data['images'])
    grouped = defaultdict(list)
    for item in data['annotations']:
        assert item['image_id'] in image_ids
        grouped[item['image_id']].append(item)
    paths = [args.source / 'images' / Path(rec['file_name']).name for rec in data['images']]
    assert len({p.stem for p in paths}) == len(paths) and all(p.is_file() for p in paths)
    assert set(paths) == {p for p in (args.source / 'images').iterdir() if p.is_file()}
    protected = {str(p): sha256(p) for p in [annotation_path, args.weights, BASE]}
    source_hashes = {str(p): sha256(p) for p in paths}
    previous = set(json.loads(OLD_HASHES.read_text(encoding='utf-8')).values())
    overlap = sum(value in previous for value in source_hashes.values())
    args.out.mkdir(parents=True, exist_ok=False)
    for folder in ['truth_masks', 'truth_overlay', 'comparison', 'ibn/pred_masks', 'ibn/pred_overlay',
                   'baseline/pred_masks', 'baseline/pred_overlay']:
        (args.out / folder).mkdir(parents=True)
    palette = Image.open(PALETTE).getpalette()
    assert palette is not None and len(palette) == 768
    colors = np.array(palette[:15], dtype=np.uint8).reshape(5, 3)
    palette[255*3:255*3+3] = [100,100,100]
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
    models = {'baseline': YOLO(str(BASE), task='semantic'), 'ibn': YOLO(str(args.weights), task='semantic')}
    assert_ibn(models['ibn'].model)
    for model in models.values():
        assert [model.names[i] for i in range(5)] == NAMES
        model.model.eval().requires_grad_(False)
    matrices = {key: np.zeros((3, 5), dtype=np.int64) for key in models}
    records, clipped_total, overlap_total = [], 0, 0
    started = time.perf_counter()
    for i, (rec, path) in enumerate(zip(data['images'], paths), 1):
        original = Image.open(path).convert('RGB')
        width, height = original.size
        assert (width, height) == (rec['width'], rec['height'])
        truth, ambiguous, clipped = rasterize(grouped[rec['id']], width, height)
        clipped_total += clipped
        overlap_total += ambiguous
        row = {'image': path.name, 'image_id': rec['id'], 'annotated_pixels': int((truth != 255).sum()),
               'ambiguous_pixels': ambiguous, 'unlabeled_pixels': int((truth == 255).sum()) - ambiguous}
        palette_mask(truth, palette, args.out / 'truth_masks' / (path.stem + '.png'))
        truth_overlay = overlay(original, truth, colors)
        truth_overlay.save(args.out / 'truth_overlay' / (path.stem + '.jpg'), quality=92)
        panels = [('Image', original), ('Annotated truth; others unknown', truth_overlay)]
        for key, model in models.items():
            result = model.predict(source=str(path), imgsz=1024, device=args.device, half=False,
                                   augment=False, save=False, verbose=False)[0].cpu()
            prediction = result.semantic_mask.data.numpy().astype(np.uint8)
            assert prediction.shape == (height, width) and np.isin(prediction, np.arange(5)).all()
            for category, (name, class_id, _) in CLASSES.items():
                region = truth == class_id
                hist = np.bincount(prediction[region], minlength=5)
                matrices[key][category] += hist
                row[name + '_truth_pixels'] = int(region.sum())
                row[key + '_' + name + '_correct_pixels'] = int(hist[class_id])
                row[key + '_' + name + '_recall'] = float(hist[class_id]/hist.sum()) if hist.sum() else ''
                for class_name, count in zip(NAMES, hist):
                    row[key + '_' + name + '_to_' + class_name] = int(count)
            palette_mask(prediction, palette, args.out / key / 'pred_masks' / (path.stem + '.png'))
            pred_overlay = overlay(original, prediction, colors)
            pred_overlay.save(args.out / key / 'pred_overlay' / (path.stem + '.jpg'), quality=92)
            panels.append(('Original simulation model' if key == 'baseline' else 'Simulation-trained IBN', pred_overlay))
        card = Image.new('RGB', (2400, 640), '#182634')
        draw = ImageDraw.Draw(card)
        for j, (title, panel) in enumerate(panels):
            view = panel.copy()
            view.thumbnail((600,600), Image.Resampling.LANCZOS)
            card.paste(view, (600*j,40))
            draw.text((600*j+12,9),title,fill='white',font=font)
        card.save(args.out / 'comparison' / (path.stem+'.jpg'),quality=93)
        records.append(row)
        if i == 1 or i % 10 == 0 or i == len(paths):
            print(f'{i}/{len(paths)} completed; elapsed {time.perf_counter()-started:.1f}s',flush=True)
    with (args.out / 'per_image.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    payload = {'created_at': datetime.now().isoformat(), 'source': str(args.source),
               'annotation_sha256': protected[str(annotation_path)], 'images': len(paths),
               'annotations': len(data['annotations']), 'imgsz': 1024, 'device': args.device,
               'training_performed': False, 'weights_selected_on': 'simulation validation mIoU',
               'ibn_training_best_epoch': 1, 'metric': 'micro-aggregated per-class annotated-pixel recall',
               'ignore_id': 255, 'category_mapping': {v[0]: v[1] for v in CLASSES.values()},
               'rasterization': 'clip [0,w-1]/[0,h-1], np.rint, OpenCV fillPoly; per-class union; exclude cross-class overlap',
               'valid_annotated_pixels': sum(r['annotated_pixels'] for r in records),
               'ambiguous_pixels': overlap_total, 'clipped_polygons_including_image_edges': clipped_total,
               'previous_200_overlap': {'matching_images': overlap, 'reference_hashes': str(OLD_HASHES)},
               'inference_and_artifact_seconds': time.perf_counter() - started, 'models': {}}
    for key, weight in [('baseline',BASE),('ibn',args.weights)]:
        payload['models'][key] = {'weights': str(weight), 'sha256': protected[str(weight)],
                                  'per_class': class_metrics(matrices[key], records, key),
                                  'matrix_rows_truth_columns_prediction': matrices[key].tolist()}
    assert np.array_equal(matrices['baseline'].sum(axis=1), matrices['ibn'].sum(axis=1))
    assert sum(matrices['ibn'].sum(axis=1)) == payload['valid_annotated_pixels']
    assert all(sha256(Path(path)) == value for path, value in protected.items())
    assert all(sha256(Path(path)) == value for path, value in source_hashes.items())
    payload['source_images_annotations_weights_unchanged'] = True
    dump(args.out / 'source_hashes.json', source_hashes)
    dump(args.out / 'metrics_annotated_pixels.json', payload)
    reports(args.out, payload, records, colors)
    for key in models:
        print(key, json.dumps(payload['models'][key]['per_class'], ensure_ascii=False), flush=True)
    print('DONE', str(args.out), flush=True)


if __name__ == '__main__':
    main()
