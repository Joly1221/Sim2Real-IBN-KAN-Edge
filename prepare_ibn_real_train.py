"""【新增 IBN-REAL-01】仅转换指定真实 train 的全部图像，不划分验证集，不读取 test。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from runtime import ROOT, NAMES, torch
import ibn_kan_edge  # 注册现有边缘模块，保留原实现。
from edge_branch import crater_boundary
from evaluate_kaggle_test import rasterize, sha256

SOURCE = Path(r'D:\Projects\Sim2real\data\real_data\kaggle_dataset\test\train')
DESTINATION = ROOT / 'real_adapter_train_data'


def write_yaml(destination: Path) -> Path:
    # 【框架占位】val 指向 train 只满足数据配置接口，专用训练器关闭所有验证调用。
    text = (f'path: {destination.as_posix()}\ntrain: images/train\nval: images/train\n'
            'masks_dir: masks\nnames:\n' + ''.join(f'  {i}: {n}\n' for i,n in enumerate(NAMES)))
    path = destination / 'data.yaml'
    path.write_text(text, encoding='utf-8')
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=DESTINATION)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'拒绝覆盖已有数据副本：{args.output}')
    torch.set_num_threads(1)
    annotation_path = args.source / 'annotations.json'
    annotation_hash = sha256(annotation_path)
    data = json.loads(annotation_path.read_text(encoding='utf-8'))
    assert {c['id']:c['name'] for c in data['categories']} == {0:'Craters',1:'Rocks',2:'Sky'}
    ids = {r['id'] for r in data['images']}
    assert len(ids) == len(data['images'])
    grouped = defaultdict(list)
    for annotation in data['annotations']:
        assert annotation['image_id'] in ids
        grouped[annotation['image_id']].append(annotation)
    paths = [(args.source / rec['file_name']).resolve() for rec in data['images']]
    assert all(p.is_relative_to(args.source.resolve()) and p.is_file() for p in paths)
    assert len({p.stem for p in paths}) == len(paths)
    assert set(paths) == {p.resolve() for p in (args.source/'images').iterdir() if p.is_file()}
    for folder in ['images/train','masks/train']:
        (args.output/folder).mkdir(parents=True)
    records = []
    for rec, source in zip(data['images'], paths):
        with Image.open(source) as image:
            assert image.size == (rec['width'],rec['height'])
        mask, overlap, clipped = rasterize(grouped[rec['id']],rec['width'],rec['height'])
        assert np.isin(mask,[1,2,4,255]).all()
        digest = sha256(source)
        image_copy = args.output/'images/train'/source.name
        mask_path = args.output/'masks/train'/(source.stem+'.png')
        shutil.copy2(source,image_copy)
        Image.fromarray(mask).save(mask_path)
        assert sha256(image_copy) == digest
        assert np.array_equal(np.array(Image.open(mask_path)),mask)
        edge, known = crater_boundary(torch.from_numpy(mask).unsqueeze(0))
        pixels = np.bincount(mask[mask!=255],minlength=5).tolist()
        records.append({'image_id':rec['id'],'file':source.name,'source':str(source),
                        'sha256':digest,'mask_sha256':sha256(mask_path),'pixels_by_class':pixels,
                        'ambiguous_pixels':overlap,'unknown_pixels':int((mask==255).sum()),
                        'clipped_polygons':clipped,'reliable_edge_positive_pixels':int(edge.sum()),
                        'reliable_edge_valid_pixels':int(known.sum())})
    assert len({r['sha256'] for r in records}) == len(records), '训练目录中发现完全相同的图像'
    assert sha256(annotation_path) == annotation_hash
    assert all(sha256(Path(r['source']))==r['sha256'] for r in records)
    yaml = write_yaml(args.output)
    report = {'source':str(args.source),'annotation_sha256':annotation_hash,'images':len(records),
              'annotations':len(data['annotations']),'all_images_used_for_training':True,
              'validation_enabled':False,'test_used':False,'class_mapping':{'Craters':1,'Rocks':2,'Sky':4},
              'ignore_id':255,'unlabeled_policy':'unknown, not lunar regolith',
              'overlap_policy':'cross-class overlap ignored; same-class polygons unioned',
              'pixels_by_class':np.asarray([r['pixels_by_class'] for r in records]).sum(0).tolist(),
              'reliable_edge_positive_pixels':sum(r['reliable_edge_positive_pixels'] for r in records),
              'images_with_reliable_positive_edges':sum(r['reliable_edge_positive_pixels']>0 for r in records),
              'source_unchanged':True,'records':records}
    (args.output/'provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False,indent=2))
    print('DATA',yaml)


if __name__ == '__main__':
    main()
