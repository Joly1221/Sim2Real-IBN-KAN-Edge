"""【新增 IBN-REAL-EVAL-02】从保存掩膜独立重算计数，核验报告与文件链接。"""
from collections import defaultdict
import csv
import json
from pathlib import Path
import re

from evaluate_kaggle_test import np, Image, rasterize, sha256, dump
from runtime import ROOT


def main():
    out=ROOT/'kaggle_full_model_test_20261112/output'
    payload=json.loads((out/'metrics_annotated_pixels.json').read_text(encoding='utf-8'))
    source=Path(payload['source'])
    annotation=json.loads((source/'annotations.json').read_text(encoding='utf-8'))
    grouped=defaultdict(list)
    for row in annotation['annotations']:
        grouped[row['image_id']].append(row)
    counts={key:np.zeros((3,5),dtype=np.int64) for key in payload['models']}
    totals={key:np.zeros(5,dtype=np.int64) for key in counts}
    unknown={key:np.zeros(5,dtype=np.int64) for key in counts}
    image_count=0
    for record in annotation['images']:
        stem=Path(record['file_name']).stem
        truth=np.asarray(Image.open(out/'truth_masks'/(stem+'.png')))
        expected,_,_=rasterize(grouped[record['id']],record['width'],record['height'])
        assert np.array_equal(truth,expected)
        assert set(np.unique(truth)).issubset({1,2,4,255})
        for key in counts:
            pred=np.asarray(Image.open(out/key/'pred_masks'/(stem+'.png')))
            assert pred.shape==truth.shape==(record['height'],record['width'])
            assert set(np.unique(pred)).issubset(set(range(5)))
            for k in range(5):
                totals[key][k]+=np.count_nonzero(pred==k)
                unknown[key][k]+=np.count_nonzero((pred==k)&(truth==255))
                for j,true_id in enumerate([1,2,4]):
                    counts[key][j,k]+=np.count_nonzero((pred==k)&(truth==true_id))
        image_count+=1
    for key in counts:
        assert np.array_equal(counts[key],payload['models'][key]['matrix_rows_truth_columns_prediction'])
        assert len(list((out/key/'pred_masks').glob('*.png')))==image_count
        assert sha256(Path(payload['models'][key]['weights']))==payload['models'][key]['sha256']
    assert len(list((out/'comparison').glob('*.jpg')))==image_count
    hashes=json.loads((out/'source_hashes.json').read_text(encoding='utf-8'))
    assert all(sha256(Path(p))==h for p,h in hashes.items())
    assert sha256(source/'annotations.json')==payload['annotation_sha256']
    page=(out/'index.html').read_text(encoding='utf-8')
    refs=re.findall(r'(?:href|src)="([^"]+)"',page)
    assert all((out/ref).is_file() for ref in refs)
    statistics={key:{'predicted_pixels_by_class':totals[key].tolist(),
                     'predicted_percentage_by_class':(totals[key]/totals[key].sum()*100).tolist(),
                     'unknown_region_prediction_percentage_by_class':(unknown[key]/unknown[key].sum()*100).tolist()}
                for key in counts}
    verification={'image_count':image_count,'mask_matrix_recomputation_exact':True,'truth_recomputed_from_original_annotations':True,
                  'all_masks_original_resolution':True,'html_references_checked':len(refs),'all_html_references_exist':True,
                  'source_images_annotations_weights_unchanged':True,'class_order':['regolith','crater','rock','mountain','sky'],
                  'prediction_area_diagnostics':statistics,'training_performed':False}
    dump(out/'evaluation_verification.json',verification)
    with (out/'per_image.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    views=[]
    for category in ['Craters','Rocks']:
        row=max(rows,key=lambda r:int(r[category+'_truth_pixels']))
        views.append(str(out/'comparison'/(Path(row['image']).stem+'.jpg')))
    print(json.dumps({'verification':verification,'review_images':views},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
