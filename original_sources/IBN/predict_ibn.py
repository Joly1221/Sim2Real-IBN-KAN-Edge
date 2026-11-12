"""【新增文件 IBN-04】预测入口；先注册 IBN 自定义类型，再加载完整权重。"""
from __future__ import annotations

import argparse
from pathlib import Path
from runtime import ROOT, YOLO, torch
from ibn_backbone import assert_ibn
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='图像或图像目录')
    parser.add_argument('--weights', type=Path, required=True, help='指定训练后 best.pt；初始权重仅用于调试')
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--name', default='predict')
    args = parser.parse_args()
    torch.set_num_threads(1)
    yolo = YOLO(str(args.weights), task='semantic')
    assert_ibn(yolo.model)
    count = 0
    for result in yolo.predict(source=args.source, imgsz=args.imgsz, device=args.device,
                               save=True, stream=True, project=str(ROOT / 'predictions'),
                               name=args.name, exist_ok=False, verbose=True):
        directory = Path(result.save_dir) / 'class_masks'
        directory.mkdir(parents=True, exist_ok=True)
        mask = result.semantic_mask.data.cpu().numpy().astype('uint8').squeeze()
        Image.fromarray(mask).save(directory / (Path(result.path).name + '.png'))
        count += 1
    print(f'预测完成：{count} 张；class_masks 是 0～4 类别索引，不是 RGB 真值颜色。')


if __name__ == '__main__':
    main()
