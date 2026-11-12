"""【新增 IBN-KE-06】加载含 IBN、KAN、edge 的完整模型预测；不训练。"""
from __future__ import annotations

import argparse
from pathlib import Path
from ibn_kan_edge import YOLO, torch, assert_combo
from runtime import ROOT
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--name', default='ibn_kan_edge_predict')
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = YOLO(str(args.weights), task='semantic')
    assert_combo(model.model)
    count = 0
    for result in model.predict(args.source, imgsz=args.imgsz, device=args.device, save=True,
                                stream=True, project=str(ROOT/'predictions'), name=args.name, exist_ok=False):
        folder = Path(result.save_dir)/'class_masks'
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.semantic_mask.data.cpu().numpy().astype('uint8')).save(folder/(Path(result.path).name+'.png'))
        count += 1
    print(f'已预测 {count} 张图像；class_masks 为 0～4 类别索引。')


if __name__ == '__main__':
    main()
