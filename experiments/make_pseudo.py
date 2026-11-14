"""Etiketsiz GONG görüntüleri için fold'a özel pseudo-label (olasılık haritası) üretimi.

Fold f'nin öğrencisi yalnızca fold f'nin öğretmeninin etiketlerini görür;
böylece validasyon görüntülerinin bilgisi CV'ye sızmaz.

Örnek: PYTHONPATH=src python experiments/make_pseudo.py --teacher outputs/unet_r34 --out outputs/pseudo/unet_r34
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from data import EXT_IMG, OUT, read_image
from predict import load_fold_model
from train import predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--keep", default=str(OUT / "gong_keep.csv"), help="quality.py filtresinden geçen kareler")
    args = ap.parse_args()

    paths = [EXT_IMG / f"{s}.jpeg" for s in sorted(pd.read_csv(args.keep).stem)]
    for f in args.folds:
        model = load_fold_model(Path(args.teacher) / f"fold{f}")
        out = Path(args.out) / f"fold{f}"
        out.mkdir(parents=True, exist_ok=True)
        kept = 0
        for p in paths:
            img = read_image(p)
            prob = predict(model, img, tta=True)
            cv2.imwrite(str(out / f"{p.stem}.png"), (prob * 255).astype(np.uint8))
            kept += 1
        print(f"fold{f}: {kept}/{len(paths)} görüntü", flush=True)


if __name__ == "__main__":
    main()
