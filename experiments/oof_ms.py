"""Deney 09: çok ölçekli TTA ile OOF olasılıkları.

Her ölçekte görüntü yeniden boyutlanır (32'nin katına), flip TTA ile tahmin edilir, olasılık 2048'e
geri ölçeklenir; ölçekler ortalanır. Çıktı outputs/oof/{run}_ms{tag}_tta2 (eval_runs ile değerlendirilir).
Örnek: PYTHONPATH=src python experiments/oof_ms.py --run effv2m_mean --scales 0.85 1.0 1.15
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from data import OUT, TRAIN_IMG, get_folds, read_image
from predict import load_fold_model
from train import predict


def predict_ms(model, img, scales, tta=1):
    acc = 0
    for s in scales:
        if s == 1.0:
            acc = acc + predict(model, img, tta=tta)
            continue
        n = int(round(2048 * s / 32)) * 32
        p = predict(model, cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC),
                    tta=tta)
        acc = acc + cv2.resize(p, (2048, 2048), interpolation=cv2.INTER_LINEAR)
    return acc / len(scales)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.85, 1.0, 1.15])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    folds = get_folds()
    out_root = OUT / "oof" / f"{args.run}_ms{args.tag}_tta2"
    for f in args.folds:
        model = load_fold_model(OUT / args.run / f"fold{f}")
        out = out_root / f"fold{f}" / "val_probs"
        out.mkdir(parents=True, exist_ok=True)
        for s in folds.loc[folds.fold == f, "stem"]:
            prob = predict_ms(model, read_image(TRAIN_IMG / f"{s}.jpeg"), args.scales, args.tta)
            np.save(out / f"{s}.npy", (prob * 255).astype(np.uint8))
        print(f"fold{f} bitti", flush=True)


if __name__ == "__main__":
    main()
