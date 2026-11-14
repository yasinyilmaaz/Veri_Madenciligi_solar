"""Deney 34: pencereli çıkarım — normalizasyonu eğitimle eşitle.

Modeller 1536 kesitte, kesitin kendi istatistiğiyle (disk pikselleri I>30) normalize edilerek eğitildi;
train.predict ise tam 2048 görüntünün istatistiğini kullanıyor (ölçülen fark: offset ~0,24σ, ölçek ~0,85).
Burada görüntü 4 adet 1536 pencereyle (köşeler 0 ve 512) kaplanır, her pencere kendi istatistiğiyle
normalize edilip D4 TTA ile tahmin edilir, örtüşmeler yumuşak (Hann) ağırlıkla birleştirilir.
Örnek: PYTHONPATH=src python experiments/tiled.py --run crop1536_effv2s --folds 0 1   (OOF -> outputs/oof/<run>tiled_tta2)
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from data import TRAIN_IMG, get_folds, read_image
from predict import load_fold_model
from train import normalize

WIN, OFFS = 1536, (0, 512)
_w1 = np.hanning(WIN + 2)[1:-1].astype(np.float32) + 1e-3
WEIGHT = np.outer(_w1, _w1)


def predict_tiled(model, img, tta=2):
    acc = np.zeros((2048, 2048), np.float32)
    wsum = np.zeros((2048, 2048), np.float32)
    flips = [[], [3], [2], [2, 3]] if tta else [[]]
    transposes = [False, True] if tta == 2 else [False]
    for y0 in OFFS:
        for x0 in OFFS:
            x = torch.from_numpy(normalize(img[y0:y0 + WIN, x0:x0 + WIN]))[None, None].cuda()
            p, n = 0, 0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                for t in transposes:
                    xt = x.transpose(2, 3) if t else x
                    for f in flips:
                        xi = xt.flip(f) if f else xt
                        q = torch.sigmoid(model(xi)[:, :1].float())
                        q = q.flip(f) if f else q
                        p = p + (q.transpose(2, 3) if t else q)
                        n += 1
            acc[y0:y0 + WIN, x0:x0 + WIN] += (p / n)[0, 0].cpu().numpy() * WEIGHT
            wsum[y0:y0 + WIN, x0:x0 + WIN] += WEIGHT
    return acc / wsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    folds = get_folds()
    for f in args.folds:
        m = load_fold_model(Path("outputs") / args.run / f"fold{f}")
        out = Path("outputs/oof") / f"{args.run}tiled_tta2" / f"fold{f}" / "val_probs"
        out.mkdir(parents=True, exist_ok=True)
        for s in folds.loc[folds.fold == f, "stem"]:
            prob = predict_tiled(m, read_image(TRAIN_IMG / f"{s}.jpeg"))
            np.save(out / f"{s}.npy", (prob * 255).astype(np.uint8))
        print(f"fold{f} bitti", flush=True)


if __name__ == "__main__":
    main()
