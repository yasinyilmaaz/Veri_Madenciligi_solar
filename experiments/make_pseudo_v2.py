"""Güçlü öğretmenle pseudo-label (v2): fold'a özel ensemble + fold'a özel aday seçici.

Fold f için:
  - öğretmen: fold f'nin 3 modeli (r34, effv2s, cnxt; ortalama hedef), flip TTA
  - aday seçici: OOF adaylarından fold f HARİÇ eğitilmiş LightGBM (fold f'nin etiketleri sızmaz)
  - skor > HIGH olan adaylar kesin filament (maske=255)
  - skor LOW..HIGH arası adaylar ve kesin filamentlerin kenar bandı yok sayılır (ağırlık=0)
Yalnızca test dönemi dışı kareler (GONG 2010, 2023-2026) kullanılır.

Örnek: PYTHONPATH=src python experiments/make_pseudo_v2.py --folds 0 1
"""
import argparse
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd

from data import OUT, ROOT, read_image
from post import prob_to_instances
from predict import load_fold_model
from scorer import CAND, FEATS, PARAMS, ROUNDS, instance_features
from train import predict

RUNS = ["tgt_mean", "effv2s_mean", "cnxt_mean"]
HIGH, LOW, BAND = 0.4, 0.1, 7
IMG_DIR = ROOT / "external" / "gong_new"


def fold_scorer(f):
    df = pd.read_csv(OUT / "oof" / "ens3_mean_tta2" / "candidates_scored.csv")
    tr = df[df.fold != f]
    return lgb.train(PARAMS, lgb.Dataset(tr[FEATS], tr["y"]), num_boost_round=ROUNDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--keep", default=str(OUT / "gong_new_keep.csv"))
    args = ap.parse_args()
    stems = sorted(pd.read_csv(args.keep).stem)
    band_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BAND + 1, 2 * BAND + 1))
    for f in args.folds:
        models = [load_fold_model(OUT / r / f"fold{f}") for r in RUNS]
        scorer = fold_scorer(f)
        out = OUT / "pseudo_v2" / f"fold{f}"
        out.mkdir(parents=True, exist_ok=True)
        rows, n_pos, n_ign = [], 0, 0
        for s in stems:
            img = read_image(IMG_DIR / f"{s}.jpeg")
            prob = sum(predict(m, img, tta=1) for m in models) / len(models)
            lab = prob_to_instances(prob, **CAND)
            feats = instance_features(prob, img, lab, CAND["thr"])
            mask = np.zeros((2048, 2048), np.uint8)
            weight = np.full((2048, 2048), 255, np.uint8)
            if feats:
                X = pd.DataFrame(feats)
                X["n_inst_img"], X["station"] = len(feats), ord(s[-2])
                sc = scorer.predict(X[FEATS])
                pos = X.inst.to_numpy()[sc > HIGH]
                mid = X.inst.to_numpy()[(sc > LOW) & (sc <= HIGH)]
                mask[np.isin(lab, pos)] = 255
                band = (cv2.dilate(mask, band_k) > 0) & (cv2.erode(mask, band_k) == 0)
                weight[band | np.isin(lab, mid)] = 0
                n_pos += len(pos)
                n_ign += len(mid)
            cv2.imwrite(str(out / f"{s}_m.png"), mask)
            cv2.imwrite(str(out / f"{s}_w.png"), weight)
            rows.append(dict(img=str(IMG_DIR / f"{s}.jpeg"), mask=str(out / f"{s}_m.png"),
                             weight=str(out / f"{s}_w.png"), date=f"{s[:4]}-{s[4:6]}-{s[6:8]}"))
        pd.DataFrame(rows).to_csv(OUT / "pseudo_v2" / f"pairs_fold{f}.csv", index=False)
        print(f"fold{f}: {len(rows)} kare, {n_pos} kesin filament, {n_ign} yok sayılan aday", flush=True)


if __name__ == "__main__":
    main()
