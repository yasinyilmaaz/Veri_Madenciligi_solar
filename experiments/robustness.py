"""Dayanıklılık testi: validasyon görüntüleri bozulunca modeller ne kadar etkileniyor?

Her model kendi validasyon fold'unda, bozulmamış ve bozulmuş görüntülerde değerlendirilir.
Post-processing her model için temiz görüntüde seçilen ayarla sabit tutulur.
Örnek: PYTHONPATH=src python experiments/robustness.py
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from data import OUT, TRAIN_IMG, get_folds, gt_frame, load_records, read_image
from metric import compute_pq, instances_to_rows
from post import prob_to_instances
from predict import load_fold_model
from train import predict

# model: (run klasörü, temiz görüntüde seçilmiş post ayarı)
MODELS = {
    "baseline": ("unet_r34", dict(thr=0.6, min_area=400, join=8, thr_high=0.9)),
    "aug_strong": ("aug_strong", dict(thr=0.4, min_area=300, join=8, thr_high=0.9)),
    "tgt_mean": ("tgt_mean", dict(thr=0.6, min_area=400, join=8, thr_high=0.9)),
}


def perturb(img, kind, rng):
    x = img.astype(np.float32)
    if kind == "temiz":
        return img
    if kind == "bulanık":
        x = cv2.GaussianBlur(x, (0, 0), 2.0)
    elif kind == "gürültü":
        x = x + rng.normal(0, 10, x.shape)
    elif kind == "düşük kontrast":
        m = x[x > 30].mean()
        x = m + 0.65 * (x - m)
    elif kind == "gamma":
        x = 255 * (x / 255) ** 1.5
    elif kind == "jpeg":
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 30])
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    elif kind == "bulut":
        small = rng.normal(0, 1, (4, 4)).astype(np.float32)
        field = cv2.resize(small, (2048, 2048), interpolation=cv2.INTER_CUBIC)
        x = x * (1 + 0.25 * field / np.abs(field).max())
    return np.clip(x, 0, 255).astype(np.uint8)


KINDS = ["temiz", "bulanık", "gürültü", "düşük kontrast", "gamma", "jpeg", "bulut"]


def main():
    records = load_records()
    folds = get_folds()
    res = []
    for f in [0, 1]:
        stems = folds.loc[folds.fold == f, "stem"].tolist()
        gt = gt_frame(records, stems)
        imgs = {s: read_image(TRAIN_IMG / f"{s}.jpeg") for s in stems}
        for name, (run, cfg) in MODELS.items():
            model = load_fold_model(OUT / run / f"fold{f}")
            for kind in KINDS:
                rng = np.random.default_rng(0)
                rows = []
                for s in stems:
                    prob = predict(model, perturb(imgs[s], kind, rng), tta=1)
                    rows += instances_to_rows(s, prob_to_instances(prob, **cfg))
                pred = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
                r = compute_pq(gt, pred)
                res.append(dict(fold=f, model=name, bozulma=kind, **r))
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[-1].items()},
                                 ensure_ascii=False, default=float), flush=True)
    df = pd.DataFrame(res)
    agg = df.groupby(["model", "bozulma"], sort=False)[["tp", "fp", "fn"]].sum()
    sums = df.assign(siou=df.sq * df.tp).groupby(["model", "bozulma"], sort=False)["siou"].sum()
    agg["pq"] = sums / (agg.tp + 0.5 * agg.fp + 0.5 * agg.fn)
    table = agg["pq"].unstack(0).loc[KINDS]
    print("\nPQ (fold 0+1, flip TTA)\n", table.round(4).to_string())
    print("\nTemize göre düşüş\n", (table - table.loc["temiz"]).round(4).to_string())
    df.to_csv(OUT / "robustness.csv", index=False)


if __name__ == "__main__":
    main()
