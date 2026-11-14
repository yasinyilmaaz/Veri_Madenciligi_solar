"""Deney 23: MAGFiLO etiketleme protokolünden (Ahmadzadeh ve ark. 2024) türetilen şekil son işlemi.

Protokol: "segmentasyon tek parça olmalı (ada yok), delik içermemeli"; parçalı çizilen filamentlerde
"eksik kısımlar elle tamamlandı". Seçilen her adaya uygulanan varyantlar:
  fill      : delikleri doldur
  close{k}  : k px morfolojik kapama + delik doldurma (parçalar arası küçük boşlukları kapatır)
  bridge{t} : aynı eyer bileşenindeki (prob>0.15) prob>t pikselleriyle parçaları köprüle + delik doldurma
Yeni pikseller yalnızca boş alana eklenir (komşu örneklerle çakışma olmaz). OOF PQ ile 5 fold değerlendirilir.
Örnek: PYTHONPATH=src python experiments/shape_post.py --oof outputs/oof/ens1536_tta2 --cut 0.24
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mu
from scipy.ndimage import binary_fill_holes

from data import get_folds, gt_frame, load_records
from metric import compute_pq, mask_to_rle

VARIANTS = ["none", "fill", "close3", "close5", "close9", "bridge30", "bridge20"]


def reshape(args):
    stem, rles, prob_path = args
    prob = np.load(prob_path).astype(np.float32) / 255
    masks = [mu.decode({"size": [2048, 2048], "counts": r.encode()}).astype(bool) for r in rles]
    cover = cv2.connectedComponents((prob > 0.15).astype(np.uint8), connectivity=8)[1]
    taken = np.zeros((2048, 2048), bool)
    for m in masks:
        taken |= m
    out = {}
    for v in VARIANTS:
        rows, occ = [], taken.copy()
        for k, m in enumerate(masks):
            if v == "none":
                new = m
            else:
                ys, xs = np.nonzero(m)
                y0, y1, x0, x1 = max(ys.min() - 20, 0), min(ys.max() + 21, 2048), max(xs.min() - 20, 0), min(xs.max() + 21, 2048)
                sub = m[y0:y1, x0:x1]
                if v.startswith("close"):
                    kk = int(v[5:])
                    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
                    sub2 = cv2.morphologyEx(sub.astype(np.uint8), cv2.MORPH_CLOSE, ker).astype(bool)
                elif v.startswith("bridge"):
                    t = int(v[6:]) / 100
                    ids = np.unique(cover[y0:y1, x0:x1][sub])
                    sub2 = sub | (np.isin(cover[y0:y1, x0:x1], ids[ids > 0]) & (prob[y0:y1, x0:x1] > t))
                else:
                    sub2 = sub
                sub2 = binary_fill_holes(sub2)
                add = sub2 & ~sub & ~occ[y0:y1, x0:x1]
                new = m.copy()
                new[y0:y1, x0:x1] |= add
                occ[y0:y1, x0:x1] |= add
            rows.append((f"{stem}_{k}", mask_to_rle(new)))
        out[v] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="outputs/oof/ens1536_tta2")
    ap.add_argument("--cut", type=float, default=0.24)
    args = ap.parse_args()
    oof = Path(args.oof)
    folds = get_folds().set_index("stem")["fold"]
    df = pd.read_pickle(oof / "candidates.pkl")[["stem", "inst", "rle"]].merge(
        pd.read_csv(oof / "candidates_scored.csv", usecols=["stem", "inst", "score"]), on=["stem", "inst"])
    df = df[df.score > args.cut]
    jobs = [(s, g.rle.tolist(), oof / f"fold{folds[s]}" / "val_probs" / f"{s}.npy") for s, g in df.groupby("stem")]
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(reshape, jobs, chunksize=4))
    records = load_records()
    gt = gt_frame(records, sorted(records))
    gfold = gt.filament_id.str.split("-", n=1).str[1].str.rsplit("_", n=1).str[0].map(folds)
    for v in VARIANTS:
        pred = pd.DataFrame([r for img in res for r in img[v]], columns=["filament_id", "segmentation_rle"])
        pf = pred.filament_id.str.rsplit("_", n=1).str[0].map(folds)
        r = compute_pq(gt, pred)
        per = [compute_pq(gt[gfold == f], pred[pf == f])["pq"] for f in range(5)]
        print(f"{v:9s} PQ {r['pq']:.4f} SQ {r['sq']:.4f} RQ {r['rq']:.4f} TP {r['tp']} | fold: " + " ".join(f"{x:.4f}" for x in per), flush=True)


if __name__ == "__main__":
    main()
