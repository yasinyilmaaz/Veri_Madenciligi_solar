"""Deney 29: iç içe eşik ağacı + kesin DP seçimi — önce oracle tavanı.

Her görüntüde olasılık haritası LEVELS eşiklerinde bileşenlere ayrılır; üst eşiğin bileşeni alt eşiğin
bileşeninin içinde kaldığı için bir ağaç oluşur. Düğüm skoru s = y - PQ/2 (y: anotatörler üzerinden
IoU·[IoU>0,5] ortalaması). DP: best(n) = max(s(n), Σ best(çocuk)); seçilen düğümler çakışmaz.
Oracle (gerçek y) ile mevcut tek seviyeli adayların oracle'ı karşılaştırılır (erken durdurma ölçütü).
Örnek: PYTHONPATH=src python experiments/tree_dp.py --oof outputs/oof/ens1536_tta2 --folds 0 1
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mu

from data import get_folds, gt_frame, load_records, poly_to_rle
from metric import compute_pq, mask_to_rle
from post import prob_to_instances

LEVELS = [0.25, 0.4, 0.55, 0.7]
HALF_PQ = 0.2275
REC = load_records()


def y_of(rles, gts):
    if not rles:
        return np.zeros(0)
    P = [{"size": [2048, 2048], "counts": r.encode()} for r in rles]
    ys = np.zeros(len(P))
    for G in gts:
        if G:
            iou = np.asarray(mu.iou(P, G, [0] * len(G))).reshape(len(P), len(G)).max(1)
            ys += np.where(iou > 0.5, iou, 0)
    return ys / len(gts)


def image(args):
    stem, prob_path = args
    prob = np.load(prob_path).astype(np.float32) / 255
    gts = [[poly_to_rle(a["seg"]) for a in an] for an in REC[stem].values()]
    nodes, parent, prev = [], [], None
    for li, t in enumerate(LEVELS):
        n, lab = cv2.connectedComponents((prob > t).astype(np.uint8), connectivity=8)
        areas = np.bincount(lab.ravel(), minlength=n)
        ids = {}
        for k in range(1, n):
            if areas[k] < 100:
                continue
            m = lab == k
            par = -1
            if prev is not None:
                pv = prev[m]
                pv = pv[pv >= 0]
                par = int(np.bincount(pv).argmax()) if pv.size else -1
            ids[k] = len(nodes)
            nodes.append(mask_to_rle(m))
            parent.append(par)
        cur = np.full(lab.shape, -1, np.int32)
        for k, idx in ids.items():
            cur[lab == k] = idx
        prev = cur
    y = y_of(nodes, gts)
    s = y - HALF_PQ
    kids = [[] for _ in nodes]
    for i, p in enumerate(parent):
        if p >= 0:
            kids[p].append(i)
    best, pick = np.zeros(len(nodes)), [None] * len(nodes)
    for i in reversed(range(len(nodes))):  # çocuklar her zaman ebeveynden sonra eklendi
        cs = sum(best[c] for c in kids[i])
        if s[i] >= cs and s[i] > 0:
            best[i], pick[i] = s[i], [i]
        else:
            best[i], pick[i] = cs, [j for c in kids[i] for j in pick[c]]
    roots = [i for i, p in enumerate(parent) if p < 0]
    chosen = [j for r in roots for j in pick[r]]
    tree_rows = [(f"{stem}_t{j}", nodes[j]) for j in chosen]
    # karşılaştırma: mevcut tek seviyeli adaylar (eyer bağlantısı) + oracle seçim
    lab = prob_to_instances(prob, thr=0.4, min_area=100, link_thr=0.15)
    flat = [mask_to_rle(lab == k) for k in range(1, lab.max() + 1)]
    yf = y_of(flat, gts)
    flat_rows = [(f"{stem}_f{k}", r) for k, (r, v) in enumerate(zip(flat, yf)) if v > HALF_PQ]
    return tree_rows, flat_rows, len(nodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="outputs/oof/ens1536_tta2")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    folds = get_folds().set_index("stem")["fold"]
    stems = [s for s in sorted(REC) if folds[s] in args.folds]
    jobs = [(s, Path(args.oof) / f"fold{folds[s]}" / "val_probs" / f"{s}.npy") for s in stems]
    with ProcessPoolExecutor(12) as ex:
        res = list(ex.map(image, jobs, chunksize=4))
    gt = gt_frame(REC, stems)
    for name, k in [("tek seviye (eyer) oracle", 1), ("ağaç + DP oracle", 0)]:
        pred = pd.DataFrame([r for img in res for r in img[k]], columns=["filament_id", "segmentation_rle"])
        r = compute_pq(gt, pred)
        print(f"{name:26s} PQ {r['pq']:.4f} SQ {r['sq']:.4f} RQ {r['rq']:.4f} TP {r['tp']} FP {r['fp']} FN {r['fn']}", flush=True)
    print("ortalama düğüm/görüntü:", round(np.mean([x[2] for x in res]), 1))


if __name__ == "__main__":
    main()
