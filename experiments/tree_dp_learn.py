"""Deney 29: eşik ağacı + kesin DP seçimi — öğrenilmiş düğüm skoru ile 5 fold değerlendirme.

tree_dp.py'deki ağacı her görüntü için kurar. Düğüm öznitelikleri: scorer.instance_features (21 öznitelik)
+ seviye eşiği, çocuk sayısı, ebeveyn/düğüm alan oranı, çocukların alan payı (kararlılık).
y_hat iç içe CV ile (fold f tahmini, f dışındaki fold'lardan) LightGBM'le öğrenilir.
DP: best(n) = max(ŷ(n) - c, Σ best(çocuk)), c kesme taraması.
Deney 31: --levels "L:T,..." ile düğümler eyer bağlantılı kurulur: bileşen = CC(prob>L) ∩ (prob>T)
(post.prob_to_instances ile aynı). L ve T azalmayan sırada verilirse düğümler iç içe kalır.
Örnek: PYTHONPATH=src python experiments/tree_dp_learn.py --oof outputs/oof/ens1536_tta2
       PYTHONPATH=src python experiments/tree_dp_learn.py --levels 0.15:0.25,0.15:0.4,0.25:0.4,0.4:0.4,0.55:0.55,0.7:0.7
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd

from data import TRAIN_IMG, get_folds, gt_frame, read_image
from metric import compute_pq, mask_to_rle
from scorer import FEATS, PARAMS, ROUNDS, instance_features
from tree_dp import LEVELS, REC, y_of
from data import poly_to_rle

TF = FEATS + ["level", "link", "n_kids", "par_ratio", "kid_cover"]


def build(args):
    stem, prob_path, fold, levels = args
    prob = np.load(prob_path).astype(np.float32) / 255
    img = read_image(TRAIN_IMG / f"{stem}.jpeg")
    gts = [[poly_to_rle(a["seg"]) for a in an] for an in REC[stem].values()]
    rows, prev = [], None
    for L, t in levels:
        n, lab = cv2.connectedComponents((prob > L).astype(np.uint8), connectivity=8)
        lab[prob <= t] = 0
        areas = np.bincount(lab.ravel(), minlength=n)
        lab[np.isin(lab, np.nonzero(areas < 100)[0])] = 0
        feats = {r["inst"]: r for r in instance_features(prob, img, lab, t)}
        cur = np.full(lab.shape, -1, np.int32)
        for k, f in feats.items():
            m = lab == k
            par = -1
            if prev is not None:
                pv = prev[m]
                pv = pv[pv >= 0]
                par = int(np.bincount(pv).argmax()) if pv.size else -1
            idx = len(rows)
            cur[m] = idx
            rows.append(dict(f, stem=stem, fold=fold, idx=idx, parent=par, level=t, link=L, rle=mask_to_rle(m)))
        prev = cur
    for r in rows:
        kids = [c for c in rows if c["parent"] == r["idx"]]
        r["n_kids"] = len(kids)
        r["kid_cover"] = sum(c["area"] for c in kids) / r["area"]
        r["par_ratio"] = r["area"] / rows[r["parent"]]["area"] if r["parent"] >= 0 else 1.0
    y = y_of([r["rle"] for r in rows], gts)
    for r, v in zip(rows, y):
        r["y"] = v
    return rows


def dp_select(g, col, c, margin=0.0, base=None):
    """margin > 0: taban düğümler (base) ile onların üstündeki birleştirme düğümleri, seçimin ancak
    margin kadar daha iyi olduğunda değiştirilir (bölme ya da birleştirme için ön yargı; deney 31)."""
    s = (g[col] - c).to_numpy()
    par = g.parent.to_numpy()
    base = np.zeros(len(g), bool) if base is None else base
    above = np.zeros(len(g), bool)  # tabanın üstündeki (birleştirme) düğümler
    for i in np.nonzero(base)[0]:
        p = par[i]
        while p >= 0 and not above[p]:
            above[p] = True
            p = par[p]
    kids = [[] for _ in range(len(g))]
    for i, p in enumerate(par):
        if p >= 0:
            kids[p].append(i)
    best, pick = np.zeros(len(g)), [None] * len(g)
    for i in reversed(range(len(g))):
        cs = sum(best[k] for k in kids[i])
        need = cs - margin if base[i] else cs + margin if above[i] else cs
        if s[i] >= need and s[i] > 0:
            best[i], pick[i] = s[i], [i]
        else:
            best[i], pick[i] = cs, [j for k in kids[i] for j in pick[k]]
    return [j for i in np.nonzero(par < 0)[0] for j in pick[i]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="outputs/oof/ens1536_tta2")
    ap.add_argument("--bonus", default="0", help="taban seviyesi (eyer 0,15/0,4) ön yargı marjları (virgülle)")
    ap.add_argument("--levels", default=",".join(f"{t}:{t}" for t in LEVELS), help="L:T çiftleri (deney 31)")
    args = ap.parse_args()
    levels = [tuple(map(float, x.split(":"))) for x in args.levels.split(",")]
    folds = get_folds().set_index("stem")["fold"]
    default = args.levels == ",".join(f"{t}:{t}" for t in LEVELS)
    cache = Path("outputs/tree_dp/nodes.pkl" if default else f"outputs/tree_dp/nodes_{args.levels.replace(':', '-').replace(',', '_')}.pkl")
    if cache.exists():
        D = pd.read_pickle(cache)
    else:
        jobs = [(s, Path(args.oof) / f"fold{folds[s]}" / "val_probs" / f"{s}.npy", int(folds[s]), levels) for s in sorted(REC)]
        with ProcessPoolExecutor(12) as ex:
            D = pd.DataFrame([r for img in ex.map(build, jobs, chunksize=4) for r in img])
        D.to_pickle(cache)
    # scorer.process_image'deki görüntü düzeyi öznitelikler: istasyon ve aynı seviyedeki aday sayısı
    D["station"] = D.stem.str[-2].map(ord)
    if "link" not in D:
        D["link"] = D.level
    D["n_inst_img"] = D.groupby(["stem", "level", "link"]).idx.transform("count")
    D["yhat"] = np.nan
    for f in range(5):
        tr, va = D.fold != f, D.fold == f
        m = lgb.train(dict(PARAMS, num_threads=12), lgb.Dataset(D.loc[tr, TF], D.loc[tr, "y"]), ROUNDS)
        D.loc[va, "yhat"] = m.predict(D.loc[va, TF])
    gt = gt_frame(REC, sorted(REC))
    gfold = gt.filament_id.str.split("-", n=1).str[1].str.rsplit("_", n=1).str[0].map(folds)
    print(f"{len(D)} düğüm, {D.stem.nunique()} görüntü", flush=True)
    margins = [float(x) for x in args.bonus.split(",")]
    D["is_base"] = (D.link == 0.15) & (D.level == 0.4)
    grid = [("y", 0.2275, 0.0)] + [("yhat", c, mg) for mg in margins for c in [0.22, 0.24, 0.26, 0.28]]
    groups = [(s, g.reset_index(drop=True)) for s, g in D.groupby("stem")]
    for col, c, mg in grid:
        rows = []
        for s, g in groups:
            for j in dp_select(g, col, c, mg, g.is_base.to_numpy()):
                rows.append((f"{s}_t{j}", g.rle[j]))
        pred = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        pf = pred.filament_id.str.rsplit("_", n=1).str[0].map(folds)
        r = compute_pq(gt, pred)
        per = [compute_pq(gt[gfold == f], pred[pf == f])["pq"] for f in range(5)]
        print(f"{col:4s} marj {mg:.2f} kesme {c:.4f}: PQ {r['pq']:.4f} SQ {r['sq']:.4f} RQ {r['rq']:.4f} TP {r['tp']} FP {r['fp']} FN {r['fn']} | "
              + " ".join(f"{x:.4f}" for x in per), flush=True)


if __name__ == "__main__":
    main()
