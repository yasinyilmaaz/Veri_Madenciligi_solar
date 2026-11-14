"""Deney 04: aday seçici varyantları (iç içe CV, ens4 aday havuzu).

Varyantlar:
  base      : mevcut LightGBM regresyonu (y = anotatörler üzerinden ortalama PQ katkısı)
  pq2       : iki model; p = P(en az bir anotatörle IoU>0.5), q = E[IoU | eşleşme] -> skor p*q
  hp_*      : hiper parametre varyantları (yaprak, tur, öğrenme hızı, min_data)
  imgfeat   : görüntü düzeyinde bağlam öznitelikleri (aday sayısı, ort. olasılık, ort. kontrast...)
Hepsi aynı adaylar ve aynı GT ile; kesme eşiği her varyant için ızgarada seçilir.

Örnek: PYTHONPATH=src python experiments/scorer_variants.py outputs/oof/ens4_tta2
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data import get_folds, gt_frame, load_records
from metric import compute_pq
from scorer import FEATS, PARAMS, ROUNDS

CUTS = np.round(np.arange(0.12, 0.40, 0.02), 2)


def nested(df, feats, target, params, rounds, objective="regression"):
    out = np.full(len(df), np.nan)
    p = dict(params, objective=objective)
    for f in sorted(df.fold.unique()):
        tr, va = (df.fold != f).values, (df.fold == f).values
        m = lgb.train(p, lgb.Dataset(df.loc[tr, feats], df.loc[tr, target]), num_boost_round=rounds)
        out[va] = m.predict(df.loc[va, feats])
    return out


def sweep(df, gt, col, folds):
    best = None
    gfold = gt.filament_id.str.split("-", n=1).str[1].str.rsplit("_", n=1).str[0].map(folds)
    for c in CUTS:
        k = df[df[col] > c]
        pred = pd.DataFrame({"filament_id": k.stem + "_" + k.inst.astype(str), "segmentation_rle": k.rle})
        r = compute_pq(gt, pred)
        if best is None or r["pq"] > best["pq"]:
            pf = pred.filament_id.str.rsplit("_", n=1).str[0].map(folds)
            per = [compute_pq(gt[gfold == f], pred[pf == f])["pq"] for f in range(5)]
            best = dict(cut=c, **r, per_fold=[round(x, 4) for x in per])
    return best


def main():
    oof = Path(sys.argv[1])
    records = load_records()
    folds = get_folds().set_index("stem")["fold"]
    df = pd.read_pickle(oof / "candidates.pkl")
    df["fold"] = df.stem.map(folds)
    gt = gt_frame(records, sorted(records))
    # eşleşme ve eşleşmede IoU hedefleri (y = mean_a IoU_a·[IoU_a>0.5]); p için y>0 yaklaşımı
    df["matched"] = (df.y > 0).astype(int)
    # görüntü düzeyi bağlam öznitelikleri
    g = df.groupby("stem")
    df["img_mean_pmean"] = g.p_mean.transform("mean")
    df["img_mean_contrast"] = g.contrast.transform("mean")
    df["img_max_area"] = g.area.transform("max")
    df["rank_pmean"] = g.p_mean.rank(ascending=False, pct=True)
    df["rank_area"] = g.area.rank(ascending=False, pct=True)
    img_feats = FEATS + ["img_mean_pmean", "img_mean_contrast", "img_max_area", "rank_pmean", "rank_area"]

    res = {}
    df["s_base"] = nested(df, FEATS, "y", PARAMS, ROUNDS)
    res["base"] = sweep(df, gt, "s_base", folds)
    # engel (hurdle) ayrıştırması: E[y] = P(y>0) * E[y | y>0]
    p = nested(df, FEATS, "matched", PARAMS, ROUNDS, objective="binary")
    m = df.matched == 1
    q = np.full(len(df), np.nan)
    for f in sorted(df.fold.unique()):
        tr = ((df.fold != f) & m).values
        va = (df.fold == f).values
        mdl = lgb.train(PARAMS, lgb.Dataset(df.loc[tr, FEATS], df.loc[tr, "y"]), num_boost_round=ROUNDS)
        q[va] = mdl.predict(df.loc[va, FEATS])
    df["s_pq2"] = p * np.clip(q, 0, 1)
    res["pq2"] = sweep(df, gt, "s_pq2", folds)
    # hiper parametreler
    for name, upd, rounds in [("hp_leaves31", dict(num_leaves=31), ROUNDS),
                              ("hp_leaves7", dict(num_leaves=7), ROUNDS),
                              ("hp_rounds800_lr015", dict(learning_rate=0.015), 800),
                              ("hp_mindata100", dict(min_data_in_leaf=100), ROUNDS),
                              ("hp_mindata20", dict(min_data_in_leaf=20), ROUNDS)]:
        df["s_" + name] = nested(df, FEATS, "y", dict(PARAMS, **upd), rounds)
        res[name] = sweep(df, gt, "s_" + name, folds)
    df["s_imgfeat"] = nested(df, img_feats, "y", PARAMS, ROUNDS)
    res["imgfeat"] = sweep(df, gt, "s_imgfeat", folds)
    base = res["base"]
    rows = []
    for k, v in res.items():
        rows.append(dict(varyant=k, pq=v["pq"], fark=v["pq"] - base["pq"], cut=v["cut"], tp=v["tp"], fp=v["fp"],
                         fn=v["fn"], per_fold=v["per_fold"],
                         fold_fark=[round(a - b, 4) for a, b in zip(v["per_fold"], base["per_fold"])]))
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))
    out.to_csv(oof / "scorer_variants.csv", index=False)


if __name__ == "__main__":
    main()
