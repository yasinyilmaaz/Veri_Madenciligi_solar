"""Çok eşikli aday havuzu + aday seçici (iç içe CV) + çakışmasız açgözlü seçim.

Her görüntüde adaylar birden fazla eşikte üretilir (iç içe bileşenler). Seçici her adayın
beklenen PQ katkısını tahmin eder; görüntü başına skora göre sıralanıp çakışmayanlar alınır.
Böylece her filament için en uygun eşik (geniş / dar maske) ayrı ayrı seçilmiş olur.

Örnek: PYTHONPATH=src python experiments/scorer_multi.py outputs/oof/ens3_mean_tta2
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pycocotools.mask as mu

from data import TRAIN_IMG, get_folds, gt_frame, load_records, poly_to_rle, read_image
from metric import compute_pq, mask_to_rle
from post import prob_to_instances
from scorer import FEATS, PARAMS, ROUNDS, instance_features

THRS = [0.3, 0.4, 0.5, 0.6]
FEATS_M = FEATS + ["cand_thr"]


def process_image(args):
    path, gt_by_annot = args
    stem = path.stem
    prob = np.load(path).astype(np.float32) / 255
    img = read_image(TRAIN_IMG / f"{stem}.jpeg")
    rows = []
    for t in THRS:
        lab = prob_to_instances(prob, thr=t, min_area=100, join=8)
        feats = instance_features(prob, img, lab, t)
        for r in feats:
            r.update(stem=stem, cand_thr=t, n_inst_img=len(feats), station=ord(stem[-2]),
                     rle=mask_to_rle(lab == r["inst"]))
        rows += feats
    if rows:
        pred = [{"size": [2048, 2048], "counts": r["rle"].encode()} for r in rows]
        ys = np.zeros(len(rows))
        for gts in gt_by_annot.values():
            if gts:
                iou = np.asarray(mu.iou(pred, gts, [0] * len(gts))).reshape(len(rows), len(gts)).max(1)
                ys += np.where(iou > 0.5, iou, 0)
        for r, y in zip(rows, ys / len(gt_by_annot)):
            r["y"] = y
    return rows


def greedy_select(df, cut):
    """Görüntü başına: skora göre azalan sırada, önceki seçilenlerle çakışmayan adayları al."""
    keep = []
    for stem, g in df[df.score > cut].groupby("stem"):
        g = g.sort_values("score", ascending=False)
        occ = None
        for idx, r in g.iterrows():
            m = mu.decode({"size": [2048, 2048], "counts": r.rle.encode()}).astype(bool)
            if occ is not None and (m & occ).any():
                continue
            occ = m if occ is None else (occ | m)
            keep.append(idx)
    return df.loc[keep]


def _sel(args):
    g, cut = args
    return greedy_select(g, cut)


def main():
    oof = Path(sys.argv[1])
    records = load_records()
    folds = get_folds().set_index("stem")["fold"]
    cache = oof / "candidates_multi.pkl"
    if cache.exists():
        df = pd.read_pickle(cache)
    else:
        paths = sorted(oof.glob("fold*/val_probs/*.npy"))
        jobs = [(p, {a: [poly_to_rle(x["seg"]) for x in anns] for a, anns in records[p.stem].items()}) for p in paths]
        with ProcessPoolExecutor(16) as ex:
            df = pd.DataFrame([r for img in ex.map(process_image, jobs, chunksize=2) for r in img])
        df.to_pickle(cache)
    df["fold"] = df.stem.map(folds)
    print(f"{len(df)} aday ({df.cand_thr.value_counts().sort_index().to_dict()})", flush=True)
    df["score"] = np.nan
    for f in sorted(df.fold.unique()):
        tr, va = df.fold != f, df.fold == f
        m = lgb.train(PARAMS, lgb.Dataset(df.loc[tr, FEATS_M], df.loc[tr, "y"]), num_boost_round=ROUNDS)
        df.loc[va, "score"] = m.predict(df.loc[va, FEATS_M])
    gt = gt_frame(records, sorted(records))
    res = []
    with ProcessPoolExecutor(16) as ex:
        for cut in [0.16, 0.20, 0.24, 0.28]:
            parts = list(ex.map(_sel, [(g, cut) for _, g in df.groupby("stem")], chunksize=8))
            k = pd.concat(parts)
            pred = pd.DataFrame({"filament_id": k.stem + "_" + k.cand_thr.astype(str) + "x" + k.inst.astype(str),
                                 "segmentation_rle": k.rle})
            r = compute_pq(gt, pred)
            res.append(dict(cut=cut, **r, n=len(k), thr_dist=k.cand_thr.value_counts().sort_index().to_dict()))
            print(res[-1], flush=True)
    pd.DataFrame(res).to_csv(oof / "scorer_multi_cut.csv", index=False)
    final = lgb.train(PARAMS, lgb.Dataset(df[FEATS_M], df["y"]), num_boost_round=ROUNDS)
    final.save_model(str(oof / "scorer_multi.txt"))


if __name__ == "__main__":
    main()
