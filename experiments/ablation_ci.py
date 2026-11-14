"""Ablasyon tablosu için belirsizlik: fold'lar arası std ve gün bloklu bootstrap %95 aralıkları.

Her adım aynı değerlendiriciyle (metric.compute_pq'nun anotatör-görüntü sayımı, tüm GT) yeniden hesaplanır.
Bootstrap birimi gözlem günüdür (stem'in ilk 8 hanesi): aynı günün yakın kareleri birlikte örneklenir.
Δ aralıkları eşleştirilmiştir (her tekrarda iki adım aynı günlerle hesaplanır).
Not: seçici kesmeleri aynı OOF üzerinde seçildi; aralıklar bu seçim iyimserliğini içermez.

Örnek: PYTHONPATH=src python experiments/ablation_ci.py
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pycocotools.mask as mask_utils

from artifacts import load_candidates_scored, mean_cnn, merge_exact
from data import OUT, gt_frame, load_records
from metric import _enc, _group, compute_pq, instances_to_rows
from post import prob_to_instances

# (etiket, OOF klasörü, el ayarlı son işlem ya da None, seçici: (skor, kesme) ya da None)
STEPS = [
    ("1 U-Net ResNet34, union target", "unet_r34_tta2", dict(thr=0.6, min_area=400, join=8, thr_high=0.9), None),
    ("2 mean-of-annotators target", "tgt_mean_tta2", dict(thr=0.5, min_area=400, join=8, thr_high=0.9), None),
    ("3 3-encoder ensemble", "ens3_mean_tta2", dict(thr=0.4, min_area=300, join=8, thr_high=0.9), None),
    ("4 + LightGBM selector", "ens3_mean_tta2", None, ("lgb", 0.22)),
    ("5 4-model ensemble + selector", "ens4_tta2", None, ("lgb", 0.24)),
    ("6 saddle linking", "ens4sad15_tta2", None, ("lgb", 0.22)),
    ("7 1536 crops, EffNetV2-S + M", "ens1536_tta2", None, ("lgb", 0.24)),
    ("8 + CNN selector (final)", "ens1536_tta2", None, ("mix", 0.25)),
]
LEGACY = {"ens4_tta2", "ens4sad15_tta2"}  # manifestten önceki önbellekler: anahtarla birleştirilir, günlük skoruyla denetlenir


def _rows(args):
    path, cfg = args
    prob = np.load(path).astype(np.float32) / 255
    return instances_to_rows(path.stem, prob_to_instances(prob, **cfg))


def predictions(run, post, sel):
    oof = OUT / "oof" / run
    if post is not None:
        paths = sorted(oof.glob("fold*/val_probs/*.npy"))
        with ProcessPoolExecutor(16) as ex:
            rows = [r for img in ex.map(_rows, [(p, post) for p in paths], chunksize=4) for r in img]
        return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    score, cut = sel
    if run in LEGACY:
        c = merge_exact(pd.read_pickle(oof / "candidates.pkl"),
                        pd.read_csv(oof / "candidates_scored.csv", usecols=["stem", "inst", "score"]),
                        ["stem", "inst"], "aday", "skor")
    else:
        c = load_candidates_scored(oof)
    if score == "mix":
        c = merge_exact(c, mean_cnn(oof), ["stem", "inst"], "aday", "CNN skoru")
        c["score"] = 0.5 * c.score + 0.5 * c.cnn
    k = c[c.score > cut]
    return pd.DataFrame({"filament_id": k.stem + "_" + k.inst.astype(str), "segmentation_rle": k.rle})


def per_key(gt_df, pred_df):
    """Anotatör-görüntü başına (IoU toplamı, TP, FP, FN); compute_pq ile aynı sayım."""
    gt, pred = _group(gt_df), _group(pred_df)
    out = []
    for ai, g in gt.items():
        stem = ai.split("-", 1)[1]
        p = pred.get(stem, [])
        if not p:
            out.append((stem, 0.0, 0, 0, len(g)))
            continue
        iou = np.asarray(mask_utils.iou(_enc(g), _enc(p), [0] * len(p))).reshape(len(g), len(p))
        hit = iou > 0.5
        out.append((stem, iou[hit].sum(), int(hit.sum()), int((hit.sum(0) == 0).sum()), int((hit.sum(1) == 0).sum())))
    df = pd.DataFrame(out, columns=["stem", "s", "tp", "fp", "fn"])
    df["d"] = df.tp + 0.5 * df.fp + 0.5 * df.fn
    return df


def main(n_boot=2000, seed=0):
    records = load_records()
    gt = gt_frame(records, sorted(records))
    fold = {p.stem: int(p.parent.parent.name[4:]) for p in (OUT / "oof" / "ens1536_tta2").glob("fold*/val_probs/*.npy")}
    S, D, res = [], [], []
    for label, run, post, sel in STEPS:
        pred = predictions(run, post, sel)
        k = per_key(gt, pred)
        pq = compute_pq(gt, pred)["pq"]
        assert abs(k.s.sum() / k.d.sum() - pq) < 1e-9
        k["fold"], k["day"] = k.stem.map(fold), k.stem.str[:8]
        g = k.groupby("fold")[["s", "d"]].sum()
        folds = g.s / g.d
        b = k.groupby("day")[["s", "d"]].sum()
        S.append(b.s)
        D.append(b.d)
        res.append(dict(step=label, pq=pq, fold_mean=folds.mean(), fold_std=folds.std(ddof=1),
                        fold_min=folds.min(), fold_max=folds.max()))
        print(f"{label}: PQ {pq:.4f}, fold {folds.mean():.4f} ± {folds.std(ddof=1):.4f}", flush=True)
    days = S[0].index
    S = np.stack([s.reindex(days).to_numpy() for s in S])
    D = np.stack([d.reindex(days).to_numpy() for d in D])
    w = np.random.default_rng(seed).multinomial(len(days), np.full(len(days), 1 / len(days)), size=n_boot)
    boot = (w @ S.T) / (w @ D.T)  # n_boot x adım
    for i, r in enumerate(res):
        r["ci_lo"], r["ci_hi"] = np.percentile(boot[:, i], [2.5, 97.5])
        if i:
            diff = boot[:, i] - boot[:, i - 1]
            r["delta"] = r["pq"] - res[i - 1]["pq"]
            r["delta_lo"], r["delta_hi"] = np.percentile(diff, [2.5, 97.5])
            r["delta_p_le0"] = (diff <= 0).mean()
    df = pd.DataFrame(res)
    print(f"\n{len(days)} gün bloğu, {n_boot} tekrar")
    print(df.round(4).to_string(index=False))
    df.to_csv(OUT / "ablation_ci.csv", index=False)


if __name__ == "__main__":
    main()
