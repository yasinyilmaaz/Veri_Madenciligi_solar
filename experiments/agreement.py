"""Deney 15: uyum-kalibreli yumuşak hedef (mean_cal).

Tek anotatörlü görüntülerde her GT poligonunun hedefi 1 yerine t = (1 + (k-1)·ĝ) / k olur;
ĝ = "başka bir anotatör bu filamenti IoU>0,5 ile çizer" olasılığı. ĝ, fold f'nin EĞİTİM görüntülerinden
çok anotatörlü olanlarda öğrenilir (öznitelik: geometri + fold f öğretmen olasılıkları + kontrast), yani
fold f validasyonu hiçbir aşamada görülmez. Çok anotatörlü görüntülerde hedef = anotatör ortalaması.
Çıktı outputs/teacher/meancal_k{k}/fold{f}/{stem}.npy; train.py --teacher_dir ... --teacher_alpha 0 ile kullanılır.
Örnek: PYTHONPATH=src python experiments/agreement.py --folds 0 1 --k 2
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
import pycocotools.mask as mu
from sklearn.metrics import roc_auc_score

from data import OUT, TRAIN_IMG, get_folds, load_records, poly_to_mask, poly_to_rle, read_image

FEATS = ["area", "logA", "bw", "bh", "fill", "r", "n_self", "station",
         "p_mean", "p_max", "p_q25", "p_frac", "contrast", "i_mean"]
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
              feature_fraction=0.8, verbose=-1, num_threads=8)
REC = load_records()


def poly_rows(args):
    """Bir görüntünün tüm GT poligonları için öznitelikler (+ çok anotatörlüyse uyum hedefi)."""
    s, teacher = args
    rec = REC[s]
    p = np.load(teacher / f"{s}.npy").astype(np.float32) / 255
    img = read_image(TRAIN_IMG / f"{s}.jpeg").astype(np.float32)
    R = {ai: [poly_to_rle(a["seg"]) for a in an] for ai, an in rec.items()}
    rows = []
    for ai, an in rec.items():
        for k, a in enumerate(an):
            g = R[ai][k]
            others = [R[b] for b in R if b != ai]
            agree = np.mean([float(np.asarray(mu.iou([g], o, [0] * len(o))).max() > .5) if o else 0.0
                             for o in others]) if others else np.nan
            m = poly_to_mask(a["seg"]).astype(bool)
            ring = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~m
            pv = p[m] if m.any() else np.zeros(1)
            bb, area = mu.toBbox(g), mu.area(g)
            cy, cx = bb[1] + bb[3] / 2, bb[0] + bb[2] / 2
            rows.append(dict(stem=s, ai=ai, k=k, agree=agree, area=area, logA=np.log1p(area), bw=bb[2], bh=bb[3],
                             fill=area / (bb[2] * bb[3] + 1), r=np.hypot(cy - 1024, cx - 1024) / 905,
                             n_self=len(an), station=ord(s[-2]), p_mean=pv.mean(), p_max=pv.max(),
                             p_q25=np.quantile(pv, .25), p_frac=(pv > .4).mean(),
                             contrast=img[ring].mean() - img[m].mean() if m.any() and ring.any() else 0.0,
                             i_mean=img[m].mean() if m.any() else 0.0))
    return rows


def save_target(args):
    s, gs, out = args
    rec = REC[s]
    if len(rec) > 1:
        masks = []
        for an in rec.values():
            m = np.zeros((2048, 2048), bool)
            for a in an:
                m |= poly_to_mask(a["seg"])
            masks.append(m)
        tgt = np.mean(masks, 0)
    else:
        tgt = np.zeros((2048, 2048), np.float32)
        for a, t in zip(next(iter(rec.values())), gs):
            m = poly_to_mask(a["seg"]).astype(bool)
            tgt[m] = np.maximum(tgt[m], t)
    np.save(out / f"{s}.npy", np.round(tgt * 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--k", type=float, default=2.0, help="t = (1 + (k-1)·ĝ) / k")
    ap.add_argument("--teacher", default="ens4", help="outputs/teacher/<ad>/foldN (make_teacher.py)")
    args = ap.parse_args()
    folds = get_folds()
    for f in args.folds:
        teacher = OUT / "teacher" / args.teacher / f"fold{f}"
        tr = folds.loc[folds.fold != f, "stem"].tolist()
        with ProcessPoolExecutor(12) as ex:
            D = pd.DataFrame([r for l in ex.map(poly_rows, [(s, teacher) for s in tr], chunksize=4) for r in l])
        M = D[D.agree.notna()].copy()
        # ĝ modelinin kalitesi: eğitim görüntüleri içinde görüntü gruplu 5'li iç CV
        grp = M.stem.map({s: i % 5 for i, s in enumerate(sorted(set(M.stem)))})
        M["g"] = np.nan
        for i in range(5):
            M.loc[grp == i, "g"] = lgb.train(PARAMS, lgb.Dataset(M.loc[grp != i, FEATS], M.loc[grp != i, "agree"]),
                                             300).predict(M.loc[grp == i, FEATS])
        model = lgb.train(PARAMS, lgb.Dataset(M[FEATS], M.agree), 300)
        S = D[D.agree.isna()].copy()
        S["g"] = np.clip(model.predict(S[FEATS]), 0, 1)
        S["t"] = (1 + (args.k - 1) * S.g) / args.k
        out = OUT / "teacher" / f"meancal_k{args.k:g}" / f"fold{f}"
        out.mkdir(parents=True, exist_ok=True)
        gs = S.sort_values(["stem", "k"]).groupby("stem").t.apply(list).to_dict()
        with ProcessPoolExecutor(12) as ex:
            list(ex.map(save_target, [(s, gs.get(s, []), out) for s in tr], chunksize=8))
        print(f"fold{f}: çok anotatörlü {len(M)} poligon, iç CV AUC {roc_auc_score(M.agree > .5, M.g):.3f}; "
              f"tek anotatörlü {len(S)} poligon, t ortalama {S.t.mean():.3f}, q10-q90 {np.quantile(S.t, [.1, .9]).round(2)}",
              flush=True)


if __name__ == "__main__":
    main()
