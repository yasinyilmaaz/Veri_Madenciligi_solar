"""Öğrenilmiş aday seçici (LightGBM), iç içe CV ile.

1. OOF olasılık haritalarından gevşek ayarla aday instance'lar üretilir.
2. Her aday için şekil/olasılık/kontrast öznitelikleri çıkarılır.
3. Hedef: adayın anotatörler üzerinden ortalama PQ katkısı
   y = mean_a [IoU_a if IoU_a > 0.5 else 0]
4. Fold f'nin adayları, yalnızca diğer fold'larla eğitilmiş modelle skorlanır.
5. Tek parametre (kesme eşiği c) havuzlanmış OOF'ta seçilir: y_hat > c ise gönder.
   Teori: p*q > PQ/2 ise göndermek beklenen PQ'yu artırır (c ~ 0.2).

Örnek: python src/scorer.py outputs/oof/tgt_mean_tta2
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
import pycocotools.mask as mu

from artifacts import probs_sha1, write_scored_link
from data import TRAIN_IMG, get_folds, load_records, poly_to_rle, read_image
from metric import compute_pq, mask_to_rle
from post import prob_to_instances

CAND = dict(thr=0.4, min_area=100, join=8, thr_high=None)  # gevşek: seçimi model yapacak
FEATS = ["area", "log_area", "p_mean", "p_max", "p_q25", "p_q75", "f_09", "f_07", "bw", "bh",
         "rect_long", "rect_short", "elong", "solidity", "radius", "contrast", "i_in", "i_ring",
         "n_frag", "n_inst_img"]  # dosya adındaki istasyon kullanılmaz: çıkarımda yalnız görüntünün kendisi (düzenleyici, forum 29.10)


def disk_geometry(img):
    d = img > 0.6 * np.median(img[img > 30])
    ys, xs = np.nonzero(d[::4, ::4])
    return ys.mean() * 4, xs.mean() * 4, np.sqrt(d.sum() / np.pi)


def instance_features(prob, img, lab, thr):
    cy, cx, R = disk_geometry(img)
    frag_lab = cv2.connectedComponents((prob > thr).astype(np.uint8), connectivity=8)[1]
    ids = np.unique(lab[lab > 0])
    k = np.ones((15, 15), np.uint8)
    rows = []
    for i in ids:
        m = lab == i
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        pad = 10
        Y0, Y1, X0, X1 = max(y0 - pad, 0), min(y1 + pad, 2048), max(x0 - pad, 0), min(x1 + pad, 2048)
        mc = m[Y0:Y1, X0:X1].astype(np.uint8)
        pv = prob[m]
        cnts, _ = cv2.findContours(mc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts = np.concatenate(cnts)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        hull = cv2.contourArea(cv2.convexHull(pts)) + 1
        ring = (cv2.dilate(mc, k) > 0) & (mc == 0)
        ic = img[Y0:Y1, X0:X1].astype(np.float32)
        i_in, i_ring = ic[mc > 0].mean(), (ic[ring].mean() if ring.any() else ic.mean())
        rows.append(dict(
            inst=int(i), area=int(m.sum()), log_area=np.log1p(m.sum()), p_mean=pv.mean(), p_max=pv.max(),
            p_q25=np.quantile(pv, 0.25), p_q75=np.quantile(pv, 0.75), f_09=(pv > 0.9).mean(),
            f_07=(pv > 0.7).mean(), bw=x1 - x0, bh=y1 - y0, rect_long=max(rw, rh), rect_short=min(rw, rh),
            elong=max(rw, rh) / (min(rw, rh) + 1), solidity=m.sum() / hull,
            radius=np.hypot(ys.mean() - cy, xs.mean() - cx) / R, contrast=i_in / (i_ring + 1e-6),
            i_in=i_in, i_ring=i_ring, n_frag=len(np.unique(frag_lab[m & (frag_lab > 0)])),
        ))
    return rows


def process_image(args):
    path, gt_by_annot = args
    stem = path.stem
    prob = np.load(path).astype(np.float32) / 255
    img = read_image(TRAIN_IMG / f"{stem}.jpeg")
    lab = prob_to_instances(prob, **CAND)
    rows = instance_features(prob, img, lab, CAND["thr"])
    for r in rows:
        r["stem"] = stem
        r["n_inst_img"] = len(rows)
        r["rle"] = mask_to_rle(lab == r["inst"])
    if rows:
        pred = [{"size": [2048, 2048], "counts": r["rle"].encode()} for r in rows]
        ys = np.zeros(len(rows))
        for gts in gt_by_annot.values():
            if not gts:
                continue
            iou = np.asarray(mu.iou(pred, gts, [0] * len(gts))).reshape(len(rows), len(gts)).max(1)
            ys += np.where(iou > 0.5, iou, 0)
        for r, y in zip(rows, ys / len(gt_by_annot)):
            r["y"] = y
    return rows


def build_candidates(oof_dir, records):
    paths = sorted(Path(oof_dir).glob("fold*/val_probs/*.npy"))
    jobs = [(p, {a: [poly_to_rle(x["seg"]) for x in anns] for a, anns in records[p.stem].items()}) for p in paths]
    with ProcessPoolExecutor(16) as ex:
        rows = [r for img in ex.map(process_image, jobs, chunksize=4) for r in img]
    return pd.DataFrame(rows)


MEMBER_FEATS = ["pm_min", "pm_max", "pm_std"]


def member_stats(lab, member_probs):
    """Her instance için ensemble üyelerinin ortalama olasılığı -> min/max/std (üye sayısından bağımsız)."""
    n = lab.max() + 1
    area = np.bincount(lab.ravel(), minlength=n) + 1e-6
    means = np.stack([np.bincount(lab.ravel(), weights=m.ravel(), minlength=n) / area for m in member_probs])
    return {"pm_min": means.min(0), "pm_max": means.max(0), "pm_std": means.std(0)}


def _member_job(args):
    path, member_dirs = args
    prob = np.load(path).astype(np.float32) / 255
    lab = prob_to_instances(prob, **CAND)
    members = [np.load(Path(d) / path.parent.parent.name / "val_probs" / path.name).astype(np.float32) / 255
               for d in member_dirs]
    st = member_stats(lab, members)
    ids = np.unique(lab[lab > 0])
    return pd.DataFrame({"stem": path.stem, "inst": ids, **{k: v[ids] for k, v in st.items()}})


def add_member_feats(df, oof_dir, member_dirs):
    paths = sorted(Path(oof_dir).glob("fold*/val_probs/*.npy"))
    with ProcessPoolExecutor(16) as ex:
        mf = pd.concat(list(ex.map(_member_job, [(p, member_dirs) for p in paths], chunksize=4)))
    return df.merge(mf, on=["stem", "inst"], how="left")


PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1)
ROUNDS = 400


def cache_manifest(oof_dir):
    """Aday önbelleğinin geçerlilik anahtarı: aday ayarı + olasılık dosyalarının içerik özeti."""
    return {"cand": {k: CAND[k] for k in sorted(CAND)}, "probs_sha1": probs_sha1(oof_dir)}


def load_candidates(oof_dir, records):
    """candidates.pkl yalnız aynı ayar ve aynı olasılık dosyalarıyla üretildiyse okunur; yoksa üretilir.
    Uyuşmazlıkta durur: ayar değiştirip aynı klasörü kullanmak eski adaylarla sessizce deney yapardı
    """
    cache, meta = oof_dir / "candidates.pkl", oof_dir / "candidates_meta.json"
    man = cache_manifest(oof_dir)
    if cache.exists():
        if not meta.exists():
            raise SystemExit(f"{cache} için {meta.name} yok; üretim ayarı doğrulanamıyor. Dosyayı silip yeniden üret.")
        old = json.loads(meta.read_text())
        if old != man:
            raise SystemExit(f"{cache} farklı ayar/olasılıkla üretilmiş:\n  önbellek {old}\n  istenen  {man}\n"
                             "Başka bir OOF klasörü kullan ya da önbelleği silip yeniden üret.")
        return pd.read_pickle(cache)
    df = build_candidates(oof_dir, records)
    df.to_pickle(cache)
    meta.write_text(json.dumps(man, indent=1))
    return df


def nested_oof_scores(df, folds, feats):
    df = df.assign(fold=df.stem.map(folds))
    df["score"] = np.nan
    for f in sorted(df.fold.unique()):
        tr, va = df.fold != f, df.fold == f
        m = lgb.train(PARAMS, lgb.Dataset(df.loc[tr, feats], df.loc[tr, "y"]), num_boost_round=ROUNDS)
        df.loc[va, "score"] = m.predict(df.loc[va, feats])
    return df


class PQModel:
    """Engel (hurdle) seçici: skor = P(eşleşme) · E[y | eşleşme] (deney 04)."""

    def __init__(self, p_path, q_path):
        self.p = lgb.Booster(model_file=str(p_path))
        self.q = lgb.Booster(model_file=str(q_path))

    def predict(self, X):
        return self.p.predict(X) * np.clip(self.q.predict(X), 0, 1)


def train_pq_final(oof_dir):
    """Tüm OOF adaylarıyla p ve q modellerini eğitip kaydeder (test zamanı için)."""
    df = pd.read_pickle(Path(oof_dir) / "candidates.pkl")
    matched = (df.y > 0).astype(int)
    p = lgb.train(dict(PARAMS, objective="binary"), lgb.Dataset(df[FEATS], matched), num_boost_round=ROUNDS)
    q = lgb.train(PARAMS, lgb.Dataset(df.loc[matched == 1, FEATS], df.loc[matched == 1, "y"]),
                  num_boost_round=ROUNDS)
    p.save_model(str(Path(oof_dir) / "scorer_p.txt"))
    q.save_model(str(Path(oof_dir) / "scorer_q.txt"))


def select_instances(prob, img, stem, model, cut, member_probs=None, return_scores=False):
    """Test zamanı: aday üret, skorla, eşiği geçenleri yeniden numaralandır.

    return_scores=True ise {yeni_id: skor} de döner (refiner için).
    """
    lab = prob_to_instances(prob, **CAND)
    rows = instance_features(prob, img, lab, CAND["thr"])
    if not rows:
        return (lab, {}) if return_scores else lab
    X = pd.DataFrame(rows)
    X["n_inst_img"] = len(rows)
    feats = list(FEATS)
    if member_probs is not None:
        st = member_stats(lab, member_probs)
        for k, v in st.items():
            X[k] = v[X["inst"].to_numpy()]
        feats += MEMBER_FEATS
    sc = model.predict(X[feats])
    keep = sc > cut
    keep_ids = X.loc[keep, "inst"].to_numpy()
    remap = np.zeros(lab.max() + 1, np.int32)
    remap[keep_ids] = np.arange(1, len(keep_ids) + 1)
    if return_scores:
        return remap[lab], {int(i): float(v) for i, v in zip(range(1, len(keep_ids) + 1), sc[keep])}
    return remap[lab]


def pq_at(df, gt, col, cut):
    keep = df[df[col] > cut]
    pred = pd.DataFrame({"filament_id": keep.stem + "_" + keep.inst.astype(str), "segmentation_rle": keep.rle})
    return compute_pq(gt, pred)


def main():
    from data import gt_frame

    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("oof_dir")
    ap.add_argument("--members", nargs="*", default=[], help="üye modellerin OOF klasörleri (uyum öznitelikleri)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--link_thr", type=float, default=None, help="eyer bağlantısı (deney 16): join yerine prob>t bileşenleri")
    args = ap.parse_args()
    if args.link_thr is not None:
        CAND.update(join=0, link_thr=args.link_thr)
    oof_dir = Path(args.oof_dir)
    records = load_records()
    folds = get_folds().set_index("stem")["fold"]
    df = load_candidates(oof_dir, records)
    feats = list(FEATS)
    if args.members:
        df = add_member_feats(df, oof_dir, args.members)
        feats += MEMBER_FEATS
    print(f"{len(df)} aday, {df.stem.nunique()} görüntü, ortalama y={df.y.mean():.3f}, {len(feats)} öznitelik")
    # değerlendirme evreni: adayı olmayan görüntülerin GT'si de FN sayılmalı
    gt = gt_frame(records, sorted(set(df.stem) | {p.stem for p in oof_dir.glob("fold*/val_probs/*.npy") if p.stem in records}))
    df = nested_oof_scores(df, folds, feats)
    res = []
    for c in np.arange(0.10, 0.36, 0.02):
        r = pq_at(df, gt, "score", c)
        per_fold = [pq_at(df[df.fold == f], gt[gt.filament_id.str.split("-", n=1).str[1].str.rsplit("_", n=1).str[0].map(folds) == f],
                          "score", c)["pq"] for f in sorted(df.fold.unique())]
        res.append(dict(cut=round(c, 2), **r, fold_min=min(per_fold), fold_std=np.std(per_fold)))
    out = pd.DataFrame(res)
    print(out.round(4).to_string(index=False))
    out.to_csv(oof_dir / f"scorer_cut{args.tag}.csv", index=False)
    # test için: tüm OOF adaylarıyla final model
    final = lgb.train(PARAMS, lgb.Dataset(df[feats], df["y"]), num_boost_round=ROUNDS)
    final.save_model(str(oof_dir / f"scorer{args.tag}.txt"))
    print("final model:", oof_dir / f"scorer{args.tag}.txt", "| en iyi kesme:", out.loc[out.pq.idxmax(), "cut"])
    df.drop(columns=["rle"]).to_csv(oof_dir / f"candidates_scored{args.tag}.csv", index=False)
    write_scored_link(oof_dir, args.tag)


if __name__ == "__main__":
    main()
