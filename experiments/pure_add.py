"""Deney 42: saf ekleme pilotu — ikinci bir modelin (ELR, deney 36) adaylarından, teslim maskeleriyle
hiç piksel paylaşmayanları öğrenilmiş skorla mevcut seçime ekle.

Dış fold f ∈ {0, 1} için:
  taban   : ens1536 teslim seçimi (0,5·LightGBM + 0,5·CNN ortalaması > 0,25), maskeler sabit
  havuz   : ELR OOF olasılığından aynı hatla (eşik 0,4, eyer 0,15, alan 100) adaylar; teslim maskeleriyle
            kesişimi sıfır olanlar
  skor    : f dışındaki fold'ların ens1536 adayları + diğer dış fold'un ELR adaylarıyla eğitilen CNN
  eşik    : diğer dış fold'da (kendi CNN'iyle) PQ'yu en yükselten eşik, f'ye uygulanır (çapraz seçim)
Rapor: tabana karşı PQ/TP/FP/FN farkı. Kabul ölçütü: iki fold'da da pozitif.
Örnek: PYTHONPATH=src python experiments/pure_add.py --src elr1536_effv2s
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pycocotools.mask as mu

from cnn_selector import crop_one, norm_img, predict_cnn, train_fold
from data import TRAIN_IMG, get_folds, gt_frame, load_records, poly_to_rle
from metric import compute_pq, mask_to_rle
from post import prob_to_instances
from tree_dp import y_of

OOF = Path("outputs/oof")
REC = load_records()


def baseline_selection(folds):
    o = OOF / "ens1536_tta2"
    d = [pd.read_csv(o / f"cnn_scores_s{s}.csv") for s in range(3)]
    df = d[0].copy()
    df["cnn"] = sum(x.cnn for x in d) / 3
    df = df.merge(pd.read_pickle(o / "candidates.pkl")[["stem", "inst", "rle"]], on=["stem", "inst"])
    df["sel"] = 0.5 * df.score + 0.5 * df.cnn > 0.25
    return df


def source_candidates(src, stems, folds, base):
    rows, crops = [], []
    for s in stems:
        prob_u8 = np.load(OOF / f"{src}_tta2" / f"fold{folds[s]}" / "val_probs" / f"{s}.npy")
        lab = prob_to_instances(prob_u8.astype(np.float32) / 255, thr=0.4, min_area=100, join=0, link_thr=0.15)
        if lab.max() == 0:
            continue
        sel = base[(base.stem == s) & base.sel]
        occ = np.zeros((2048, 2048), bool)
        for r in sel.rle:
            occ |= mu.decode({"size": [2048, 2048], "counts": r.encode()}).astype(bool)
        img = norm_img(TRAIN_IMG / f"{s}.jpeg")
        gts = [[poly_to_rle(a["seg"]) for a in an] for an in REC[s].values()]
        rles, cs = [], []
        for k in range(1, lab.max() + 1):
            m = lab == k
            if (m & occ).any():  # saf ekleme: teslim maskeleriyle sıfır piksel kesişimi
                continue
            rles.append(mask_to_rle(m))
            cs.append(crop_one(img, prob_u8, m))
        if rles:
            for r, v in zip(rles, y_of(rles, gts)):
                rows.append(dict(stem=s, fold=int(folds[s]), rle=r, y=v))
            crops += cs
    return pd.DataFrame(rows), (np.stack(crops) if crops else np.zeros((0, 3, 160, 160), np.uint8))


def pq_with(gt, base_f, add):
    sel = base_f[base_f.sel]
    rows = list(zip(sel.stem + "_b" + sel.inst.astype(str), sel.rle))
    rows += [(f"{s}_a{i}", r) for i, (s, r) in enumerate(zip(add.stem, add.rle))]
    return compute_pq(gt, pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="elr1536_effv2s")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1], help="değerlendirilecek dış fold'lar")
    ap.add_argument("--cut", type=float, default=None, help="önceden sabitlenmiş kesme (ön kayıt: 0,36)")
    args = ap.parse_args()
    folds = get_folds().set_index("stem")["fold"]
    base = baseline_selection(folds)
    base["fold"] = base.stem.map(folds)
    z = np.load(OOF / "ens1536_tta2" / "cnn_crops.npz")
    order = pd.DataFrame({"stem": z["stem"], "inst": z["inst"], "row": np.arange(len(z["stem"]))})
    ens = pd.read_pickle(OOF / "ens1536_tta2" / "candidates.pkl")[["stem", "inst", "y"]].merge(order, on=["stem", "inst"])
    ens["fold"] = ens.stem.map(folds)
    cand, C = {}, {}
    src_folds = sorted({int(folds[s]) for s in REC if (OOF / f"{args.src}_tta2" / f"fold{folds[s]}" / "val_probs" / f"{s}.npy").exists()})
    for f in src_folds:
        stems = sorted(s for s in REC if folds[s] == f)
        cand[f], C[f] = source_candidates(args.src, stems, folds, base)
        print(f"fold {f}: {len(cand[f])} saf ekleme adayı, ortalama y={cand[f].y.mean():.3f}", flush=True)
    for f in args.folds:  # dış fold f'nin CNN'i: f dışındaki ens adayları + f dışındaki kaynak adayları
        others = [g for g in src_folds if g != f]
        tr = ens[ens.fold != f]
        Xtr = np.concatenate([z["X"][tr.row.to_numpy()]] + [C[g] for g in others])
        ytr = np.concatenate([tr.y.to_numpy()] + [cand[g].y.to_numpy() for g in others])
        cand[f]["s"] = train_fold(Xtr, ytr, C[f], args.epochs, 42 + f)
    gts = {f: gt_frame(REC, sorted(s for s in REC if folds[s] == f)) for f in args.folds}
    cuts = [args.cut] if args.cut is not None else list(np.round(np.arange(0.20, 0.62, 0.04), 2))
    res = {}
    for f in args.folds:
        b = base[base.fold == f]
        r0 = pq_with(gts[f], b, cand[f].iloc[:0])
        res[f] = {c: pq_with(gts[f], b, cand[f][cand[f].s > c]) for c in cuts}
        print(f"fold {f} taban PQ {r0['pq']:.4f} TP {r0['tp']} FP {r0['fp']} FN {r0['fn']}", flush=True)
        for c in cuts:
            r = res[f][c]
            print(f"  kesme {c:.2f}: +{(cand[f].s > c).sum():3d} aday | PQ {r['pq']:.4f} ({r['pq'] - r0['pq']:+.4f}) "
                  f"TP {r['tp']} FP {r['fp']} FN {r['fn']}", flush=True)
        res[f]["taban"] = r0
    S = lambda r: r["pq"] * (r["tp"] + 0.5 * r["fp"] + 0.5 * r["fn"])  # eşleşen IoU toplamı
    D = lambda r: r["tp"] + 0.5 * r["fp"] + 0.5 * r["fn"]
    for c in cuts:  # birleşik PQ = ΣS / ΣD (fold PQ ortalaması değil)
        b0 = sum(S(res[f]["taban"]) for f in args.folds) / sum(D(res[f]["taban"]) for f in args.folds)
        b1 = sum(S(res[f][c]) for f in args.folds) / sum(D(res[f][c]) for f in args.folds)
        print(f"birleşik (fold {args.folds}) kesme {c:.2f}: taban {b0:.4f} → {b1:.4f} ({b1 - b0:+.4f})", flush=True)
    if len(cuts) == 1:
        block_bootstrap(base, cand, gts, args.folds, cuts[0])


def block_bootstrap(base, cand, gts, eval_folds, cut, n=3000, seed=20261029):
    """Eşleştirilmiş zaman blok bootstrap: dosya adındaki zamana göre sıralı görüntüler, aralarında
    24 saatten kısa boşluk olanlar aynı bloğa. Birleşik ΔPQ = ΣS/ΣD farkı."""
    from datetime import datetime
    rows = []
    for f in eval_folds:
        g = gts[f]
        gstem = g.filament_id.str.split("-", n=1).str[1].str.rsplit("_", n=1).str[0]
        for s in sorted(set(gstem)):
            b = base[(base.stem == s)]
            add = cand[f][(cand[f].stem == s) & (cand[f].s > cut)]
            r0, r1 = pq_with(g[gstem == s], b, add.iloc[:0]), pq_with(g[gstem == s], b, add)
            d0 = r0["tp"] + 0.5 * r0["fp"] + 0.5 * r0["fn"]
            d1 = r1["tp"] + 0.5 * r1["fp"] + 0.5 * r1["fn"]
            rows.append((s, r0["pq"] * d0, d0, r1["pq"] * d1, d1))
    t = sorted(rows, key=lambda r: r[0][:14])
    times = [datetime.strptime(r[0][:14], "%Y%m%d%H%M%S") for r in t]
    blk, k = [0], 0
    for a, b in zip(times, times[1:]):
        k += (b - a).total_seconds() >= 86400
        blk.append(k)
    M = np.array([r[1:] for r in t])
    blk = np.array(blk)
    ids = np.unique(blk)
    rng = np.random.default_rng(seed)
    point = M[:, 2].sum() / M[:, 3].sum() - M[:, 0].sum() / M[:, 1].sum()
    deltas = []
    for _ in range(n):
        pick = rng.choice(ids, len(ids))
        sub = np.concatenate([M[blk == i] for i in pick])
        deltas.append(sub[:, 2].sum() / sub[:, 3].sum() - sub[:, 0].sum() / sub[:, 1].sum())
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    print(f"zaman blok bootstrap ({len(ids)} blok, {n} tekrar): ΔPQ {point:+.4f}, %95 aralık [{lo:+.4f}, {hi:+.4f}]", flush=True)

if __name__ == "__main__":
    main()
