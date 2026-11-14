"""Veri seti denge ve genelleme denetimi.

1. Train ve test dağılımı: istasyon, yıl (güneş döngüsü).
2. Train içi denge: filament boyutu, diskteki konum, görüntü başına filament, anotatör sayısı.
3. Model zayıflığı: grup bazında OOF PQ ve boyut/konum bazında recall (en iyi sistem: ens3 + seçici).
Örnek: PYTHONPATH=src python experiments/data_audit.py outputs/oof/ens3_mean_tta2 0.22
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pycocotools.mask as mu

from data import TEST_IMG, get_folds, gt_frame, load_records, poly_to_rle
from metric import compute_pq

CENTER, RADIUS = 1024, 870  # train ortalaması (quality.py: yarıçap medyanı ~868)


def stem_info(stem):
    return dict(station=stem[-2], year=int(stem[:4]))


def main():
    oof = Path(sys.argv[1])
    cut = float(sys.argv[2])
    records = load_records()
    stems = sorted(records)

    # 1. train ve test dağılımı
    tr = pd.DataFrame([stem_info(s) for s in stems])
    te = pd.DataFrame([stem_info(p.stem) for p in TEST_IMG.glob("*.jpeg")])
    for col in ["station", "year"]:
        t = pd.concat([tr[col].value_counts(normalize=True).rename("train"),
                       te[col].value_counts(normalize=True).rename("test")], axis=1).fillna(0).sort_index()
        print(f"\n== {col} dağılımı (oran)\n", (t * 100).round(1).to_string())

    # 2. filament düzeyinde denge (anotatör-filament)
    fil = []
    for s in stems:
        for ai, anns in records[s].items():
            for k, a in enumerate(anns, 1):
                rle = poly_to_rle(a["seg"])
                x, y, w, h = mu.toBbox(rle)
                r = np.hypot(x + w / 2 - CENTER, y + h / 2 - CENTER) / RADIUS
                fil.append(dict(stem=s, ai=ai, fid=f"{ai}_{k}", area=float(mu.area(rle)), r=r,
                                n_annot=len(records[s]), **stem_info(s)))
    fil = pd.DataFrame(fil)
    fil["size"] = pd.cut(fil.area, [0, 300, 1000, 3000, 1e9], labels=["<300", "300-1k", "1k-3k", ">3k"])
    fil["pos"] = pd.cut(fil.r, [0, 0.5, 0.8, 0.95, 2], labels=["merkez<0.5", "0.5-0.8", "0.8-0.95", "limb>0.95"])
    print("\n== filament boyutu (%)\n", (fil["size"].value_counts(normalize=True).sort_index() * 100).round(1).to_string())
    print("\n== diskteki konum r/R (%)\n", (fil["pos"].value_counts(normalize=True).sort_index() * 100).round(1).to_string())
    per_img = fil.groupby("ai").size()
    print("\n== anotatör-görüntü başına filament: medyan", per_img.median(), "| p10/p90", per_img.quantile([.1, .9]).tolist())

    # 3. model zayıflığı: seçici sonrası tahminler
    cand = pd.read_csv(oof / "candidates_scored.csv")
    # rle'ler pickle'da
    rle = pd.read_pickle(oof / "candidates.pkl")[["stem", "inst", "rle"]]
    cand = cand.merge(rle, on=["stem", "inst"])
    keep = cand[cand.score > cut]
    pred = pd.DataFrame({"filament_id": keep.stem + "_" + keep.inst.astype(str), "segmentation_rle": keep.rle})
    gt = gt_frame(records, stems)
    gt_stem = gt.filament_id.str.split("-", n=1).str[1].str.rsplit("_", n=1).str[0]
    pred_stem = pred.filament_id.str.rsplit("_", n=1).str[0]

    print("\n== grup bazında OOF PQ")
    for name, key in [("istasyon", lambda s: s[-2]), ("yıl", lambda s: s[:4]),
                      ("anotatör sayısı", lambda s: str(len(records[s])))]:
        g_gt, g_pr = gt_stem.map(key), pred_stem.map(key)
        rows = []
        for v in sorted(set(g_gt)):
            r = compute_pq(gt[g_gt == v], pred[g_pr == v])
            rows.append(dict(grup=v, n_img=sum(key(s) == v for s in stems), pq=r["pq"], sq=r["sq"], rq=r["rq"]))
        print(f"-- {name}\n", pd.DataFrame(rows).round(3).to_string(index=False))

    # boyut ve konum bazında recall (GT filament eşleşti mi)
    pred_by = defaultdict(list)
    for s, c in zip(pred_stem, pred.segmentation_rle):
        pred_by[s].append({"size": [2048, 2048], "counts": c.encode()})
    matched = []
    for (s, ai), g in fil.groupby(["stem", "ai"]):
        gts = [poly_to_rle(a["seg"]) for a in records[s][ai]]
        p = pred_by.get(s, [])
        best = np.asarray(mu.iou(gts, p, [0] * len(p))).reshape(len(gts), len(p)).max(1) if p else np.zeros(len(gts))
        matched += list(best > 0.5)
    fil["matched"] = matched
    for col in ["size", "pos"]:
        t = fil.groupby(col, observed=True).agg(n=("matched", "size"), recall=("matched", "mean"))
        print(f"\n== {col} bazında recall\n", t.round(3).to_string())
    fil.to_csv(oof / "audit_filaments.csv", index=False)


if __name__ == "__main__":
    main()
