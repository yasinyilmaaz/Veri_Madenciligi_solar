"""Aday havuzu kapsamı: GT anotasyon kayıtlarından kaçının havuzda IoU>0,5 adayı var.

Seçici sonrası PQ'dan bağımsız olarak piksel modelinin aday üretimini ölçer. Hat mevcut sistemle aynı
(eşik 0,4, eyer bağlantısı 0,15, en küçük alan 100). Kaçan GT'ler iki sınıfa ayrılır:
  sinyal_yok : GT'nin en fazla yarısı p>0,4 (yeniden gruplama yetmez, piksel desteği gerekir)
  diğer      : GT'nin yarısından fazlası p>0,4 ama havuzda IoU>0,5 aday yok (gruplama/kesme sorunu)
Örnek: PYTHONPATH=src python experiments/pool_coverage.py crop1536_effv2s dml1536_effv2s --folds 0 1
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pycocotools.mask as mu

from data import get_folds, load_records, poly_to_rle
from metric import mask_to_rle
from post import prob_to_instances

REC = load_records()


def consensus(stem):
    """Her GT kaydı için uzlaşı grubu: tek (tek anotatörlü görüntü), tümü / bazıları / hiçbiri (diğer
    anotatörlerin IoU>0,5 eşleşmesi)."""
    rec = list(REC[stem].values())
    rles = [[poly_to_rle(a["seg"]) for a in anns] for anns in rec]
    out = []
    for i, gi in enumerate(rles):
        for g in gi:
            if len(rles) == 1:
                out.append("tek")
                continue
            hits = sum(1 for j, gj in enumerate(rles) if j != i and gj and np.max(mu.iou([g], gj, [0] * len(gj))) > 0.5)
            out.append("tümü" if hits == len(rles) - 1 else "bazıları" if hits else "hiçbiri")
    return out


def _image(args):
    stem, path = args
    prob = np.load(path).astype(np.float32) / 255
    lab = prob_to_instances(prob, thr=0.4, min_area=100, join=0, link_thr=0.15)
    P = [{"size": [2048, 2048], "counts": mask_to_rle(lab == k).encode()} for k in range(1, lab.max() + 1)]
    groups = consensus(stem)
    res, k = [], 0
    for anns in REC[stem].values():
        for a in anns:
            G = poly_to_rle(a["seg"])
            best = np.max(mu.iou([G], P, [0] * len(P))) if P else 0.0
            if best > 0.5:
                kind = "eşleşen"
            else:
                m = mu.decode(G).astype(bool)
                kind = "sinyal_yok" if (prob[m] > 0.4).mean() <= 0.5 else "diğer"
            res.append((stem, k, kind, groups[k]))
            k += 1
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--ref", default=None, help="referans run: kayıt bazında kazanılan/kaybedilen eşleşmeler")
    ap.add_argument("--save", type=int, default=1, help="kayıt bazında sonucu <oof>/pool_coverage.csv'ye yaz")
    args = ap.parse_args()
    folds = get_folds().set_index("stem")["fold"]
    stems = [s for s in sorted(REC) if folds[s] in args.folds]
    import pandas as pd
    for r in args.runs:
        jobs = [(s, Path("outputs/oof") / f"{r}_tta2" / f"fold{folds[s]}" / "val_probs" / f"{s}.npy") for s in stems]
        with ProcessPoolExecutor(12) as ex:
            D = pd.DataFrame([x for img in ex.map(_image, jobs, chunksize=4) for x in img],
                             columns=["stem", "k", "kind", "grup"])
        c = D.kind.value_counts()
        print(f"{r:24s} havuzda eşleşen {c.get('eşleşen', 0)}/{len(D)} ({c.get('eşleşen', 0) / len(D):.4f}) | "
              f"sinyal_yok {c.get('sinyal_yok', 0)} | diğer {c.get('diğer', 0)}", flush=True)
        print("   eşleşen, uzlaşı grubuna göre:", D[D.kind == "eşleşen"].grup.value_counts().to_dict(), flush=True)
        if args.save:
            D.to_csv(Path("outputs/oof") / f"{r}_tta2" / "pool_coverage.csv", index=False)
        ref = Path("outputs/oof") / f"{args.ref}_tta2" / "pool_coverage.csv" if args.ref else None
        if ref is not None and ref.exists() and r != args.ref:
            R = pd.read_csv(ref).merge(D, on=["stem", "k"], suffixes=("_ref", ""))
            R = R[R.stem.isin(stems)]
            new, ref_ = R.kind == "eşleşen", R.kind_ref == "eşleşen"
            print(f"   {args.ref} karşısında: ikisi {int((new & ref_).sum())} | yalnız yeni {int((new & ~ref_).sum())} "
                  f"{R[new & ~ref_].grup_ref.value_counts().to_dict()} | yalnız referans {int((~new & ref_).sum())} "
                  f"{R[~new & ref_].grup_ref.value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
