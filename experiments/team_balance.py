"""Takım dengeli LightGBM seçici: aday ağırlığı = 1 / takımın aday sayısı (ön kayıtlı deney, 02.11).

Ölçüm 1: takım fold'lu OOF'ta takım dışı iç içe değerlendirme (seçici diğer takımlarla eğitilir).
Ölçüm 2: ana sistemin görüntü fold'larında kontrol.
Örnek: PYTHONPATH=src python experiments/team_balance.py
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from artifacts import load_candidates_scored
from data import OUT, gt_frame, load_records
from metric import compute_pq
from scorer import FEATS, PARAMS, ROUNDS

CUT = 0.24
PARAMS = dict(PARAMS, num_threads=8)  # küçük veride 24 iş parçacığı eğitimi ~1000 kat yavaşlatıyor


def team_of(records):
    """Görüntünün etiketleme takımı (anotatör kimliğinin ilk iki hanesi; görüntü başına tek takım)."""
    return {s: next(iter(records[s])).split("-")[0][:2] for s in records}


def nested(df, fold_col, balanced):
    score = np.full(len(df), np.nan)
    for f in sorted(df[fold_col].unique()):
        tr, va = (df[fold_col] != f).to_numpy(), (df[fold_col] == f).to_numpy()
        w = None
        if balanced:
            n = df.loc[tr, "team"].map(df.loc[tr, "team"].value_counts())
            w = (1.0 / n) * tr.sum() / (1.0 / n).sum()
        m = lgb.train(PARAMS, lgb.Dataset(df.loc[tr, FEATS], df.loc[tr, "y"], weight=w), num_boost_round=ROUNDS)
        score[va] = m.predict(df.loc[va, FEATS])
    return score


def pq_sel(df, gt, col, stems=None):
    k = df[df[col] > CUT]
    if stems is not None:
        k = k[k.stem.isin(stems)]
        gt = gt[gt.filament_id.str.split("-", n=1).str[1].str.rsplit("_", n=1).str[0].isin(stems)]
    return compute_pq(gt, pd.DataFrame({"filament_id": k.stem + "_" + k.inst.astype(str), "segmentation_rle": k.rle}))["pq"]


def main():
    records = load_records()
    team = team_of(records)
    gt = gt_frame(records, sorted(records))

    # ölçüm 1: takım dışı
    oof = OUT / "oof" / "team1536_effv2s_tta2"
    df = load_candidates_scored(oof, "_team").drop(columns=["score"])
    df["team"] = df.stem.map(team)
    df["base"], df["bal"] = nested(df, "team", False), nested(df, "team", True)
    stems_all = sorted({p.stem for p in oof.glob("fold*/val_probs/*.npy")})
    gt1 = gt[gt.filament_id.str.split("-", n=1).str[1].str.rsplit("_", n=1).str[0].isin(stems_all)]
    rows = []
    for t in sorted(df.team.unique()):
        st = [s for s in stems_all if team[s] == t]
        rows.append(dict(team=t, n_img=len(st), base=pq_sel(df, gt1, "base", st), bal=pq_sel(df, gt1, "bal", st)))
    r = pd.DataFrame(rows)
    r["diff"] = r.bal - r.base
    print(r.round(4).to_string(index=False))
    print(f"makro: taban {r.base.mean():.4f}, dengeli {r.bal.mean():.4f}, fark {r['diff'].mean():+.4f}, "
          f"iyileşen takım {(r['diff'] > 0).sum()}/{len(r)}")
    print(f"havuz: taban {pq_sel(df, gt1, 'base'):.4f}, dengeli {pq_sel(df, gt1, 'bal'):.4f}")

    # ölçüm 2: ana sistem, görüntü fold'ları
    oof = OUT / "oof" / "ens1536_tta2"
    df = load_candidates_scored(oof).drop(columns=["score"])
    df["team"] = df.stem.map(team)
    folds = {p.stem: int(p.parent.parent.name[4:]) for p in oof.glob("fold*/val_probs/*.npy")}
    df["fold"] = df.stem.map(folds)
    df["base"], df["bal"] = nested(df, "fold", False), nested(df, "fold", True)
    print(f"kontrol (ens1536, görüntü fold'ları, kesme {CUT}): taban {pq_sel(df, gt, 'base'):.4f}, dengeli {pq_sel(df, gt, 'bal'):.4f}")


if __name__ == "__main__":
    main()
