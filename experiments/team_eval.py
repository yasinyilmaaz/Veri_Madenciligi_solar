"""Deney 45: bir OOF koşusunun düz eyer hattı PQ'sunu anotatör ekibine göre böler.

Hat mevcut sistemle aynı (eşik 0,4, eyer bağlantısı 0,15, en küçük alan 100), seçici yok. Ekip bilgisi
outputs/folds_team.csv'den (fold = ekip − 1); birleşik PQ = ΣS/ΣD. OOF görüntü-fold ya da ekip-fold
düzeninde olabilir: olasılık dosyaları fold*/val_probs altında aranır.
Örnek: PYTHONPATH=src python experiments/team_eval.py crop1536_effv2s team1536_effv2s
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from data import gt_frame, load_records
from metric import compute_pq, mask_to_rle
from post import prob_to_instances

REC = load_records()
OOF = Path("outputs/oof")


def _pred(path):
    lab = prob_to_instances(np.load(path).astype(np.float32) / 255, thr=0.4, min_area=100, join=0, link_thr=0.15)
    return [(f"{path.stem}_{k}", mask_to_rle(lab == k)) for k in range(1, lab.max() + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    team = pd.read_csv("outputs/folds_team.csv").set_index("stem")["fold"] + 1
    for run in args.runs:
        paths = {p.stem: p for p in sorted((OOF / f"{run}_tta2").glob("fold*/val_probs/*.npy"))}
        stems = sorted(s for s in paths if s in REC)
        with ProcessPoolExecutor(12) as ex:
            preds = dict(zip(stems, ex.map(_pred, [paths[s] for s in stems], chunksize=4)))
        S = D = 0.0
        print(f"{run}: {len(stems)} görüntü")
        for t in sorted(team[stems].unique()):
            ts = [s for s in stems if team[s] == t]
            pred = pd.DataFrame([r for s in ts for r in preds[s]], columns=["filament_id", "segmentation_rle"])
            r = compute_pq(gt_frame(REC, ts), pred)
            d = r["tp"] + 0.5 * r["fp"] + 0.5 * r["fn"]
            S, D = S + r["pq"] * d, D + d
            print(f"  ekip {t:02d} ({len(ts):3d} görüntü): PQ {r['pq']:.4f}  SQ {r['sq']:.4f}  RQ {r['rq']:.4f}  "
                  f"TP {r['tp']} FP {r['fp']} FN {r['fn']}", flush=True)
        print(f"  birleşik PQ (ΣS/ΣD): {S / D:.4f}", flush=True)


if __name__ == "__main__":
    main()
