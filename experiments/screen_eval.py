"""Tarama deneylerinin toplu ve adil değerlendirmesi (fold 0+1).

Her deney için: D4 TTA OOF (yoksa üretilir) -> kendi post-processing ızgarası -> fold bazında PQ.
Kendi tabanıyla karşılaştırılır; protokol: havuzlanmış >= +0.005 VE iki fold da iyileşmeli.
Örnek: PYTHONPATH=src python experiments/screen_eval.py
"""
import itertools
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from data import OUT, get_folds, gt_frame, load_records
from metric import compute_pq, instances_to_rows
from post import prob_to_instances
from tune_post import GRID

# deney -> taban
RUNS = {
    "tgt_mean": None, "effv2s_mean": None, "cnxt_mean": None,
    "annot": "tgt_mean", "clean_mean": "tgt_mean", "pl2_mix": "tgt_mean", "r34_mean_strong": "tgt_mean",
    "bal_year": "tgt_mean", "bal_inst": "tgt_mean", "bal_sizeloss": "tgt_mean", "aug_rot360": "tgt_mean",
    "hrnet_mean": "tgt_mean", "effv2s_random": "effv2s_mean", "effv2s_mean_strong": "effv2s_mean",
    "effv2m_mean": "effv2s_mean", "cnxtS_mean": "cnxt_mean",
}
FOLDS = [0, 1]


def _rows(args):
    path, cfgs = args
    prob = np.load(path).astype(np.float32) / 255
    return [instances_to_rows(path.stem, prob_to_instances(prob, **c)) for c in cfgs]


def evaluate(run, gt, folds):
    oof = OUT / "oof" / f"{run}_tta2"
    missing = [f for f in FOLDS if not (oof / f"fold{f}" / "val_probs").exists()]
    if missing:
        subprocess.run([sys.executable, "src/oof.py", "--runs", f"outputs/{run}", "--tta", "2", "--folds",
                        *map(str, missing), "--out", str(oof)], check=True)
    paths = sorted(p for f in FOLDS for p in (oof / f"fold{f}" / "val_probs").glob("*.npy"))
    cfgs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    cfgs = [c for c in cfgs if c["thr_high"] is None or c["thr_high"] > c["thr"]]
    with ProcessPoolExecutor(16) as ex:
        per_img = list(ex.map(_rows, [(p, cfgs) for p in paths], chunksize=4))
    best = None
    for i, c in enumerate(cfgs):
        pred = pd.DataFrame([r for img in per_img for r in img[i]], columns=["filament_id", "segmentation_rle"])
        r = compute_pq(gt, pred)
        if best is None or r["pq"] > best[0]["pq"]:
            best = (r, c, pred)
    r, c, pred = best
    gfold = gt.filament_id.str.split("-", n=1).str[1].str.rsplit("_", n=1).str[0].map(folds)
    pfold = pred.filament_id.str.rsplit("_", n=1).str[0].map(folds)
    per = {f: compute_pq(gt[gfold == f], pred[pfold == f])["pq"] for f in FOLDS}
    return dict(run=run, pq=r["pq"], **{f"f{f}": per[f] for f in FOLDS}, tp=r["tp"], fp=r["fp"], fn=r["fn"],
                cfg=c)


def main():
    records = load_records()
    folds = get_folds().set_index("stem")["fold"]
    stems = [s for s in sorted(records) if folds[s] in FOLDS]
    gt = gt_frame(records, stems)
    res = {}
    for run in RUNS:
        res[run] = evaluate(run, gt, folds)
        print(run, round(res[run]["pq"], 4), flush=True)
    rows = []
    for run, base in RUNS.items():
        r = res[run]
        row = dict(deney=run, taban=base or "-", pq=r["pq"], f0=r["f0"], f1=r["f1"], tp=r["tp"], fp=r["fp"],
                   fn=r["fn"])
        if base:
            b = res[base]
            row["fark"] = r["pq"] - b["pq"]
            row["f0_fark"], row["f1_fark"] = r["f0"] - b["f0"], r["f1"] - b["f1"]
            ok = row["fark"] >= 0.005 and row["f0_fark"] > 0 and row["f1_fark"] > 0
            row["karar"] = "KABUL" if ok else ("sınırda" if row["fark"] > 0 and min(row["f0_fark"], row["f1_fark"]) > -0.002 else "ret")
        rows.append(row)
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    df.to_csv(OUT / "screen_eval.csv", index=False)


if __name__ == "__main__":
    main()
