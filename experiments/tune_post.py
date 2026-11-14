"""OOF olasılık haritaları üzerinde post-processing ızgara araması.

Tüm fold'ların val_probs klasörleri birlikte verilir; PQ havuzlanmış OOF
üzerinden hesaplanır ve fold'lar arası standart sapma da raporlanır
(tek fold'a uyan ayarları seçmemek için).

Örnek: PYTHONPATH=src python experiments/tune_post.py outputs/unet_r34
"""
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from data import get_folds, gt_frame, load_records
from metric import compute_pq, instances_to_rows
from post import prob_to_instances

GRID = dict(
    thr=[0.4, 0.5, 0.6],
    min_area=[300, 400, 600],
    join=[0, 8],
    thr_high=[None, 0.7, 0.8, 0.9],
)


def rows_for(args):
    path, cfgs = args
    prob = np.load(path).astype(np.float32) / 255
    out = []
    for c in cfgs:
        if c["thr_high"] is not None and c["thr_high"] <= c["thr"]:
            out.append(None)
            continue
        out.append(instances_to_rows(path.stem, prob_to_instances(prob, **c)))
    return out


def prob_paths(run_dirs):
    paths = []
    for rd in run_dirs:
        rd = Path(rd)
        dirs = [rd] if rd.name == "val_probs" else sorted(rd.glob("fold*/val_probs"))
        for d in dirs:
            paths += sorted(d.glob("*.npy"))
    return paths


def main():
    paths = prob_paths(sys.argv[1:])
    records = load_records()
    folds = get_folds().set_index("stem")["fold"]
    stems = [p.stem for p in paths]
    gt = gt_frame(records, stems)
    gt_fold = gt.filament_id.str.split("-", n=1).str[1].str.rsplit("_", n=1).str[0].map(folds)
    keys = list(GRID)
    cfgs = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    with ProcessPoolExecutor(16) as ex:
        per_img = list(ex.map(rows_for, [(p, cfgs) for p in paths]))
    res = []
    for i, c in enumerate(cfgs):
        if per_img[0][i] is None:
            continue
        pred = pd.DataFrame([r for img in per_img for r in img[i]], columns=["filament_id", "segmentation_rle"])
        entry = dict(c, **compute_pq(gt, pred))
        pred_fold = pred.filament_id.str.rsplit("_", n=1).str[0].map(folds)
        per_fold = [compute_pq(gt[gt_fold == f], pred[pred_fold == f])["pq"] for f in sorted(set(gt_fold))]
        entry["fold_std"] = float(np.std(per_fold))
        entry["fold_min"] = float(np.min(per_fold))
        res.append(entry)
    df = pd.DataFrame(res).sort_values("pq", ascending=False)
    print(f"{len(paths)} görüntü, {gt_fold.nunique()} fold")
    print(df.head(15).round(4).to_string(index=False))
    out = Path(sys.argv[1])
    df.to_csv((out.parent if out.name == "val_probs" else out) / "post_grid.csv", index=False)


if __name__ == "__main__":
    main()
