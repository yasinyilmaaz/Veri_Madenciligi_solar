"""Açgözlü ensemble üye seçimi (fold 0+1 OOF olasılık ortalaması).

Her adımda ensemble'a en çok PQ katan aday eklenir; kazanç MIN_GAIN altına düşünce durulur.
Post-processing her aday ensemble için küçük bir ızgarada yeniden ayarlanır.
Örnek: PYTHONPATH=src python experiments/ens_select.py
"""
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from data import OUT, get_folds, gt_frame, load_records
from metric import compute_pq, instances_to_rows
from post import prob_to_instances

CANDIDATES = ["tgt_mean", "effv2s_mean", "cnxt_mean", "effv2m_mean", "hrnet_mean", "effv2s_random", "annot",
              "effv2s_mean_strong"]
FOLDS = [0, 1]
GRID = [dict(thr=t, min_area=a, join=8, thr_high=0.9) for t, a in itertools.product([0.4, 0.5, 0.6], [300, 400])]
MIN_GAIN = 0.002


def _rows(args):
    stem, members = args
    prob = np.mean([np.load(OUT / "oof" / f"{m}_tta2" / f"fold{f}" / "val_probs" / f"{stem}.npy").astype(np.float32)
                    for m, f in members], 0) / 255
    return [instances_to_rows(stem, prob_to_instances(prob, **c)) for c in GRID]


def score(members, stems, folds, gt, ex):
    jobs = [(s, [(m, folds[s]) for m in members]) for s in stems]
    per_img = list(ex.map(_rows, jobs, chunksize=4))
    best = (-1, None)
    for i, c in enumerate(GRID):
        pred = pd.DataFrame([r for img in per_img for r in img[i]], columns=["filament_id", "segmentation_rle"])
        pq = compute_pq(gt, pred)["pq"]
        if pq > best[0]:
            best = (pq, c)
    return best


def main():
    records = load_records()
    folds = get_folds().set_index("stem")["fold"]
    stems = [s for s in sorted(records) if folds[s] in FOLDS]
    gt = gt_frame(records, stems)
    chosen, cur = [], 0.0
    with ProcessPoolExecutor(16) as ex:
        while True:
            trials = {}
            for c in CANDIDATES:
                if c in chosen:
                    continue
                trials[c] = score(chosen + [c], stems, folds, gt, ex)
                print(f"  {chosen + [c]} -> {trials[c][0]:.4f}", flush=True)
            if not trials:
                break
            c, (pq, cfg) = max(trials.items(), key=lambda kv: kv[1][0])
            if chosen and pq - cur < MIN_GAIN:
                print(f"durdu: en iyi ek {c} yalnızca +{pq - cur:.4f}")
                break
            chosen.append(c)
            cur = pq
            print(f"SEÇİLDİ: {c} -> ensemble {chosen} PQ {pq:.4f} {cfg}", flush=True)
    print("final üyeler:", chosen, "PQ", round(cur, 4))


if __name__ == "__main__":
    main()
