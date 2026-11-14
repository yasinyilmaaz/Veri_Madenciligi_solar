"""Sistemleri resmi rubriğin tüm nicel ölçütleriyle karşılaştırır (5-fold OOF).

PQ + IoU/Dice dağılımı + parçalanma (bire-çok) ve birleşme (çoka-bir) oranları.
Örnek: python src/rubric_eval.py
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from artifacts import load_candidates_scored, mean_cnn, merge_exact
from data import OUT, gt_frame, load_records
from metric import compute_pq, instances_to_rows, rubric_stats
from post import prob_to_instances

SYSTEMS = {
    "ResNet34 birleşim, el ayarlı": ("oof/unet_r34_tta2", dict(thr=0.6, min_area=400, join=8, thr_high=0.9)),
    "ResNet34 ortalama, el ayarlı": ("oof/tgt_mean_tta2", dict(thr=0.5, min_area=400, join=8, thr_high=0.9)),
    "3 encoder ens., el ayarlı": ("oof/ens3_mean_tta2", dict(thr=0.4, min_area=300, join=8, thr_high=0.9)),
    "ens1536 (S+M), eyer hattı": ("oof/ens1536_tta2", dict(thr=0.4, min_area=100, join=0, link_thr=0.15)),
}
# seçicili sistemler: (OOF klasörü, skor, kesme); skor "lgb" = LightGBM, "mix" = 0,5·LightGBM + 0,5·CNN (3 tohum)
SELECTED = {
    "3 encoder ens. + aday seçici": ("ens3_mean_tta2", "lgb", 0.22),
    "ens1536 + LightGBM seçici": ("ens1536_tta2", "lgb", 0.24),
    "ens1536 + LightGBM/CNN seçici (teslim)": ("ens1536_tta2", "mix", 0.25),
}
FINAL = "ens1536 + LightGBM/CNN seçici (teslim)"


def _rows(args):
    path, cfg = args
    prob = np.load(path).astype(np.float32) / 255
    return instances_to_rows(path.stem, prob_to_instances(prob, **cfg))


def selected(run, score, cut):
    """OOF adaylarından seçicinin geçirdikleri, submission biçiminde."""
    oof = OUT / "oof" / run
    c = load_candidates_scored(oof)
    if score == "mix":
        c = merge_exact(c, mean_cnn(oof), ["stem", "inst"], "aday", "CNN skoru")
        c["score"] = 0.5 * c.score + 0.5 * c.cnn
    k = c[c.score > cut]
    return pd.DataFrame({"filament_id": k.stem + "_" + k.inst.astype(str), "segmentation_rle": k.rle})


def main():
    records = load_records()
    gt = gt_frame(records, sorted(records))
    res = []
    for name, (d, cfg) in SYSTEMS.items():
        paths = sorted((OUT / d).glob("fold*/val_probs/*.npy"))
        with ProcessPoolExecutor(16) as ex:
            rows = [r for img in ex.map(_rows, [(p, cfg) for p in paths], chunksize=4) for r in img]
        pred = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        res.append(dict(sistem=name, n_pred=len(pred), **compute_pq(gt, pred), **rubric_stats(gt, pred)))
    for name, (run, score, cut) in SELECTED.items():
        pred = selected(run, score, cut)
        res.append(dict(sistem=name, n_pred=len(pred), **compute_pq(gt, pred), **rubric_stats(gt, pred)))
    # nihai sistemin dağılımları (rapor histogramları)
    final = selected(*SELECTED[FINAL])
    np.savez(OUT / "rubric_final_dist.npz", **rubric_stats(gt, final, return_arrays=True))
    df = pd.DataFrame(res)
    cols = ["sistem", "n_pred", "pq", "sq", "rq", "iou_mean", "dice_mean", "gt_miss", "gt_split", "pred_miss",
            "pred_merge"]
    print(df[cols].round(4).to_string(index=False))
    df.to_csv(OUT / "rubric_eval.csv", index=False)


if __name__ == "__main__":
    main()
