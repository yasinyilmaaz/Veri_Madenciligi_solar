"""Rapor analizi: modelin yüksek güvenle bulduğu ama hiçbir anotatörün çizmediği adaylar.

Aday seçicinin OOF skorlarını (candidates_scored.csv) kullanır; anotatör
maskeleriyle hiç kesişmeyen yüksek skorlu adayları sayar ve örnek kırpıntıları çizer.
Örnek: PYTHONPATH=src python experiments/missing_labels.py outputs/oof/ens3_mean_tta2
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from data import ROOT, TRAIN_IMG, load_records, poly_to_mask, read_image
from post import prob_to_instances
from scorer import CAND


def main():
    oof = Path(sys.argv[1])
    df = pd.read_csv(oof / "candidates_scored.csv")
    records = load_records()
    hi = df[(df.score > 0.35) & (df.y == 0)].sort_values("score", ascending=False)
    rows, crops = [], []
    for stem, g in hi.groupby("stem", sort=False):
        prob = np.load(next(oof.glob(f"fold*/val_probs/{stem}.npy"))).astype(np.float32) / 255
        lab = prob_to_instances(prob, **CAND)
        ann = np.zeros((2048, 2048), bool)
        for anns in records[stem].values():
            for a in anns:
                ann |= poly_to_mask(a["seg"])
        img = read_image(TRAIN_IMG / f"{stem}.jpeg")
        for _, r in g.iterrows():
            m = lab == r.inst
            untouched = not (m & ann).any()
            rows.append(dict(stem=stem, inst=r.inst, score=r.score, area=r.area, untouched=untouched,
                             n_annot=len(records[stem])))
            if untouched and len(crops) < 12:
                ys, xs = np.nonzero(m)
                cy, cx = int(ys.mean()), int(xs.mean())
                y0, x0 = np.clip(cy - 128, 0, 1792), np.clip(cx - 128, 0, 1792)
                c = cv2.cvtColor(img[y0:y0 + 256, x0:x0 + 256], cv2.COLOR_GRAY2BGR)
                for mask, col in [(ann, (0, 200, 0)), (m, (0, 0, 255))]:
                    cnts, _ = cv2.findContours(mask[y0:y0 + 256, x0:x0 + 256].astype(np.uint8),
                                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(c, cnts, -1, col, 1)
                cv2.putText(c, f"{r.score:.2f} n={len(records[stem])}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                crops.append(cv2.resize(c, (320, 320)))
    res = pd.DataFrame(rows)
    res.to_csv(oof / "missing_label_candidates.csv", index=False)
    print(f"skor>0.35 ve hiçbir anotatörle eşleşmeyen aday: {len(res)}; hiç dokunulmamış: {int(res.untouched.sum())}")
    print("anotatör sayısına göre (dokunulmamış):", res[res.untouched].n_annot.value_counts().sort_index().to_dict())
    print("toplam aday içindeki payı:", f"{res.untouched.sum() / len(df):.1%}", "| tüm FP içindeki payı:",
          f"{res.untouched.sum() / (df.y == 0).sum():.1%}")
    while len(crops) % 4:
        crops.append(np.zeros_like(crops[0]))
    grid = np.vstack([np.hstack(crops[i:i + 4]) for i in range(0, len(crops), 4)])
    out = ROOT / "viz" / "report_missing_labels.jpg"
    cv2.imwrite(str(out), grid)
    print("görsel:", out)


if __name__ == "__main__":
    main()
