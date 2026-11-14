"""Deneylerin adil değerlendirmesi (fold 0+1) ve ensemble katkı testi; sonucu deney günlüğüne ekler.

Her run için D4 TTA OOF (yoksa üretilir) + kendi post ızgarası; tabanla fold bazında karşılaştırma.
--ens verilirse, her run mevcut ensemble'a eklenip PQ katkısı ölçülür.

Örnek: PYTHONPATH=src python experiments/eval_runs.py --base effv2m_mean --runs unetpp_effv2m --ens effv2m_mean cnxt_mean hrnet_mean effv2s_random --doc docs/experiments/03-unetpp-guclu-encoder.md
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import screen_eval as se
from data import get_folds, gt_frame, load_records
from ens_select import score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ens", nargs="*", default=[], help="mevcut ensemble üyeleri (katkı testi için)")
    ap.add_argument("--doc", default=None, help="sonuç tablosunun ekleneceği günlük dosyası")
    args = ap.parse_args()
    records = load_records()
    folds = get_folds().set_index("stem")["fold"]
    stems = [s for s in sorted(records) if folds[s] in se.FOLDS]
    gt = gt_frame(records, stems)
    res = {r: se.evaluate(r, gt, folds) for r in [args.base] + args.runs}
    b = res[args.base]
    lines = [f"### Ayarlı PQ (fold 0+1, D4 TTA; {datetime.now():%Y-%m-%d %H:%M})",
             "| Run | PQ | Fark | Fold 0 | Fold 1 | TP | FP | FN | Karar |", "|---|---|---|---|---|---|---|---|---|"]
    for r, v in res.items():
        d0, d1, d = v["f0"] - b["f0"], v["f1"] - b["f1"], v["pq"] - b["pq"]
        dec = "taban" if r == args.base else ("KABUL" if d >= 0.005 and d0 > 0 and d1 > 0 else
                                               ("sınırda" if d > 0 and min(d0, d1) > -0.002 else "ret"))
        lines.append(f"| {r} | {v['pq']:.4f} | {d:+.4f} | {v['f0']:.4f} ({d0:+.4f}) | {v['f1']:.4f} ({d1:+.4f}) | "
                     f"{v['tp']} | {v['fp']} | {v['fn']} | {dec} |")
    if args.ens:
        with ProcessPoolExecutor(16) as ex:
            e0, _ = score(args.ens, stems, folds, gt, ex)
            lines += ["", "### Ensemble katkısı (fold 0+1, eşit ağırlık)", "| Ensemble | PQ | Fark |", "|---|---|---|",
                      f"| mevcut ({', '.join(args.ens)}) | {e0:.4f} | – |"]
            for r in args.runs:
                e1, _ = score(args.ens + [r], stems, folds, gt, ex)
                lines.append(f"| + {r} | {e1:.4f} | {e1 - e0:+.4f} |")
    text = "\n".join(lines)
    print(text)
    if args.doc:
        with open(args.doc, "a") as f:
            f.write("\n" + text + "\n")


if __name__ == "__main__":
    main()
