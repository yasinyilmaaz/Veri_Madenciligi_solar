"""OOF olasılık dizinlerini eşit ağırlıkla ortalar (run başına uint8 harita, yuvarlama pipeline_ens1536 ile aynı).

Örnek: PYTHONPATH=src python experiments/ens_avg.py --runs crop1536_effv2s_tta2 effv2m_c1536b_tta2 ssl1536_effv2s_tta2 --out ens1536ssl_tta2
"""
import argparse

import numpy as np

from data import OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="outputs/oof altındaki dizin adları")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    oof = OUT / "oof"
    n = 0
    for f in range(5):
        out = oof / args.out / f"fold{f}" / "val_probs"
        out.mkdir(parents=True, exist_ok=True)
        for p in sorted((oof / args.runs[0] / f"fold{f}" / "val_probs").glob("*.npy")):
            m = sum(np.load(oof / r / f"fold{f}" / "val_probs" / p.name).astype(np.float32) for r in args.runs) / len(args.runs)
            np.save(out / p.name, np.round(m).astype(np.uint8))
            n += 1
    print(f"{n} görüntü -> {oof / args.out}")


if __name__ == "__main__":
    main()
