"""Deney 10: ensemble damıtması için fold'a özel öğretmen olasılıkları.

Fold f'nin öğretmeni = fold f'nin ensemble modelleri (fold f validasyonunu hiç görmedi). Bu modeller
fold f'nin EĞİTİM görüntülerini tahmin eder; train.py --teacher_dir ile hedefe karıştırılır.
Böylece fold f'nin CV skoruna validasyon bilgisi sızmaz.
Örnek: PYTHONPATH=src python experiments/make_teacher.py --runs effv2m_mean cnxt_mean hrnet_mean effv2s_random --folds 0 1
"""
import argparse

import numpy as np

from data import OUT, TRAIN_IMG, get_folds, read_image
from predict import load_fold_model
from train import predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--name", default="ens4")
    ap.add_argument("--tta", type=int, default=1)
    args = ap.parse_args()
    folds = get_folds()
    for f in args.folds:
        models = [load_fold_model(OUT / r / f"fold{f}") for r in args.runs]
        out = OUT / "teacher" / args.name / f"fold{f}"
        out.mkdir(parents=True, exist_ok=True)
        stems = folds.loc[folds.fold != f, "stem"].tolist()
        for s in stems:
            img = read_image(TRAIN_IMG / f"{s}.jpeg")
            prob = sum(predict(m, img, tta=args.tta) for m in models) / len(models)
            np.save(out / f"{s}.npy", (prob * 255).astype(np.uint8))
        print(f"fold{f}: {len(stems)} eğitim görüntüsü", flush=True)


if __name__ == "__main__":
    main()
