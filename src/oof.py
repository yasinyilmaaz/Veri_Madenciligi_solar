"""Her fold'un best.pt'si ile kendi validasyon görüntülerine OOF olasılık haritası üretir.

Birden fazla run verilirse aynı fold'un modelleri ortalanır (ensemble OOF).
Örnek: python src/oof.py --runs outputs/unet_r34 --tta 2 --out outputs/oof/unet_r34_tta2
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from data import TRAIN_IMG, get_folds, read_image
from train import load_model, predict


def check_manifest(fd, stems, allow_legacy=False):
    """OOF'u yazılacak görüntüler tam olarak modelin kayıtlı doğrulama kümesi olmalı ve eğitim kümesiyle
    kesişmemeli. Manifestsiz eski modeller yalnız --allow_legacy ile kabul
    edilir; onlarda fold şeması yalnız FOLD_SCHEME ortam değişkeniyle belirleniyordu ve doğrulanamaz."""
    a = json.loads((fd / "args.json").read_text())
    if "train_stems" not in a:
        if not allow_legacy:
            raise SystemExit(f"{fd} manifestsiz (eski model); fold şeması doğrulanamıyor. Bilerek kullanmak için --allow_legacy")
        print(f"uyarı: {fd} manifestsiz (eski model), --allow_legacy ile doğrulamasız kullanılıyor", flush=True)
        return
    scheme = os.environ.get("FOLD_SCHEME", "image")
    if a["fold_scheme"] != scheme:
        raise SystemExit(f"{fd} {a['fold_scheme']} fold şemasıyla eğitildi, OOF {scheme} şemasıyla isteniyor")
    if not stems:
        raise SystemExit(f"{fd}: OOF görüntü listesi boş")
    leak = set(stems) & set(a["train_stems"])
    if leak:
        raise SystemExit(f"{fd}: {len(leak)} OOF görüntüsü modelin eğitim kümesinde (ör. {sorted(leak)[0]})")
    if set(stems) != set(a["val_stems"]):
        extra, miss = set(stems) - set(a["val_stems"]), set(a["val_stems"]) - set(stems)
        raise SystemExit(f"{fd}: OOF listesi kayıtlı doğrulama kümesinden farklı ({len(extra)} fazla, {len(miss)} eksik)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tta", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--allow_legacy", action="store_true", help="manifestsiz eski modelleri doğrulamasız kabul et")
    args = ap.parse_args()

    folds = get_folds()
    for f in args.folds:
        stems = folds.loc[folds.fold == f, "stem"].tolist()
        for r in args.runs:  # modeller GPU'ya yüklenmeden önce
            check_manifest(Path(r) / f"fold{f}", stems, args.allow_legacy)
        models = [load_model(Path(r) / f"fold{f}") for r in args.runs]
        out = Path(args.out) / f"fold{f}" / "val_probs"
        out.mkdir(parents=True, exist_ok=True)
        for s in stems:
            img = read_image(TRAIN_IMG / f"{s}.jpeg")
            prob = sum(predict(m, img, tta=args.tta) for m in models) / len(models)
            np.save(out / f"{s}.npy", (prob * 255).astype(np.uint8))
        print(f"fold{f} bitti", flush=True)


if __name__ == "__main__":
    main()
