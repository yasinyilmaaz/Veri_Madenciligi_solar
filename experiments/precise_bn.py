"""Deney 35: PreciseBN — BatchNorm çalışan istatistiklerini fold'un eğitim görüntüleriyle yeniden hesapla.

Kaynak: Wu ve Johnson 2021, "Rethinking Batch in BatchNorm", arXiv:2105.07576.
Eğitimde BN istatistikleri 1536 kesitlerin üstel ortalaması (momentum 0,1, bs 2); çıkarım ise tam 2048
görüntüde, tam görüntü normalizasyonuyla. Burada ağırlıklar sabit; yalnızca BN ortalama/varyansları,
o fold'un EĞİTİM görüntülerinde (validasyon ve test hiç kullanılmaz) çıkarımla aynı ön işlemeyle
(tam 2048, train.normalize) birikimli olarak (momentum=None) yeniden hesaplanır.
Çıktı outputs/<run>_pbn/fold{f}/ (best.pt + args.json); eval_runs ile tabana karşı ölçülür.
Örnek: PYTHONPATH=src python experiments/precise_bn.py --run crop1536_effv2s --folds 0 1
"""
import argparse
import json
import shutil
from pathlib import Path

import torch

from data import TRAIN_IMG, get_folds, read_image
from predict import load_fold_model
from train import normalize


def recompute(model, stems, flips=True):
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # birikimli (eşit ağırlıklı) ortalama
    model.train()  # yalnızca BN'ler için; modelde dropout yok (U-Net + EffNetV2/drop_path kapalı)
    for m in model.modules():
        if not isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.training = False
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for s in stems:
            x = torch.from_numpy(normalize(read_image(TRAIN_IMG / f"{s}.jpeg")))[None, None].cuda()
            model(x)
            if flips:
                model(x.transpose(2, 3).flip(3))
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    folds = get_folds()
    for f in args.folds:
        src = Path("outputs") / args.run / f"fold{f}"
        dst = Path("outputs") / f"{args.run}_pbn" / f"fold{f}"
        dst.mkdir(parents=True, exist_ok=True)
        model = recompute(load_fold_model(src), folds.loc[folds.fold != f, "stem"].tolist())
        torch.save(model.state_dict(), dst / "best.pt")
        a = json.loads((src / "args.json").read_text())
        (dst / "args.json").write_text(json.dumps(a))
        shutil.copy(src / "log.csv", dst / "log.csv")
        print(f"fold{f} BN yeniden hesaplandı", flush=True)


if __name__ == "__main__":
    main()
