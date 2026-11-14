"""Deney 51: etiketsiz GONG görüntüleriyle öz-denetimli ön eğitim (maskeli görüntü modelleme, U-Net).

Girdi görüntüsünün rastgele 32x32 yamalarının yarısı gizlenir (normalizasyon sonrası 0), ağ özgün görüntüyü
üretir; kayıp gizli piksellerde L1 (+ 0,1 · görünür piksellerde L1). Kodlayıcı ImageNet ağırlıklarıyla başlar.
Etiket kullanılmaz. Görüntüler: outputs/gong_keep.csv (test günleri ±3 gün indirmede dışlandı) ve
outputs/gong_new_keep.csv (2010 ve 2023, test yıllarının dışında).
Çıktı: outputs/ssl/<name>/model.pt -> train.py --ssl_init ile segmentasyon eğitiminin başlangıcı.

Örnek: PYTHONPATH=src python experiments/ssl_pretrain.py --name mim_effv2s --steps 15000
"""
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset

from data import EXT_IMG, OUT, ROOT, read_image
from train import normalize


def image_paths():
    a = [EXT_IMG / f"{s}.jpeg" for s in pd.read_csv(OUT / "gong_keep.csv").stem]
    b = [ROOT / "external" / "gong_new" / f"{s}.jpeg" for s in pd.read_csv(OUT / "gong_new_keep.csv").stem]
    return [p for p in a + b if p.exists()]


class MIMDataset(Dataset):
    def __init__(self, paths, crop, length):
        self.paths, self.crop, self.length = paths, crop, length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        x = normalize(read_image(random.choice(self.paths)))
        c = self.crop
        y0, x0 = random.randint(0, 2048 - c), random.randint(0, 2048 - c)
        x = x[y0:y0 + c, x0:x0 + c]
        if random.random() < 0.5:
            x = x[:, ::-1]
        x = np.rot90(x, random.randint(0, 3)).copy()
        g = c // 32
        keep = np.random.rand(g, g) > 0.5
        mask = np.kron(~keep, np.ones((32, 32), bool))  # True = gizli
        inp = np.where(mask, 0, x).astype(np.float32)
        return torch.from_numpy(inp[None]), torch.from_numpy(x[None].astype(np.float32)), torch.from_numpy(mask[None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--encoder", default="tu-tf_efficientnetv2_s")
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    out = OUT / "ssl" / args.name
    out.mkdir(parents=True, exist_ok=True)
    paths = image_paths()
    print(f"{len(paths)} etiketsiz görüntü", flush=True)
    ds = MIMDataset(paths, args.crop, args.steps * args.bs)
    dl = DataLoader(ds, batch_size=args.bs, num_workers=args.workers, pin_memory=True, drop_last=True,
                    worker_init_fn=lambda w: (np.random.seed(args.seed * 100 + w), random.seed(args.seed * 100 + w)))
    model = smp.Unet(args.encoder, encoder_weights="imagenet", in_channels=1, classes=1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    t0, run = time.time(), 0.0
    for step, (inp, x, mask) in enumerate(dl, 1):
        inp, x, mask = inp.cuda(non_blocking=True), x.cuda(non_blocking=True), mask.cuda(non_blocking=True).float()
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(inp)
        err = (pred.float() - x).abs()
        loss = (err * mask).sum() / mask.sum().clamp(min=1) + 0.1 * (err * (1 - mask)).sum() / (1 - mask).sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 500 == 0 or step == args.steps:
            print(json.dumps(dict(step=step, loss=round(run, 4), min=round((time.time() - t0) / 60, 1))), flush=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    print("kaydedildi:", out / "model.pt")


if __name__ == "__main__":
    main()
