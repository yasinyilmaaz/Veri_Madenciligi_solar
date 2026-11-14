"""Kırpılmış bölge düzeltici (refiner): seçilen her adayın etrafı kesilip ayrı bir U-Net ile yeniden çizilir.

1. build: tüm adaylar için 256x256 kesitler (normalize görüntü, ensemble olasılığı, aday maskesi) ve
   yumuşak hedef (her anotatörün adayla en çok örtüşen filamenti, IoU > 0.1; anotatör ortalaması).
2. train: fold f dışındaki görüntülerin kesitleriyle eğitim, fold f kesitlerinde Dice ile seçim.
3. eval: fold f'nin seçilen adayları düzeltilir, tam çözünürlüğe geri yapıştırılır; orijinal ile PQ karşılaştırması.
4. refine_instances: test zamanı için tek görüntü üzerinde aynı işlem (predict.py'ye henüz bağlı değil).

Örnek: PYTHONPATH=src python experiments/refiner.py build && PYTHONPATH=src python experiments/refiner.py train --fold 0 && PYTHONPATH=src python experiments/refiner.py eval
"""
import argparse
import json
import os
import random
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mu
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data import OUT, TRAIN_IMG, get_folds, gt_frame, load_records, poly_to_mask, poly_to_rle, read_image
from metric import compute_pq, mask_to_rle, rubric_stats
from train import dice_loss, normalize

OOF = OUT / "oof" / os.environ.get("REFINER_OOF", "ens4_tta2")  # deney 18: REFINER_OOF=ens1536_tta2
REF = OUT / "refiner"
CS = 256          # kesit boyutu
JOIN = 8          # aday üretimindeki join ile aynı (bileşen seçimi için)
DEV = os.environ.get("REFINER_DEVICE", "cuda")  # GPU doluyken deneme için: REFINER_DEVICE=cpu


# ---------------------------------------------------------------- kesit geometrisi

def crop_box(m):
    """Maskenin kutusu + kenar payı -> görüntü içinde kalan kare kutu (y0, x0, kenar).

    Kırpma yerine kaydırma: kare kalır, 256'ya ölçeklemede en-boy oranı bozulmaz.
    """
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    side = int(min(2048, max(h, w) + 2 * max(16, 0.25 * max(h, w))))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    by = int(np.clip(cy - side // 2, 0, 2048 - side))
    bx = int(np.clip(cx - side // 2, 0, 2048 - side))
    return by, bx, side


def to_crop(a, box):
    """Tam görüntüden kutuyu kes, 256x256'ya ölçekle (float32)."""
    y, x, s = box
    c = a[y:y + s, x:x + s].astype(np.float32)
    return cv2.resize(c, (CS, CS), interpolation=cv2.INTER_AREA if s > CS else cv2.INTER_LINEAR)


def from_crop(c, box):
    """256x256 olasılığı kutu boyutuna geri ölçekle."""
    s = box[2]
    return cv2.resize(c.astype(np.float32), (s, s), interpolation=cv2.INTER_LINEAR)


def make_input(norm, prob, m, box):
    """3 kanal: normalize görüntü, ensemble olasılığı (0..1), aday maskesi (0/1)."""
    return np.stack([to_crop(norm, box), to_crop(prob, box), (to_crop(m, box) > 0.5).astype(np.float32)])


# ---------------------------------------------------------------- build

def build_image(args):
    stem, fold, cands, record = args
    img = read_image(TRAIN_IMG / f"{stem}.jpeg")
    norm = normalize(img)
    prob = np.load(OOF / f"fold{fold}" / "val_probs" / f"{stem}.npy").astype(np.float32) / 255
    pred = [{"size": [2048, 2048], "counts": r.encode()} for r in cands.rle]
    tgt = [np.zeros((CS, CS), np.float32) for _ in pred]
    boxes = [crop_box(mu.decode(p)) for p in pred]
    # her anotatör: adayla en çok örtüşen GT (IoU > 0.1), yoksa boş; ortalama yumuşak hedef
    for anns in record.values():
        if not anns:
            continue
        gts = [poly_to_rle(a["seg"]) for a in anns]
        iou = np.asarray(mu.iou(pred, gts, [0] * len(gts))).reshape(len(pred), len(gts))
        for k in range(len(pred)):
            j = int(iou[k].argmax())
            if iou[k, j] > 0.1:
                tgt[k] += to_crop(poly_to_mask(anns[j]["seg"]).astype(np.float32), boxes[k])
    n_annot = len(record)
    out = dict(img=[], prob=[], mask=[], tgt=[], box=[])
    for p, b, t in zip(pred, boxes, tgt):
        x = make_input(norm, prob, mu.decode(p), b)
        out["img"].append(x[0].astype(np.float16))
        out["prob"].append(np.round(x[1] * 255).astype(np.uint8))
        out["mask"].append(x[2].astype(np.uint8))
        out["tgt"].append(np.round(np.clip(t / n_annot, 0, 1) * 255).astype(np.uint8))
        out["box"].append(b)
    return stem, {k: np.stack(v) for k, v in out.items()}


def cmd_build(args):
    records = load_records()
    folds = get_folds().set_index("stem")["fold"]
    df = pd.read_pickle(OOF / "candidates.pkl")
    sc = pd.read_csv(OOF / "candidates_scored.csv", usecols=["stem", "inst", "score"])
    df = df.merge(sc, on=["stem", "inst"], how="left")
    stems = sorted(df.stem.unique())[:args.limit or None]
    groups = {s: g for s, g in df[df.stem.isin(stems)].groupby("stem")}
    jobs = [(s, int(folds[s]), groups[s][["inst", "rle"]], records[s]) for s in stems]
    t0 = time.time()
    with ProcessPoolExecutor(16) as ex:
        res = dict(ex.map(build_image, jobs, chunksize=2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in range(5):
        ss = [s for s in stems if folds[s] == f]
        if not ss:
            continue
        g = pd.concat([groups[s] for s in ss])
        arr = {k: np.concatenate([res[s][k] for s in ss]) for k in res[ss[0]]}
        np.savez(out / f"crops_fold{f}.npz", **arr, stem=g.stem.to_numpy().astype(str), inst=g.inst.to_numpy(),
                 score=g.score.to_numpy(), y=g.y.to_numpy(), area=g.area.to_numpy(), fold=np.full(len(g), f))
        print(f"fold{f}: {len(g)} kesit, {len(ss)} görüntü, hedef dolu oranı "
              f"{(arr['tgt'].reshape(len(g), -1).max(1) > 127).mean():.3f}", flush=True)
    print(f"{len(res)} görüntü, {time.time() - t0:.0f} sn -> {out}")


def load_crops(folds, d=REF):
    parts = [dict(np.load(Path(d) / f"crops_fold{f}.npz")) for f in folds
             if (Path(d) / f"crops_fold{f}.npz").exists()]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def crop_tensors(c, idx):
    """Önbellekten model girdisi (N, 3, 256, 256) float32."""
    return np.stack([c["img"][idx].astype(np.float32), c["prob"][idx].astype(np.float32) / 255,
                     c["mask"][idx].astype(np.float32)], 1)


def cmd_viz(args):
    c = load_crops(range(5), args.out)
    rng = np.random.default_rng(0)
    full = np.nonzero(c["tgt"].reshape(len(c["tgt"]), -1).max(1) > 127)[0]
    empty = np.setdiff1d(np.arange(len(c["tgt"])), full)
    idx = np.concatenate([rng.choice(full, 4, replace=False), rng.choice(empty, min(2, len(empty)), replace=False)])
    rows = []
    for i in idx:
        im = np.clip((c["img"][i].astype(np.float32) * 40 + 128), 0, 255).astype(np.uint8)
        # mask ve hedef konturları görüntü üstünde: yeşil aday, kırmızı hedef > 0.5
        ov = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        for m, col in ((c["mask"][i], (0, 255, 0)), ((c["tgt"][i] > 127).astype(np.uint8), (0, 0, 255))):
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cnts, -1, col, 1)
        tiles = [cv2.cvtColor(t, cv2.COLOR_GRAY2BGR) for t in (im, c["prob"][i], c["mask"][i] * 255, c["tgt"][i])]
        row = np.concatenate(tiles + [ov], 1)
        cv2.putText(row, f"{c['stem'][i]} #{c['inst'][i]} s={c['score'][i]:.2f} y={c['y'][i]:.2f} box={c['box'][i][2]}",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        rows.append(row)
    path = Path(args.viz)
    path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(path), np.concatenate(rows, 0))
    print("kanallar: görüntü | olasılık | aday | hedef | kontur (yeşil aday, kırmızı hedef) ->", path)


# ---------------------------------------------------------------- train

class CropDS(Dataset):
    def __init__(self, c, aug):
        self.c, self.aug = c, aug

    def __len__(self):
        return len(self.c["tgt"])

    def __getitem__(self, i):
        x = crop_tensors(self.c, [i])[0]
        y = self.c["tgt"][i][None].astype(np.float32) / 255
        if self.aug:  # tüm kanallara ve hedefe aynı flip + rot90
            xy = np.concatenate([x, y])
            if random.random() < 0.5:
                xy = xy[:, :, ::-1]
            if random.random() < 0.5:
                xy = xy[:, ::-1]
            xy = np.rot90(xy, random.randrange(4), (1, 2))
            x, y = xy[:3], xy[3:]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def make_model(weights="imagenet"):
    return smp.Unet("resnet34", encoder_weights=weights, in_channels=3, classes=1)


@torch.no_grad()
def predict_crops(models, x, bs=64, tta=False):
    """x: (N, 3, 256, 256) numpy -> (N, 256, 256) olasılık, modellerin (ve flip TTA'nın) ortalaması."""
    out = []
    flips = [[], [3], [2], [2, 3]] if tta else [[]]
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(DEV)
        acc = 0
        with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
            for m in models:
                for f in flips:
                    p = torch.sigmoid(m(xb.flip(f) if f else xb).float())
                    acc = acc + (p.flip(f) if f else p)
        out.append((acc / (len(models) * len(flips)))[:, 0].cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, CS, CS), np.float32)


def crop_dice(p, t, thr=0.5):
    """Kesit başına sert Dice (ikisi de boşsa 1), ortalama."""
    p, t = p > thr, t > 0.5
    inter = (p & t).sum((1, 2))
    return float(((2 * inter + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)).mean())


def cmd_train(args):
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    run = Path(args.out) / f"fold{args.fold}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(args), indent=1))
    tr = load_crops([f for f in range(5) if f != args.fold], args.crops)
    va = load_crops([args.fold], args.crops)
    xv, tv = crop_tensors(va, np.arange(len(va["tgt"]))), va["tgt"].astype(np.float32) / 255
    sel = va["score"] > args.cut
    base = crop_dice(xv[:, 2], tv)  # referans: aday maskesinin kendisi
    print(f"train {len(tr['tgt'])} kesit, val {len(tv)} kesit; aday maskesi Dice: tümü {base:.4f}, "
          f"seçilen {crop_dice(xv[sel, 2], tv[sel]):.4f}", flush=True)
    dl = DataLoader(CropDS(tr, True), batch_size=args.bs, shuffle=True, num_workers=args.workers,
                    drop_last=True, pin_memory=True, persistent_workers=True)
    steps = min(len(dl), args.limit_steps or len(dl))
    model = make_model().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=DEV == "cuda")
    best, log = -1, []
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        if DEV == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for n, (x, y) in enumerate(dl):
            if n >= steps:
                break
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV == "cuda"):
                logits = model(x)
            logits = logits.float()
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        sec = time.time() - t0
        model.eval()
        pv = predict_crops([model], xv, bs=args.bs)
        entry = dict(epoch=ep, loss=tot / steps, dice=crop_dice(pv, tv), dice_sel=crop_dice(pv[sel], tv[sel]),
                     sec=round(sec), gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if DEV == "cuda" else 0)
        if entry["dice"] > best:
            best = entry["dice"]
            torch.save(model.state_dict(), run / "best.pt")
        log.append(entry)
        print(json.dumps({k: round(float(v), 4) for k, v in entry.items()}), flush=True)
        pd.DataFrame(log).to_csv(run / "log.csv", index=False)
    print(f"en iyi val Dice {best:.4f} (aday maskesi {base:.4f}) -> {run / 'best.pt'}")


# ---------------------------------------------------------------- geri yapıştırma

def keep_component(binary, orig, join=JOIN):
    """Düzeltilmiş maskenin orijinal adayla en çok örtüşen bileşeni (join ile birleşik parçalar tek bileşen).

    Boşsa veya örtüşme yoksa None döner (çağıran orijinal maskeyi korur).
    """
    if not binary.any():
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * join + 1, 2 * join + 1))
    _, lab = cv2.connectedComponents(cv2.dilate(binary.astype(np.uint8), k), connectivity=8)
    lab = lab * binary
    ov = np.bincount(lab[orig], minlength=lab.max() + 1)
    ov[0] = 0
    if ov.max() == 0:
        return None
    return lab == ov.argmax()


def paste_back(masks, boxes, scores, probs=None, base_prob=None, thr=0.5, blend=0.0):
    """Düzeltilmiş kesitleri tam çözünürlüğe yerleştir, çakışmaları skora göre çöz.

    masks: tam 2048 orijinal aday maskeleri; probs: (N, 256, 256) refiner olasılığı (None = özdeşlik:
    orijinal maske aynen kullanılır, yapıştırma/çakışma mantığını doğrular). blend > 0 ise kutudaki tam
    çözünürlüklü ensemble olasılığıyla karıştırılır: p = (1 - blend) * refined + blend * ensemble.
    Dönüş: etiket haritası (1..N, skor sırası korunmadan aday sırasına göre) ve geri düşme sayısı.
    """
    lab = np.zeros((2048, 2048), np.int32)
    fallback = 0
    for k in np.argsort(-np.asarray(scores), kind="stable"):
        y, x, s = boxes[k]
        orig = masks[k][y:y + s, x:x + s]
        if probs is None:
            new = orig
        else:
            p = from_crop(probs[k], boxes[k])
            if blend > 0:
                p = (1 - blend) * p + blend * base_prob[y:y + s, x:x + s]
            new = keep_component(p > thr, orig)
            if new is None:
                new, fallback = orig, fallback + 1
        # aday kutusunun dışındaki orijinal pikseller (yoksa) kaybolmaz: kutu adayı tamamen içerir
        win = lab[y:y + s, x:x + s]
        win[new & (win == 0)] = k + 1  # yüksek skorlu instance çekişmeli pikselleri tutar
    return lab, fallback


def refine_instances(img, prob, lab, scores, models, cut=0.24, thr=0.5, blend=0.0, tta=False):
    """Test zamanı: tek görüntü için seçilmiş instance etiket haritasını düzelt.

    img: ham gri görüntü (2048x2048 uint8); prob: ensemble olasılığı (float 0..1); lab: etiket haritası;
    scores: {instance_id: seçici skoru} (skoru cut'ı geçmeyenler atılır); models: refiner listesi (cuda, eval).
    Dönüş: yeniden numaralandırılmış, çakışmasız etiket haritası.
    """
    ids = [i for i in np.unique(lab[lab > 0]) if scores.get(int(i), 0) > cut]
    if not ids:
        return np.zeros_like(lab, dtype=np.int32)
    norm = normalize(img)
    masks = [lab == i for i in ids]
    boxes = [crop_box(m) for m in masks]
    x = np.stack([make_input(norm, prob, m, b) for m, b in zip(masks, boxes)])
    probs = predict_crops(models, x, tta=tta)
    new, _ = paste_back(masks, boxes, [scores[int(i)] for i in ids], probs, prob, thr, blend)
    ks = np.unique(new[new > 0])
    remap = np.zeros(len(ids) + 1, np.int32)
    remap[ks] = np.arange(1, len(ks) + 1)
    return remap[new]


def load_refiners(run_dir, folds=range(5)):
    ms = {}
    for f in folds:
        p = Path(run_dir) / f"fold{f}" / "best.pt"
        if p.exists():
            m = make_model(None)
            m.load_state_dict(torch.load(p, map_location="cpu"))
            ms[f] = m.to(DEV).eval()
    return ms


# ---------------------------------------------------------------- eval

def eval_image(args):
    """Bir görüntünün varyantları -> {varyant: [(filament_id, rle), ...]}."""
    stem, fold, rles, boxes, scores, probs, thrs, blends = args
    masks = [mu.decode({"size": [2048, 2048], "counts": r.encode()}).astype(bool) for r in rles]
    base_prob = None
    if probs is not None and any(b > 0 for b in blends):
        base_prob = np.load(OOF / f"fold{fold}" / "val_probs" / f"{stem}.npy").astype(np.float32) / 255
    out, fb = {}, {}
    variants = [("identity", None, 0.5, 0.0)]
    if probs is not None:
        variants += [(f"refined_t{t}" + (f"_b{b}" if b else ""), probs, t, b) for t in thrs for b in blends]
    for name, p, t, b in variants:
        lab, fb[name] = paste_back(masks, boxes, scores, p, base_prob, t, b)
        out[name] = [(f"{stem}_{n}", mask_to_rle(lab == k)) for n, k in enumerate(np.unique(lab[lab > 0]), 1)]
    return out, fb


def cmd_eval(args):
    records = load_records()
    folds = get_folds()
    df = pd.read_pickle(OOF / "candidates.pkl")[["stem", "inst", "rle"]]
    models = {} if args.identity else load_refiners(args.run)
    res_rows, pred_rows, gt_parts, fb_tot = [], {}, [], {}
    for f in args.folds:
        if not args.identity and f not in models:
            warnings.warn(f"fold{f}: refiner modeli yok ({args.run}/fold{f}/best.pt), atlanıyor")
            continue
        stems = sorted(folds.loc[folds.fold == f, "stem"])[:args.limit or None]
        c = load_crops([f], args.crops)
        idx = np.nonzero(np.isin(c["stem"], stems) & (c["score"] > args.cut))[0]
        t0 = time.time()
        probs = None if args.identity else predict_crops([models[f]], crop_tensors(c, idx), args.bs, args.tta)
        key = pd.DataFrame({"stem": c["stem"][idx], "inst": c["inst"][idx], "k": np.arange(len(idx))})
        key = key.merge(df, on=["stem", "inst"], how="left")
        jobs = []
        for s, g in key.groupby("stem"):
            k = g.k.to_numpy()
            jobs.append((s, f, g.rle.tolist(), [tuple(b) for b in c["box"][idx[k]]], c["score"][idx[k]],
                         None if probs is None else probs[k], args.thr, args.blend))
        with ProcessPoolExecutor(16) as ex:
            for out, fb in ex.map(eval_image, jobs, chunksize=2):
                for name, rows in out.items():
                    pred_rows.setdefault(name, []).extend(rows)
                for name, v in fb.items():
                    fb_tot[name] = fb_tot.get(name, 0) + v
        # orijinal: seçilen adayların RLE'si doğrudan
        orig = key.assign(fid=key.stem + "_" + key.inst.astype(str))
        pred_rows.setdefault("original", []).extend(zip(orig.fid, orig.rle))
        gt_parts.append(gt_frame(records, stems))
        print(f"fold{f}: {len(stems)} görüntü, {len(idx)} seçilen aday, {time.time() - t0:.0f} sn", flush=True)
    if not gt_parts:
        return
    gt = pd.concat(gt_parts)
    for name in ["original"] + [n for n in pred_rows if n != "original"]:
        pred = pd.DataFrame(pred_rows[name], columns=["filament_id", "segmentation_rle"])
        res_rows.append(dict(variant=name, n_pred=len(pred), fallback=fb_tot.get(name, 0),
                             **compute_pq(gt, pred), **rubric_stats(gt, pred)))
    res = pd.DataFrame(res_rows)
    print(res.round(4).to_string(index=False))
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.save, index=False)
    print("->", args.save)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--limit", type=int, default=0, help="yalnızca ilk N görüntü (deneme)")
    b.add_argument("--out", default=str(REF))
    v = sub.add_parser("viz")
    v.add_argument("--out", default=str(REF))
    v.add_argument("--viz", default=str(OUT.parent / "viz" / "refiner_crops_check.jpg"))
    t = sub.add_parser("train")
    t.add_argument("--fold", type=int, default=0)
    t.add_argument("--epochs", type=int, default=25)
    t.add_argument("--bs", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--cut", type=float, default=0.24, help="val Dice'ı ayrıca seçilen adaylarda raporla")
    t.add_argument("--limit_steps", type=int, default=0, help="epoch başına en fazla adım (deneme)")
    t.add_argument("--workers", type=int, default=6)
    t.add_argument("--crops", default=str(REF))
    t.add_argument("--out", default=str(REF))
    e = sub.add_parser("eval")
    e.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    e.add_argument("--cut", type=float, default=0.24)
    e.add_argument("--thr", type=float, nargs="+", default=[0.5])
    e.add_argument("--blend", type=float, nargs="+", default=[0.0], help="ör. 0 0.5: ensemble ile karışım oranları")
    e.add_argument("--tta", type=int, default=0, help="1: flip TTA (4 yön)")
    e.add_argument("--bs", type=int, default=64)
    e.add_argument("--identity", action="store_true", help="model yok: orijinal maske geri yapıştırılır (doğrulama)")
    e.add_argument("--limit", type=int, default=0, help="fold başına ilk N görüntü (deneme)")
    e.add_argument("--run", default=str(REF))
    e.add_argument("--crops", default=str(REF))
    e.add_argument("--save", default=str(REF / "eval.csv"))
    args = ap.parse_args()
    dict(build=cmd_build, viz=cmd_viz, train=cmd_train, eval=cmd_eval)[args.cmd](args)


if __name__ == "__main__":
    main()
