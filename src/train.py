"""U-Net eğitimi: 2048 görüntülerden rastgele kesitler, validasyon tam 2048'de.

Örnek: python src/train.py --fold 0 --name unet_r34
"""
import argparse
import io
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from augment import cloud_field, make_aug, seed_worker
from contrast import PixelContrast
from copy_paste import PasteAug
from losses import dice_loss, dml_loss, sdf_target, skeleton_recall_loss, skeleton_target, tversky_loss  # noqa: F401 (refiner.py dice_loss'u buradan alır)
from data import EXT_IMG, OUT, TRAIN_IMG, gt_frame, get_folds, load_records, poly_to_mask, read_image
from metric import compute_pq, instances_to_rows
from post import prob_to_instances


def normalize(img):
    """Disk içindeki piksellere göre standartlaştır (istasyonlar arası kontrast farkı)."""
    x = img.astype(np.float32)
    disk = x[x > 30]
    m, s = (disk.mean(), disk.std()) if disk.size else (x.mean(), x.std())
    return (x - m) / (s + 1e-6)


def style_jitter(m):
    """Deney 49: anotatör çizim üslubu artırması. Maske 1/3 olasılıkla aynı kalır, 1/3 olasılıkla 1 px
    kalınlaştırılır, 1/3 olasılıkla 1 px inceltilir; inceltmede tamamen kaybolan filament olduğu gibi korunur."""
    r = random.random()
    if r < 1 / 3:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    u = m.astype(np.uint8)
    if r < 2 / 3:
        return cv2.dilate(u, k) > 0
    e = cv2.erode(u, k)
    n, lab = cv2.connectedComponents(u, connectivity=8)
    lost = np.setdiff1d(np.arange(1, n), np.unique(lab[e > 0]))
    return (e > 0) | np.isin(lab, lost)


def build_target(record, mode):
    """record: {annotator_image: [ann, ...]} -> float32 hedef maske."""
    masks = []
    for anns in record.values():
        m = np.zeros((2048, 2048), bool)
        for a in anns:
            m |= poly_to_mask(a["seg"])
        masks.append(m)
    if mode == "union":
        return np.any(masks, 0).astype(np.float32)
    if mode == "mean":
        return np.mean(masks, 0).astype(np.float32)
    if mode == "sqrt":  # deney 39: kısmi uzlaşıyı yukarı it (1/3 -> 0,58, 1/2 -> 0,71); tek anotatörde ortalamayla aynı
        return np.sqrt(np.mean(masks, 0)).astype(np.float32)
    if mode == "random":
        return masks[random.randrange(len(masks))].astype(np.float32)
    if mode == "mean_jitter":  # deney 49: her anotatör maskesi ayrı ayrı üslup artırmasından geçer
        return np.mean([style_jitter(m) for m in masks], 0).astype(np.float32)
    raise ValueError(mode)


def cleaned_mean_target(record, audit):
    """Etiket denetimine göre ortalama hedef: şüpheli bölgede o anotatör ortalamaya katılmaz.

    audit: {anotatör_öneki: (kaçırma_rle, sahte_rle)}. Hiçbir anotatörün geçerli olmadığı pikseller
    ağırlık 0 (yok say) alır. Etiket eklenmez/silinmez.
    """
    import pycocotools.mask as mu
    num = np.zeros((2048, 2048), np.float32)
    den = np.zeros((2048, 2048), np.float32)
    for ai, anns in record.items():
        m = np.zeros((2048, 2048), bool)
        for a in anns:
            m |= poly_to_mask(a["seg"])
        valid = np.ones((2048, 2048), bool)
        if ai.split("-")[0] in audit:
            omit, spur = audit[ai.split("-")[0]]
            valid &= ~(mu.decode(omit).astype(bool) | mu.decode(spur).astype(bool))
        num += m * valid
        den += valid
    tgt = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return tgt, (den > 0).astype(np.float32)


def annotator_index(records, min_images=5):
    """Anotatör öneki (ör. '010402') -> başlık indeksi; az görüntülü öneklerin hepsi ortak bir başlıkta."""
    from collections import Counter
    cnt = Counter(ai.split("-")[0] for rec in records.values() for ai in rec)
    common = sorted(p for p, c in cnt.items() if c >= min_images)
    idx = {p: i for i, p in enumerate(common)}
    rare = len(common)
    return {p: idx.get(p, rare) for p in cnt}, rare + 1


class TrainDS(Dataset):
    def __init__(self, stems, records, crop, target, repeats, pseudo=None, pseudo_n=0, aug="base",
                 year_weight=None, limb_weight=1.0, small_weight=1.0, size_loss=False, annot_map=None,
                 audit_dir=None, teacher_dir=None, teacher_alpha=1.0, aux_chir=False, aux_spine=False, paste=None):
        # yıl bazında çoğaltma: az temsil edilen yıllar (ör. 2019-2022) listeye tekrar eklenir
        year_weight = year_weight or {}
        stems = [s for s in stems for _ in range(year_weight.get(int(s[:4]), 1))]
        self.stems = stems * repeats
        self.records = records
        # instance düzeyinde kesit merkezi: (cy, cx, ağırlık); limb ve küçük filamentler ağırlıklandırılır
        self.limb_weight, self.small_weight = limb_weight, small_weight
        self.size_loss = size_loss
        self.inst = {}
        if limb_weight != 1.0 or small_weight != 1.0:
            for s in set(stems):
                cs = []
                for anns in records[s].values():
                    for a in anns:
                        p = np.asarray(a["seg"][0]).reshape(-1, 2)
                        cx, cy = p.mean(0)
                        area = cv2.contourArea(p.astype(np.float32))
                        w = (limb_weight if np.hypot(cx - 1024, cy - 1024) / 870 > 0.8 else 1.0)
                        w *= small_weight if area < 1000 else 1.0
                        cs.append((cy, cx, w))
                self.inst[s] = cs
        # pseudo: [(görüntü yolu, olasılık png yolu)], her epoch'ta rastgele pseudo_n tanesi
        self.pseudo = pseudo or []
        self.pseudo_n = pseudo_n if self.pseudo else 0
        self.crop = crop
        self.target = target
        self.skel_recall = False
        self.sdf = False
        self.audit_dir = audit_dir  # etiket denetimi klasörü (label_audit.py çıktısı)
        # damıtma (deney 10): hedef = alpha·anotatör hedefi + (1-alpha)·öğretmen olasılığı
        self.teacher_dir, self.teacher_alpha = teacher_dir, teacher_alpha
        self.annot_map = annot_map  # verilirse her anotatörün maskesi ayrı döndürülür (en fazla 3)
        # deney 26: kiralite yardımcı hedefi; Left / Right / Unidentifiable maskelerinin anotatör ortalaması
        self.aux_chir = aux_chir
        # deney 27: omurga yardımcı hedefi; 5 px kalınlığında çizilen omurga çizgilerinin anotatör ortalaması
        self.aux_spine = aux_spine
        self.aug_kind = aug
        self.aug = make_aug(aug)
        self.paste = PasteAug(sorted(set(stems)), records, lambda r: build_target(r, "mean"), **(paste or {}))  # deney 48

    def __len__(self):
        return len(self.stems) + self.pseudo_n

    def __getitem__(self, i):
        s, ign = None, None
        ann_masks, ann_idx = [], []
        if i < len(self.stems):
            s = self.stems[i]
            img = read_image(TRAIN_IMG / f"{s}.jpeg")
            if self.annot_map is not None:
                for ai, anns in self.records[s].items():
                    m = np.zeros((2048, 2048), bool)
                    for a in anns:
                        m |= poly_to_mask(a["seg"])
                    ann_masks.append(m.astype(np.float32))
                    ann_idx.append(self.annot_map[ai.split("-")[0]])
                tgt = np.mean(ann_masks, 0).astype(np.float32)
            elif self.audit_dir is not None and (self.audit_dir / f"{s}.pkl").exists():
                import pickle
                with open(self.audit_dir / f"{s}.pkl", "rb") as f:
                    tgt, ign = cleaned_mean_target(self.records[s], pickle.load(f))
            else:
                tgt = build_target(self.records[s], self.target)
            if self.aux_chir:
                cm = np.zeros((3, 2048, 2048), np.float32)
                for anns in self.records[s].values():
                    for a in anns:
                        k = min(int(a["cat"]), 3) - 1  # 4 (Ambiguous) -> Unidentifiable
                        cm[k] = np.maximum(cm[k], poly_to_mask(a["seg"]))  # aynı anotatörde çakışmaya karşı
                ann_masks = list(cm / len(self.records[s]))
            if self.aux_spine:
                sp = np.zeros((2048, 2048), np.float32)
                for anns in self.records[s].values():
                    m = np.zeros((2048, 2048), np.uint8)
                    for a in anns:
                        if a["spine"] and len(a["spine"]) >= 4:
                            pts = np.round(np.asarray(a["spine"], np.float64).reshape(-1, 2)).astype(np.int32)
                            cv2.polylines(m, [pts], False, 1, thickness=5)
                    sp += m
                ann_masks = [sp / len(self.records[s])]
            if self.teacher_dir is not None:
                t = np.load(self.teacher_dir / f"{s}.npy").astype(np.float32) / 255
                tgt = self.teacher_alpha * tgt + (1 - self.teacher_alpha) * t
        else:
            item = random.choice(self.pseudo)
            img = read_image(item[0])
            tgt = read_image(item[1]).astype(np.float32) / 255
            # isteğe bağlı 3. eleman: ağırlık haritası (0 = yok say, kararsız pseudo bölgeler)
            ign = read_image(item[2]).astype(np.float32) / 255 if len(item) > 2 else None
        c = self.crop
        # çoğunlukla bir filamentin etrafından, bazen diskten rastgele kesit
        if s in self.inst and self.inst[s] and random.random() < 0.7:
            cs = self.inst[s]
            cy, cx, _ = random.choices(cs, weights=[w for *_, w in cs])[0]
        else:
            samp = tgt[::8, ::8] ** 2 if self.target == "sqrt" else tgt[::8, ::8]  # sqrt hedefte örnekleyici ortalama hedefi kullanır
            ys, xs = np.nonzero(samp > 0.5)
            if len(ys) and random.random() < 0.7:
                k = random.randrange(len(ys))
                cy, cx = ys[k] * 8, xs[k] * 8
            else:
                cy, cx = random.randint(300, 1748), random.randint(300, 1748)
        y0 = int(np.clip(cy - c // 2 + random.randint(-c // 4, c // 4), 0, 2048 - c))
        x0 = int(np.clip(cx - c // 2 + random.randint(-c // 4, c // 4), 0, 2048 - c))
        img, tgt = img[y0:y0 + c, x0:x0 + c], tgt[y0:y0 + c, x0:x0 + c]
        img, tgt = self.paste(img, tgt, y0, x0, ok=s is not None and not ann_masks and ign is None)
        ann_masks = [m[y0:y0 + c, x0:x0 + c] for m in ann_masks]
        if ign is not None:
            ign = ign[y0:y0 + c, x0:x0 + c]
        # boyut-dengeli kayıp: küçük filament pikselleri daha ağır (sqrt(1000/alan), 1..3)
        wmap = np.ones_like(tgt)
        if self.size_loss and s is not None:
            n, lab, st, _ = cv2.connectedComponentsWithStats((tgt > 0).astype(np.uint8), connectivity=8)
            wk = np.clip(np.sqrt(1000.0 / np.maximum(st[:, cv2.CC_STAT_AREA], 1)), 1, 3)
            wk[0] = 1
            wmap = wk[lab].astype(np.float32)
        if ign is not None:
            wmap = wmap * ign
        out = self.aug(image=img, masks=[tgt, wmap] + ann_masks)
        if self.aug_kind == "strong" and random.random() < 0.3:
            out["image"] = cloud_field(out["image"])
        x = normalize(out["image"])[None]
        y, w = (m[None].astype(np.float32) for m in out["masks"][:2])
        if self.aux_spine:
            ya = (out["masks"][2][None].astype(np.float32) if s is not None else np.zeros((1, c, c), np.float32))
            ia = np.full(3, 0 if s is not None else -1, np.int64)
        elif self.aux_chir:
            ya = np.stack(out["masks"][2:5]).astype(np.float32) if s is not None else np.zeros((3, c, c), np.float32)
            ia = np.full(3, 0 if s is not None else -1, np.int64)  # dış/pseudo örnekte kiralite bilinmiyor
        elif self.annot_map is not None:
            ya = np.zeros((3, c, c), np.float32)
            ia = np.full(3, -1, np.int64)
            for k, (m, a) in enumerate(zip(out["masks"][2:], ann_idx)):
                ya[k], ia[k] = m, a
        elif self.skel_recall:  # deney 43: çoğunluk bölgesinin 3x3 kalınlaştırılmış iskeleti
            ya = skeleton_target(out["masks"][0] > 0.5)
            ia = np.full(3, -1, np.int64)
        elif self.sdf:  # deney 50: sınır uzaklığı yardımcı hedefi
            ya = sdf_target(out["masks"][0] > 0.5)
            ia = np.full(3, -1, np.int64)
        else:
            ya, ia = np.zeros((3, 1, 1), np.float32), np.full(3, -1, np.int64)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w), torch.from_numpy(ya), torch.from_numpy(ia)


def annot_loss(logits, ya, ia, lam, eps=1e-6):
    """Zhang ve ark. 2020 (NeurIPS) tarzı: gerçek olasılık p + anotatör başına piksel hata oranları.

    q_a = p (1 - fn_a) + (1 - p) fp_a ; fn, fp in [0, 0.5] (tanımlanabilirlik için).
    İz (trace) düzenlemesi lam * mean(2 - fn - fp): üslup farkları hata oranlarına aktarılır, p temiz kalır.
    """
    p = torch.sigmoid(logits[:, :1])
    total, n = 0.0, 0
    for k in range(ia.shape[1]):
        sel = ia[:, k] >= 0
        if not sel.any():
            continue
        a = ia[sel, k]
        h, w = logits.shape[2:]
        ch_fn = (1 + 2 * a).view(-1, 1, 1, 1).expand(-1, 1, h, w)
        fn = 0.5 * torch.sigmoid(torch.gather(logits[sel], 1, ch_fn))
        fp = 0.5 * torch.sigmoid(torch.gather(logits[sel], 1, ch_fn + 1))
        q = (p[sel] * (1 - fn) + (1 - p[sel]) * fp).clamp(eps, 1 - eps)
        y = ya[sel, k:k + 1]
        inter = (q * y).sum((2, 3))
        dice = 1 - ((2 * inter + 1) / (q.sum((2, 3)) + y.sum((2, 3)) + 1)).mean()
        total = total + F.binary_cross_entropy(q, y) + dice + lam * (2 - fn - fp).mean()
        n += 1
    return total / max(n, 1)


def load_state(path):
    """state_dict'i okur; dosya repoda parçalara bölünmüşse (path.part00, part01, ...) parçaları bellekte birleştirir."""
    path = Path(path)
    if path.exists():
        return torch.load(path, map_location="cpu")
    parts = sorted(path.parent.glob(path.name + ".part*"))
    if not parts:
        raise SystemExit(f"{path} yok (parçaları da yok)")
    return torch.load(io.BytesIO(b"".join(p.read_bytes() for p in parts)), map_location="cpu")


def load_model(fd):
    """Fold klasöründen (args.json + best.pt) değerlendirme modunda model; fp16 kaydedilmiş ağırlıkları float'a çevirir."""
    a = json.loads((Path(fd) / "args.json").read_text())
    m = getattr(smp, a["arch"])(a["encoder"], encoder_weights=None, in_channels=1, classes=a.get("n_out", 1))
    sd = load_state(Path(fd) / "best.pt")
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
    return m.cuda().eval()


@torch.no_grad()
def predict(model, img, tta=False):
    """Tam 2048 görüntü için olasılık haritası (float32, 2048x2048)."""
    x = torch.from_numpy(normalize(img))[None, None].cuda()
    flips = [[], [3], [2], [2, 3]] if tta else [[]]
    transposes = [False, True] if tta == 2 else [False]  # tta=2: D4 (8 yön)
    acc, n = 0, 0
    with torch.autocast("cuda", dtype=torch.float16):
        for t in transposes:
            xt = x.transpose(2, 3) if t else x
            for f in flips:
                xi = xt.flip(f) if f else xt
                p = torch.sigmoid(model(xi)[:, :1].float())  # anotatör modelinde 0. kanal = gerçek olasılık
                p = p.flip(f) if f else p
                acc = acc + (p.transpose(2, 3) if t else p)
                n += 1
    return (acc / n)[0, 0].cpu().numpy()


def evaluate(model, stems, gt_df, post_cfg, save_dir=None):
    model.eval()
    rows = []
    for s in stems:
        prob = predict(model, read_image(TRAIN_IMG / f"{s}.jpeg"))
        if save_dir is not None:
            np.save(save_dir / f"{s}.npy", (prob * 255).astype(np.uint8))
        rows += instances_to_rows(s, prob_to_instances(prob, **post_cfg))
    pred = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    return compute_pq(gt_df, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--name", default="unet_r34")
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--contrast", type=float, default=0.0, help=">0: görüntüler arası piksel kontrast kaybı ağırlığı (deney 44)")
    ap.add_argument("--skel_recall", type=float, default=0.0, help=">0: Skeleton Recall Loss ağırlığı (deney 43)")
    ap.add_argument("--ssl_init", default=None, help="öz-denetimli ön eğitim ağırlıkları (experiments/ssl_pretrain.py, deney 51)")
    ap.add_argument("--drop_path", type=float, default=0.0, help="kodlayıcı stochastic depth oranı (deney 52)")
    ap.add_argument("--aux_sdf", type=float, default=0.0, help=">0: sınır uzaklığı yardımcı kanalı ve L1 ağırlığı (deney 50)")
    ap.add_argument("--enc_lr_mult", type=float, default=1.0, help="kodlayıcı öğrenme hızı çarpanı (deney 36)")
    ap.add_argument("--grad_ckpt", type=int, default=0, help="1: kodlayıcıda gradient checkpointing (bellek için)")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--target", default="union")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--eval_from", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--pseudo_dir", default=None, help="fold'a özel pseudo-label klasörü kökü (içinde foldN/)")
    ap.add_argument("--pseudo_n", type=int, default=0, help="epoch başına pseudo/dış örnek sayısı")
    ap.add_argument("--ext_pairs", default=None, help="dış etiketli veri CSV'si (img,mask sütunları)")
    ap.add_argument("--aug", default="base", choices=["base", "rot360", "strong"])
    ap.add_argument("--year_weight", default="", help="ör. '2019:3,2020:3,2021:3,2022:3' (yıl:tekrar)")
    ap.add_argument("--limb_weight", type=float, default=1.0, help="r/R>0.8 filamentlerin kesit merkezi ağırlığı")
    ap.add_argument("--small_weight", type=float, default=1.0, help="alanı <1000 px filamentlerin kesit merkezi ağırlığı")
    ap.add_argument("--size_loss", type=int, default=0, help="1: küçük filament piksellerine ağır BCE")
    ap.add_argument("--train_frac", type=float, default=1.0, help="öğrenme eğrisi için train alt kümesi")
    ap.add_argument("--seed", type=int, default=0, help="rastgelelik tohumu (tohum çeşitliliği deneyleri)")
    ap.add_argument("--ema", type=float, default=0.0, help=">0 ise ağırlıkların üstel ortalaması (ör. 0.999); değerlendirme/kayıt EMA ile")
    ap.add_argument("--loss", default="bce_dice", choices=["bce_dice", "bce_lovasz", "bce_dice_lovasz", "bce_dml", "bce_tversky"])
    ap.add_argument("--teacher_dir", default=None, help="öğretmen kökü (içinde foldN/); make_teacher.py çıktısı")
    ap.add_argument("--teacher_alpha", type=float, default=0.5, help="anotatör hedefinin ağırlığı")
    ap.add_argument("--audit_dir", default=None, help="etiket denetimi klasörü; verilirse temizlenmiş ortalama hedef")
    ap.add_argument("--annot_lambda", type=float, default=0.01, help="annot hedefi: iz düzenlemesi katsayısı")
    ap.add_argument("--annot_anchor", type=float, default=0.3, help="annot hedefi: p'nin ortalama hedefe bağlanma ağırlığı")
    ap.add_argument("--aux_chir", type=float, default=0.0, help=">0: kiralite yardımcı başlığı (3 kanal) ve kayıp ağırlığı (deney 26)")
    ap.add_argument("--aux_spine", type=float, default=0.0, help=">0: omurga yardımcı başlığı (1 kanal) ve kayıp ağırlığı (deney 27)")
    ap.add_argument("--paste_p", type=float, default=0.0, help="deney 48: kesit başına kopyala-yapıştır olasılığı")
    ap.add_argument("--paste_n", type=int, default=2, help="deney 48: ortalama yapıştırılan filament sayısı (1..2n-1)")
    ap.add_argument("--init", default=None, help="başlangıç ağırlıkları (ör. pseudo ön eğitimi best.pt)")
    args = ap.parse_args()
    if args.aux_chir > 0:  # hedef, anotatörler arası oy yerine birleşimi anotatör sayısına bölüyor
        raise SystemExit("--aux_chir: kiralite hedefi hatalı üretiliyor; düzeltilmeden kullanılamaz")

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    run = OUT / args.name / f"fold{args.fold}"
    run.mkdir(parents=True, exist_ok=True)

    records = load_records()
    annot_map, n_annot = (annotator_index(records) if args.target == "annot" else (None, 0))
    args.n_out = 1 + 2 * n_annot + (3 if args.aux_chir > 0 else 0) + (1 if args.aux_spine > 0 else 0) + (1 if args.aux_sdf > 0 else 0)
    folds = get_folds()
    tr = folds.loc[folds.fold != args.fold, "stem"].tolist()
    if args.train_frac < 1.0:
        tr = sorted(random.Random(1).sample(tr, int(len(tr) * args.train_frac)))
        print(f"train alt kümesi: {len(tr)} görüntü", flush=True)
    year_weight = {int(k): int(v) for k, v in (x.split(":") for x in args.year_weight.split(",") if x)}
    va = folds.loc[folds.fold == args.fold, "stem"].tolist()
    # run manifesti: OOF bu listelerle çakışmayı denetler
    manifest = dict(fold_scheme=os.environ.get("FOLD_SCHEME", "image"), train_stems=tr, val_stems=va)
    (run / "args.json").write_text(json.dumps({**vars(args), **manifest}, indent=1))
    gt_va = gt_frame(records, va)

    pseudo = None
    if args.pseudo_dir:
        pdir = Path(args.pseudo_dir) / f"fold{args.fold}"
        pseudo = [(EXT_IMG / f"{p.stem}.jpeg", p) for p in sorted(pdir.glob("*.png"))]
        print(f"{len(pseudo)} pseudo-label görüntüsü", flush=True)
    if args.ext_pairs:
        ext = pd.read_csv(args.ext_pairs.format(fold=args.fold))
        if "date" in ext.columns:
            # bu fold'un validasyon günlerinin +-3 günündeki dış kareler atılır (aynı filamentler -> CV şişmesi)
            to_date = lambda s: pd.Timestamp(f"{s[:4]}-{s[4:6]}-{s[6:8]}")
            va_days = pd.Series([to_date(s) for s in va])
            dates = pd.to_datetime(ext["date"])
            near = dates.apply(lambda d: (va_days - d).abs().min() <= pd.Timedelta(days=3))
            ext = ext[~near]
            print(f"validasyon günlerine yakın {int(near.sum())} dış kare atıldı", flush=True)
        cols = ["img", "mask"] + (["weight"] if "weight" in ext.columns else [])
        pseudo = list(ext[cols].itertuples(index=False, name=None))
        print(f"{len(pseudo)} dış etiketli görüntü", flush=True)
    ds = TrainDS(tr, records, args.crop, args.target, args.repeats, pseudo, args.pseudo_n, args.aug,
                 year_weight, args.limb_weight, args.small_weight, bool(args.size_loss), annot_map,
                 Path(args.audit_dir) if args.audit_dir else None,
                 Path(args.teacher_dir) / f"fold{args.fold}" if args.teacher_dir else None, args.teacher_alpha,
                 args.aux_chir > 0, args.aux_spine > 0, dict(paste_p=args.paste_p, paste_n=args.paste_n))
    ds.skel_recall = args.skel_recall > 0
    ds.sdf = args.aux_sdf > 0
    dl = DataLoader(ds, batch_size=args.bs,
                    shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True,
                    persistent_workers=True, worker_init_fn=seed_worker)
    enc_kw = dict(drop_path_rate=args.drop_path) if args.drop_path > 0 else {}  # deney 52: kodlayıcıda rastgele blok atlama
    model = getattr(smp, args.arch)(args.encoder, encoder_weights="imagenet", in_channels=1, classes=args.n_out, **enc_kw).cuda()
    if args.ssl_init:  # deney 51: öz-denetimli ön eğitim ağırlıkları; segmentasyon başlığı yeniden başlar
        own = model.state_dict()
        sd = {k: v for k, v in torch.load(args.ssl_init, map_location="cpu").items()
              if k in own and v.shape == own[k].shape and not k.startswith("segmentation_head")}
        model.load_state_dict(sd, strict=False)
        print(f"ön eğitim ağırlıkları: {len(sd)}/{len(own)} tensör ({args.ssl_init})", flush=True)
    if args.grad_ckpt:  # kodlayıcıda gradient checkpointing: büyük kodlayıcılar bs 2 ile sığsın (deney 30)
        model.encoder.model.set_grad_checkpointing(True)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cuda"))
    ema = None
    if args.ema > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)
    lovasz = smp.losses.LovaszLoss(mode="binary", per_image=True) if "lovasz" in args.loss else None
    enc = list(model.encoder.parameters())  # deney 36: kodlayıcıya ayrı (düşük) öğrenme hızı
    rest = [q for q in model.parameters() if all(q is not e for e in enc)]
    lrs = [args.lr * args.enc_lr_mult, args.lr]
    groups = [{"params": enc, "lr": lrs[0]}, {"params": rest, "lr": lrs[1]}]
    pc = PixelContrast().cuda().attach(model) if args.contrast > 0 else None  # deney 44; best.pt'ye girmez
    if pc is not None:
        groups.append({"params": pc.parameters(), "lr": args.lr}), lrs.append(args.lr)
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    total = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lrs, total_steps=total, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    post_cfg = dict(thr=0.5, min_area=200, join=0)

    best, log = -1, []
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y, w, ya, ia in dl:
            x, y, w = x.cuda(non_blocking=True), y.cuda(non_blocking=True), w.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            logits = logits.float()
            p_logit = logits[:, :1]
            loss = F.binary_cross_entropy_with_logits(p_logit, y, weight=w)
            if pc is not None:
                loss = loss + args.contrast * pc(pc.dec, y, p_logit)
            if args.loss in ("bce_dice", "bce_dice_lovasz"):
                loss = loss + dice_loss(p_logit, y, (w > 0).float())
            elif args.loss == "bce_dml":
                loss = loss + dml_loss(p_logit, y, (w > 0).float())
            if args.skel_recall > 0:
                loss = loss + args.skel_recall * skeleton_recall_loss(p_logit, ya.cuda(non_blocking=True))
            if args.loss == "bce_tversky":
                loss = loss + tversky_loss(p_logit, y, (w > 0).float())
            if lovasz is not None:  # IoU'yu doğrudan optimize eder; yumuşak hedef 0.5'te ikilileştirilir
                loss = loss + lovasz(p_logit, (y > 0.5).float())
            if args.aux_chir > 0:
                ok = (ia[:, 0] >= 0).cuda(non_blocking=True)
                if ok.any():
                    loss = loss + args.aux_chir * F.binary_cross_entropy_with_logits(
                        logits[ok, 1:4], ya.cuda(non_blocking=True)[ok])
            if args.aux_spine > 0:
                ok = (ia[:, 0] >= 0).cuda(non_blocking=True)
                if ok.any():
                    loss = loss + args.aux_spine * F.binary_cross_entropy_with_logits(
                        logits[ok, 1:2], ya.cuda(non_blocking=True)[ok])
            if args.aux_sdf > 0:
                loss = loss + args.aux_sdf * F.l1_loss(torch.tanh(logits[:, -1:]), ya.cuda(non_blocking=True))
            if annot_map is not None:
                loss = args.annot_anchor * loss + annot_loss(logits, ya.cuda(non_blocking=True),
                                                             ia.cuda(non_blocking=True), args.annot_lambda)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)
            tot += loss.item()
        entry = dict(epoch=ep, loss=tot / len(dl), sec=round(time.time() - t0))
        if ep + 1 >= args.eval_from and (ep % 2 == 1 or ep == args.epochs - 1):
            eval_model = ema.module if ema is not None else model
            entry.update(evaluate(eval_model, va, gt_va, post_cfg))
            if entry["pq"] > best:
                best = entry["pq"]
                torch.save(eval_model.state_dict(), run / "best.pt")
        log.append(entry)
        print(json.dumps({k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
                          for k, v in entry.items()}), flush=True)
        pd.DataFrame(log).to_csv(run / "log.csv", index=False)

    if best < 0:
        # validasyonsuz koşu (ör. dış veriyle ön eğitim): son epoch kaydedilir, fold bilgisi sızmaz
        torch.save((ema.module if ema is not None else model).state_dict(), run / "best.pt")
        return
    # en iyi modelle validasyon olasılık haritalarını kaydet (post-processing ayarı için)
    model.load_state_dict(torch.load(run / "best.pt"))
    probs = run / "val_probs"
    probs.mkdir(exist_ok=True)
    res = evaluate(model, va, gt_va, post_cfg, save_dir=probs)
    print("best", json.dumps({k: float(v) for k, v in res.items()}))


if __name__ == "__main__":
    main()
