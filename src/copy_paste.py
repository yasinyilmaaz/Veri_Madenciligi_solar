"""Deney 48: oran tabanlı, r/R koruyan filament kopyala-yapıştır (01.11).

Banka YALNIZ verilen (bu fold'un eğitim) görüntülerinden kurulur. Bir filament yaması, kaynaktaki filamentsiz
arka plana bölünerek oran haritasına çevrilir ve hedef kesitte çarpan olarak uygulanır: I' = I·(1 + α(oran − 1)).
Böylece istasyon parlaklığı ve limb kararması taşınmaz, yalnız göreli kararma taşınır. Yama kaynaktaki disk
merkezi uzaklığında (±0,02 R) kalır ve disk merkezi etrafında döndürülür. Hedef = max(hedef, yamanın yumuşak
anotatör ortalaması): kaynağın hedef değerleri aynen taşınır. Bu bir yumuşak hedefli sentetik artırma
varsayımıdır: hedef görüntüyü çizen anotatörlerin yapıştırılan yapıya aynı oyu vereceği doğrulanmaz. Orta kontrastlı yamalar
(çekirdek oranı ≥ 0,85) 2 kat sık seçilir; kararması olmayan (> 0,95) yamalar bankaya alınmaz.

Kontrol: python src/copy_paste.py --fold 0 --n 150   (CPU; K1 banka, K2 kontrast sadakati, K3 dikiş izi,
K5 r/R, K6 hedef değerlerinin korunması)
"""
import math
import random

import cv2
import numpy as np

from data import TRAIN_IMG, read_image

CY = CX = 1023.5  # disk merkezi (yaklaşık); R = 905 px
R_DISK = 905.0
K5, K9 = np.ones((5, 5), np.uint8), np.ones((9, 9), np.uint8)


class PasteAug:
    """TrainDS'e bağlanır; paste_p = 0 iken banka kurulmaz ve kesit aynen döner."""

    def __init__(self, stems, records, target_fn, paste_p=0.0, paste_n=2):
        self.paste_p, self.paste_n = paste_p, paste_n
        self.bank, self.wts = build_paste_bank(stems, records, target_fn) if paste_p > 0 else ([], [])

    def __call__(self, img, tgt, y0, x0, ok=True):
        if not (self.bank and ok and random.random() < self.paste_p):
            return img, tgt
        n = random.randint(1, 2 * self.paste_n - 1)
        return paste_filaments(img, tgt.copy(), self.bank, self.wts, n, y0, x0)


def background(img, fil):
    """Filamentsiz yerel arka plan: etiketli pikseller 9 px genişletilip Telea inpaint, sonra 31 px medyan (m4)."""
    return cv2.medianBlur(cv2.inpaint(img, cv2.dilate(fil, K9), 5, cv2.INPAINT_TELEA), 31).astype(np.float32)


def core_ratio(ratio, m):
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    return float(ratio[dt >= max(dt.max() * 0.5, 1)].mean())


def build_paste_bank(stems, records, target_fn, pad=6, max_side=256, max_core=0.95, mid=0.85, mid_w=2.0):
    """Döner: (banka, ağırlıklar). Yama = (oran, alfa, yumuşak hedef, r, θ, çekirdek oranı)."""
    bank, wts = [], []
    for s in stems:
        img = read_image(TRAIN_IMG / f"{s}.jpeg")
        tgt = target_fn(records[s])
        fil = (tgt > 0).astype(np.uint8)
        bg = background(img, fil)
        n, lab, st, cen = cv2.connectedComponentsWithStats(fil, connectivity=8)
        for k in range(1, n):
            x, y, w, h, a = st[k]
            if a < 50 or max(w, h) + 2 * pad > max_side:
                continue
            y0, x0, y1, x1 = max(y - pad, 0), max(x - pad, 0), min(y + h + pad, 2048), min(x + w + pad, 2048)
            if (bg[y0:y1, x0:x1] < 30).any():  # disk kenarını aşan yama alınmaz
                continue
            m = (lab[y0:y1, x0:x1] == k).astype(np.uint8)
            ratio = np.clip(img[y0:y1, x0:x1] / (bg[y0:y1, x0:x1] + 1), 0.3, 1.5).astype(np.float32)
            cr = core_ratio(ratio, m)
            if cr > max_core:  # kararması olmayan etiketli yama alınmaz
                continue
            alpha = cv2.GaussianBlur(cv2.dilate(m, K5).astype(np.float32), (5, 5), 0)  # 7×7 ve 9×9 K3'ü değiştirmedi (1,19 / 1,18)
            dy, dx = cen[k][1] - CY, cen[k][0] - CX
            bank.append((ratio.astype(np.float16), alpha.astype(np.float16),
                         (tgt[y0:y1, x0:x1] * m).astype(np.float16), math.hypot(dy, dx), math.atan2(dy, dx), cr))
            wts.append(mid_w if cr >= mid else 1.0)
    return bank, wts


def _rot(a, deg, size, interp=cv2.INTER_LINEAR):
    """Yamayı merkezi etrafında deg derece döndürüp size x size tuvale koyar (dışı 0)."""
    M = cv2.getRotationMatrix2D(((a.shape[1] - 1) / 2, (a.shape[0] - 1) / 2), deg, 1.0)
    M[0, 2] += (size - a.shape[1]) / 2
    M[1, 2] += (size - a.shape[0]) / 2
    return cv2.warpAffine(a, M, (size, size), flags=interp, borderValue=0)


def paste_filaments(img, tgt, bank, wts, n, y0, x0, gap=15, tries=12, log=None):
    """Kesite (y0, x0 ofsetli) n filament yapıştırır: img *= 1 + α(oran − 1); hedef = max(hedef, yama).
    Yama kaynaktaki r'de (±0,02 R) kalır ve disk merkezi etrafında döndürülür: limb kısalması ve
    radyal yönelim korunur. Yalnız disk içine (I >= 30) ve mevcut hedefin gap px uzağına."""
    c = img.shape[0]
    busy = cv2.dilate((tgt > 0).astype(np.uint8), np.ones((2 * gap + 1, 2 * gap + 1), np.uint8))
    out = img.astype(np.float32)
    for ratio, alpha, pt, r, th, cr in random.choices(bank, weights=wts, k=n):
        size = int(math.ceil(math.hypot(*ratio.shape))) | 1
        for _ in range(tries):
            th2 = random.uniform(-math.pi, math.pi)
            r2 = min(r + random.uniform(-0.02, 0.02) * R_DISK, 0.98 * R_DISK)
            yy = int(round(CY + r2 * math.sin(th2) - y0 - size / 2))
            xx = int(round(CX + r2 * math.cos(th2) - x0 - size / 2))
            if yy < 0 or xx < 0 or yy + size > c or xx + size > c:
                continue
            deg = -math.degrees(th2 - th)  # cv2: pozitif açı ekranda saat yönünün tersi (y aşağı)
            ra, aa = (_rot(v.astype(np.float32), deg, size) for v in (ratio, alpha))
            # hedef en yakın komşuyla: anotatör ortalaması değerleri (1/3, 1/2, 1 ...) ara değere kaymasın (K6)
            pa = _rot(pt.astype(np.float32), deg, size, cv2.INTER_NEAREST)
            on = aa > 0.01
            win = (slice(yy, yy + size), slice(xx, xx + size))
            if busy[win][on].any() or (img[win][on] < 30).any():
                continue
            out[win] *= 1 + aa * (ra - 1)
            tgt[win] = np.maximum(tgt[win], pa)
            busy[win][on] = 1
            if log is not None:
                log.append((yy, xx, size, deg, r, r2, cr, pt))
            break
    return np.clip(out, 0, 255).astype(np.uint8), tgt


if __name__ == "__main__":  # CPU kontrolleri K1–K6
    import argparse
    import time

    from data import get_folds, load_records
    from train import build_target
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--bank_imgs", type=int, default=0, help=">0: hız için banka bu kadar görüntüden")
    a = ap.parse_args()
    random.seed(0), np.random.seed(0)
    recs, folds = load_records(), get_folds()
    tr = sorted(folds.loc[folds.fold != a.fold, "stem"])
    t = time.time()
    bank, wts = build_paste_bank(tr[:a.bank_imgs] if a.bank_imgs else tr, recs, lambda r: build_target(r, "mean"))
    crs = np.array([b[5] for b in bank])
    print(f"K1 banka: {len(bank)} yama, {time.time() - t:.0f} sn; orta kontrast (>=0,85) payı {np.mean(crs >= .85):.2f}")
    dev, gr, rr, kv, km = [], [], [], [], []
    for _ in range(a.n):
        s = random.choice(tr)
        img, tgt0 = read_image(TRAIN_IMG / f"{s}.jpeg"), build_target(recs[s], "mean")
        y0, x0 = random.randint(0, 512), random.randint(0, 512)
        cimg, ctgt = img[y0:y0 + 1536, x0:x0 + 1536], tgt0[y0:y0 + 1536, x0:x0 + 1536]
        log = []
        pimg, ptgt = paste_filaments(cimg, ctgt.copy(), bank, wts, 3, y0, x0, log=log)
        bg = background(pimg, (ptgt > 0).astype(np.uint8))
        rat = pimg / (bg + 1)
        sob = np.hypot(cv2.Sobel(pimg.astype(np.float32), cv2.CV_32F, 1, 0), cv2.Sobel(pimg.astype(np.float32), cv2.CV_32F, 0, 1))
        nat = (ctgt > 0).astype(np.uint8)
        nring = (cv2.dilate(nat, K5) > 0) & (nat == 0)
        for yy, xx, size, deg, r, r2, cr, src_pt in log:
            win = (slice(yy, yy + size), slice(xx, xx + size))
            m = ((ptgt[win] > 0) & (ctgt[win] == 0)).astype(np.uint8)
            if m.sum() < 30:
                continue
            dev.append(core_ratio(np.clip(rat[win], 0.3, 1.5), m) - cr)
            ring = (cv2.dilate(m, K5) > 0) & (m == 0)
            if nring.any():
                gr.append(sob[win][ring].mean() / max(sob[nring].mean(), 1e-3))
            rr.append(abs(r2 - r) / R_DISK)
            pa = _rot(src_pt.astype(np.float32), deg, size, cv2.INTER_NEAREST)  # bu yamanın kendi hedefi
            sv, pv = np.unique(src_pt[src_pt > 0].astype(np.float32)), ptgt[win][pa > 0]
            kv.append(np.isin(pv, sv).mean())  # yapıştırılan değerlerin kaynakta var olan uzlaşı düzeylerinden olma payı
            km.append(pv.mean() - src_pt[src_pt > 0].astype(np.float32).mean())
    dev = np.array(dev)
    print(f"K2 yapıştırılan çekirdek oranı − kaynak: ort {dev.mean():+.3f}, |fark| p90 {np.percentile(np.abs(dev), 90):.3f} (n={len(dev)}; kabul |ort| ≤ 0,02, p90 ≤ 0,05)")
    print(f"K3 kenar gradyanı (yapıştırılan / gerçek halka): medyan {np.median(gr):.2f}, p10–p90 {np.percentile(gr, 10):.2f}–{np.percentile(gr, 90):.2f} (kabul medyan 0,90–1,15)")
    print(f"K5 |Δr|/R en büyük {max(rr):.3f} (kabul ≤ 0,02)")
    print(f"K6 yeni uzlaşı düzeyi üretilmiyor: yapıştırılan değerlerin kaynakta var olma payı en küçük {min(kv):.4f} (kabul 1,0); "
          f"ortalama hedef farkı ort {np.mean(km):+.4f}, |fark| p90 {np.percentile(np.abs(km), 90):.4f} (kabul |ort| ≤ 0,02)")
