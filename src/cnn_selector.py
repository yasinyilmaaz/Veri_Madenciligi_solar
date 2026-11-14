"""Deney 33: görüntü tabanlı aday kalite ağı (CNN seçici). Aday üretimi ve maskeler sabit.

Her aday için bağlamlı bir kesit (görüntü, ensemble olasılığı, aday maskesi; 3 kanal) çıkarılır.
Küçük bir CNN (timm resnet18, ImageNet), LightGBM seçiciyle aynı hedefi öğrenir:
y = anotatörler üzerinden IoU·[IoU>0,5] ortalaması (yumuşak BCE).
İç içe CV: fold f adaylarının skoru, f dışındaki dört fold'un adaylarıyla eğitilen CNN'den gelir.
Raporlanan: LightGBM / CNN / karışım (a·lgb + (1-a)·cnn) için kesme taraması, fold bazında PQ.
Birincil karşılaştırma tüm havuzu yeniden seçme; "yalnız eleme" ve "yalnız ekleme" tanı amaçlı.
Örnek: python src/cnn_selector.py --oof outputs/oof/ens1536_tta2
"""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mu
import timm
import torch
import torch.nn.functional as F

from artifacts import load_candidates_scored, merge_exact
from data import TRAIN_IMG, get_folds, gt_frame, load_records, read_image
from metric import compute_pq
from train import normalize

SIDE = 160


def crop_one(img, prob, m, scale=1.0):
    """img, prob: uint8 2048²; m: 0/1 aday maskesi -> (3, SIDE, SIDE) uint8 bağlamlı kesit.
    scale > 1: aynı merkezde daha geniş bağlam penceresi (CNN v2)."""
    ys, xs = np.nonzero(m)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    s = int(scale * np.clip(1.6 * max(np.ptp(ys), np.ptp(xs)) + 48, 96, 768)) // 2
    y0, x0 = cy - s, cx - s
    win = np.zeros((3, 2 * s, 2 * s), np.uint8)
    a0, b0 = max(y0, 0), max(x0, 0)
    a1, b1 = min(cy + s, 2048), min(cx + s, 2048)
    for c, src in enumerate((img, prob, m.astype(np.uint8) * 255)):
        win[c, a0 - y0:a1 - y0, b0 - x0:b1 - x0] = src[a0:a1, b0:b1]
    return np.stack([cv2.resize(w, (SIDE, SIDE), interpolation=cv2.INTER_AREA) for w in win])


def norm_img(path):
    return np.clip(normalize(read_image(path)) * 32 + 128, 0, 255).astype(np.uint8)


def crops_for_image(args):
    stem, prob_path, rows, ctx = args
    img, prob = norm_img(TRAIN_IMG / f"{stem}.jpeg"), np.load(prob_path)
    out = []
    for inst, rle in rows:
        m = mu.decode({"size": [2048, 2048], "counts": rle.encode()})
        c = crop_one(img, prob, m)
        out.append((inst, np.concatenate([c, crop_one(img, prob, m, ctx)]) if ctx > 1 else c))
    return stem, out


def crops_key(df, oof_dir, ctx):
    """Kesit önbelleğinin anahtarı: aday üretim manifesti + kesit bağlamı + her adayın (stem, inst, rle)'si.
    (stem, inst) aynı kalıp geometri değişirse eski kesitler kullanılmasın."""
    h = hashlib.sha1((oof_dir / "candidates_meta.json").read_bytes() + f"ctx={ctx:g}".encode())
    for r in df.sort_values(["stem", "inst"]).itertuples():
        h.update(f"{r.stem}|{r.inst}|{r.rle}\n".encode())
    return h.hexdigest()


def build_cache(df, oof_dir, folds, cache, ctx=1.0):
    jobs = [(s, oof_dir / f"fold{folds[s]}" / "val_probs" / f"{s}.npy", list(zip(g.inst, g.rle)), ctx)
            for s, g in df.groupby("stem")]
    X, keys = [], []
    with ProcessPoolExecutor(8) as ex:
        for stem, out in ex.map(crops_for_image, jobs, chunksize=4):
            for inst, c in out:
                keys.append((stem, inst))
                X.append(c)
    np.savez(cache, X=np.stack(X), stem=np.array([k[0] for k in keys]), inst=np.array([k[1] for k in keys]),
             key=crops_key(df, oof_dir, ctx))


def d4(x, k):
    x = torch.rot90(x, k % 4, (2, 3))
    return x.flip(3) if k >= 4 else x


AUX_W = 0.0  # >0: ikinci çıkış eşiksiz ortalama IoU'yu öğrenir (yalnız eğitimde)
PROB_T = 0.0  # >0: eğitimde olasılık kanalına örnek başına sıcaklık T ~ exp(U(-PROB_T, PROB_T))


def train_fold(X, y, Xva, epochs, seed, keep_model=False):
    torch.manual_seed(seed)
    y = y.reshape(len(y), -1)  # sütun 0: y; varsa sütun 1: eşiksiz ortalama IoU (yardımcı hedef)
    m = timm.create_model("resnet18", pretrained=True, in_chans=X.shape[1], num_classes=y.shape[1]).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    n, bs = len(X), 64
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=epochs * ((n + bs - 1) // bs))
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y.astype(np.float32))
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = d4(Xt[idx].cuda().float() / 255, int(torch.randint(8, (1,))))
            if PROB_T > 0:
                t = torch.exp((torch.rand(len(idx), 1, 1, device="cuda") * 2 - 1) * PROB_T)
                q = xb[:, 1].clamp(1e-3, 1 - 1e-3)
                xb[:, 1] = torch.sigmoid(torch.logit(q) / t)
            with torch.autocast("cuda", dtype=torch.float16):
                out = m(xb).float()
            yb = yt[idx].cuda()
            loss = F.binary_cross_entropy_with_logits(out[:, 0], yb[:, 0])
            if y.shape[1] > 1:
                loss = loss + AUX_W * F.binary_cross_entropy_with_logits(out[:, 1], yb[:, 1])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    m.eval()
    return m if keep_model else predict_cnn(m, Xva)


def predict_cnn(m, Xva):
    preds = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        Xv = torch.from_numpy(Xva)
        for i in range(0, len(Xv), 256):
            xb = Xv[i:i + 256].cuda().float() / 255
            preds.append(sum(torch.sigmoid(m(d4(xb, k))[:, 0].float()) for k in range(8)).cpu().numpy() / 8)
    return np.concatenate(preds)


def _iou_mean(args):
    stem, rows = args
    from data import poly_to_rle
    P = [{"size": [2048, 2048], "counts": r.encode()} for r in rows]
    acc, recs = np.zeros(len(P)), load_records()[stem]
    for anns in recs.values():
        G = [poly_to_rle(a["seg"]) for a in anns]
        if G:
            acc += np.asarray(mu.iou(P, G, [0] * len(G))).reshape(len(P), len(G)).max(1)
    return stem, acc / len(recs)


def iou_mean_table(df, oof_dir):
    """Her aday için anotatör başına en iyi IoU'nun eşiksiz ortalaması (önbellekli)."""
    path = oof_dir / "cand_iou_mean.csv"
    if not path.exists():
        groups = [(s, list(g.rle)) for s, g in df.groupby("stem")]
        insts = {s: list(g.inst) for s, g in df.groupby("stem")}
        rows = []
        with ProcessPoolExecutor(8) as ex:
            for s, v in ex.map(_iou_mean, groups, chunksize=4):
                rows += [(s, i, x) for i, x in zip(insts[s], v)]
        pd.DataFrame(rows, columns=["stem", "inst", "iou_mean"]).to_csv(path, index=False)
    return pd.read_csv(path)


def report(df, gt, folds, col, cuts, mask=None):
    gfold = gt.filament_id.str.split("-", n=1).str[1].str.rsplit("_", n=1).str[0].map(folds)
    best = None
    for c in cuts:
        keep = df[col] > c if mask is None else mask(df, c)
        sel = df[keep]
        pred = pd.DataFrame({"filament_id": sel.stem + "_" + sel.inst.astype(str), "segmentation_rle": sel.rle})
        r = compute_pq(gt, pred)
        pf = sel.fold.to_numpy()
        per = [compute_pq(gt[gfold == f], pred[pf == f])["pq"] for f in sorted(df.fold.unique())]
        print(f"  {col:10s} kesme {c:.2f}: PQ {r['pq']:.4f} TP {r['tp']} FP {r['fp']} FN {r['fn']} | "
              + " ".join(f"{x:.4f}" for x in per), flush=True)
        if best is None or r["pq"] > best[1]:
            best = (c, r["pq"])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="outputs/oof/ens1536_tta2")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--final", type=int, default=0, help=">0: bu kadar tohumla tüm OOF adaylarında eğit, kaydet, test gönderimi üret")
    ap.add_argument("--mix", type=float, default=0.5, help="final skor = mix·LightGBM + (1-mix)·CNN")
    ap.add_argument("--cut", type=float, default=0.25)
    ap.add_argument("--reuse", type=int, default=0, help="1: kayıtlı final CNN ağırlıklarını yükle")
    ap.add_argument("--ctx", type=float, default=1.0, help=">1: ek geniş bağlam kesiti (6 kanal, CNN v2)")
    ap.add_argument("--variants", type=int, default=0, help="1: eğitime karşı-olgusal maske varyantları ekle (deney 37)")
    ap.add_argument("--no_prob", type=int, default=0, help="1: olasılık kanalını sabit nötr (128) tut — bağımlılık kontrolü")
    ap.add_argument("--aux_iou", type=float, default=0.0, help=">0: eşiksiz ortalama IoU yardımcı başlığı ağırlığı (deney 40)")
    ap.add_argument("--prob_t", type=float, default=0.0, help=">0: olasılık kanalı sıcaklık augmentasyonu (log-genlik)")
    ap.add_argument("--test_probs", default="outputs/test_probs/ens1536_tta2")
    ap.add_argument("--out", default="outputs/sub_ens1536_cnnsel.csv")
    args = ap.parse_args()
    global PROB_T, AUX_W
    PROB_T, AUX_W = args.prob_t, args.aux_iou
    oof_dir = Path(args.oof)
    records = load_records()
    folds = get_folds().set_index("stem")["fold"]
    df = load_candidates_scored(oof_dir)  # aday manifesti güncel kaynaklarla doğrulanır, eksik skor satırı varsa durur
    df["fold"] = df.stem.map(folds)
    cache = oof_dir / ("cnn_crops.npz" if args.ctx <= 1 else f"cnn_crops_ctx{args.ctx:g}.npz")
    key = crops_key(df, oof_dir, args.ctx)
    if not cache.exists() or str(np.load(cache).get("key", "")) != key:
        print(f"{cache.name} yok ya da adaylarla uyuşmuyor: yeniden üretiliyor", flush=True)
        build_cache(df, oof_dir, folds, cache, args.ctx)
    z = np.load(cache)
    order = pd.DataFrame({"stem": z["stem"], "inst": z["inst"], "row": np.arange(len(z["stem"]))})
    df = merge_exact(df, order, ["stem", "inst"], "aday", "kesit").sort_values("row").reset_index(drop=True)
    X = z["X"][df.row.to_numpy()]
    if args.no_prob:
        X = X.copy()
        X[:, 1] = 128
    print(f"{len(df)} aday, kesit {X.shape}", flush=True)
    if args.aux_iou:
        df = df.merge(iou_mean_table(df, oof_dir), on=["stem", "inst"])
    if args.final:
        return final_and_test(args, X, df.y.to_numpy(), oof_dir)
    if args.variants:  # deney 37: yalnız eğitim kümesine varyant ekle; optimizasyon adımı sayısı sabit
        from cnn_variants import build_variants
        vcache = oof_dir / "cnn_crops_var.npz"
        if not vcache.exists():
            build_variants(df, oof_dir, folds, vcache)
        V = np.load(vcache)
    df["cnn"] = np.nan
    for f in sorted(df.fold.unique()):  # fold sayısı veriden (ekip düzeninde 6)
        tr, va = (df.fold != f).to_numpy(), (df.fold == f).to_numpy()
        Y = np.stack([df.y, df.iou_mean], 1) if args.aux_iou else df.y.to_numpy()
        Xtr, ytr, ep = X[tr], Y[tr], args.epochs
        if args.variants:
            vt = V["fold"] != f
            Xtr, ytr = np.concatenate([Xtr, V["X"][vt]]), np.concatenate([ytr, V["y"][vt]])
            ep = max(1, round(args.epochs * tr.sum() / len(Xtr)))
        df.loc[va, "cnn"] = train_fold(Xtr, ytr, X[va], ep, args.seed + f)
        print(f"fold {f} bitti ({len(Xtr)} eğitim örneği, {ep} epoch)", flush=True)
    assert np.isfinite(df.cnn).all(), "CNN skoru olmayan aday var (NaN adaylar eşikte sessizce elenirdi)"
    df[["stem", "inst", "fold", "y", "score", "cnn"]].to_csv(oof_dir / f"cnn_scores{'' if args.ctx <= 1 else f'_ctx{args.ctx:g}'}{'_var' if args.variants else ''}{'_noprob' if args.no_prob else ''}{f'_pt{args.prob_t:g}' if args.prob_t else ''}{f'_aux{args.aux_iou:g}' if args.aux_iou else ''}_s{args.seed}.csv", index=False)
    print("korelasyon y~lgb", round(np.corrcoef(df.y, df.score)[0, 1], 4), "y~cnn", round(np.corrcoef(df.y, df.cnn)[0, 1], 4))
    gt = gt_frame(records, sorted({p.stem for p in oof_dir.glob("fold*/val_probs/*.npy") if p.stem in records}))
    cuts = np.round(np.arange(0.16, 0.34, 0.02), 2)
    for a in (1.0, 0.0, 0.5, 0.7):
        df[f"mix{a}"] = a * df.score + (1 - a) * df.cnn
        print(f"karışım a={a} (1 = yalnız LightGBM)")
        report(df, gt, folds, f"mix{a}", cuts)
    print("tanı: yalnız eleme (LightGBM 0,24 seçimleri içinde karışım 0,5 > c)")
    report(df, gt, folds, "mix0.5", cuts, mask=lambda d, c: (d.score > 0.24) & (d["mix0.5"] > c))
    print("tanı: yalnız ekleme (LightGBM 0,24 seçimleri + karışım 0,5 > c)")
    report(df, gt, folds, "mix0.5", cuts, mask=lambda d, c: (d.score > 0.24) | (d["mix0.5"] > c))


def final_and_test(args, X, y, oof_dir):
    """Tüm OOF adaylarıyla tohum başına bir CNN eğitir (outputs/.../cnn_selector/), test adaylarını
    LightGBM + CNN karışımıyla seçer ve gönderim dosyası yazar."""
    import lightgbm as lgb

    from data import TEST_IMG
    from metric import check_submission, instances_to_rows
    from post import prob_to_instances
    from scorer import CAND, FEATS, instance_features

    wdir = oof_dir / "cnn_selector"
    wdir.mkdir(exist_ok=True)
    models = []
    for s in range(args.final):
        if args.reuse and (wdir / f"seed{s}.pt").exists():  # kaydedilmiş final modeli kullan (yalnız test varyantı)
            m = timm.create_model("resnet18", pretrained=False, in_chans=3, num_classes=1).cuda().eval()
            m.load_state_dict(torch.load(wdir / f"seed{s}.pt", map_location="cuda"))
        else:
            m = train_fold(X, y, None, args.epochs, 100 + s, keep_model=True)
            torch.save(m.state_dict(), wdir / f"seed{s}.pt")
            print("final CNN", s, "kaydedildi", flush=True)
        models.append(m)
    if not Path(args.test_probs).exists():  # yalnız seçici eğitimi (notebook'taki yeniden eğitim; test paket ile tahmin edilir)
        print(f"{args.test_probs} yok: test gönderimi atlandı", flush=True)
        return
    CAND.update(join=0, link_thr=0.15)
    booster = lgb.Booster(model_file=str(oof_dir / "scorer.txt"))
    rows, n_sel = [], []
    for p in sorted(TEST_IMG.glob("*.jpeg")):
        prob_u8 = np.load(Path(args.test_probs) / f"{p.stem}.npy")
        prob, img_raw = prob_u8.astype(np.float32) / 255, read_image(p)
        lab = prob_to_instances(prob, **CAND)
        feats = instance_features(prob, img_raw, lab, CAND["thr"])
        if not feats:
            continue
        F_ = pd.DataFrame(feats)
        F_["n_inst_img"] = len(feats)
        img = norm_img(p)
        C = np.stack([crop_one(img, prob_u8, lab == k) for k in F_.inst])
        cnn = sum(predict_cnn(m, C) for m in models) / len(models)
        sc = args.mix * booster.predict(F_[FEATS]) + (1 - args.mix) * cnn
        keep = F_.inst.to_numpy()[sc > args.cut]
        remap = np.zeros(lab.max() + 1, np.int32)
        remap[keep] = np.arange(1, len(keep) + 1)
        rows += instances_to_rows(p.stem, remap[lab])
        n_sel.append(len(keep) / len(F_))
    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    check_submission(sub)
    sub.to_csv(args.out, index=False)
    print(f"{len(sub)} filament -> {args.out}; test seçim oranı {np.mean(n_sel):.3f}", flush=True)


if __name__ == "__main__":
    main()
