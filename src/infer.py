"""Teslimat için uçtan uca çıkarım: ağırlık paketi -> submission.csv (tek komut).

Üç alt komut:
  export: eğitim çıktılarından taşınabilir bir paket kurar (fp16 best.pt + args.json + LightGBM seçici + ayarlar), sonra pack.
  pack:   paketi repoya girecek hale getirir: 50 MB'tan büyük dosyaları 45 MB'lık parçalara böler (GitHub dosya
          sınırı 100 MB) ve her dosyanın sha256'sını MANIFEST.json'a yazar.
  run:    paketi doğrular ve yükler, bir klasördeki görüntüleri tahmin eder, süreyi ölçer, submission.csv yazar.

Örnek:
  python src/infer.py export --runs outputs/crop1536_effv2s outputs/effv2m_c1536b \
      --scorer outputs/oof/ens1536_tta2/scorer.txt --cut 0.24 --link_thr 0.15 --dst weights/ens1536
  python src/infer.py run --bundle weights/ens1536 --images MAGFiLO_1.0_Kaggle_2026/test/test_images --out submission.csv
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from data import read_image
from metric import check_submission, instances_to_rows
from post import prob_to_instances
from scorer import CAND, FEATS, instance_features, select_instances
from train import load_model, load_state, predict

PART_MB = 45  # parça boyu; 50 MB üstü dosyalar bölünür (GitHub 50 MB'ta uyarır, 100 MB'ta reddeder)


def export(args):
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    runs = []
    for rd in map(Path, args.runs):
        folds = []
        for fd in sorted(rd.glob("fold*")):
            if not (fd / "best.pt").exists():
                continue
            out = dst / rd.name / fd.name
            out.mkdir(parents=True, exist_ok=True)
            sd = torch.load(fd / "best.pt", map_location="cpu")
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in sd.items()}, out / "best.pt")
            shutil.copy(fd / "args.json", out / "args.json")
            folds.append(f"{rd.name}/{fd.name}")
        runs.append(folds)
        print(rd.name, len(folds), "fold")
    shutil.copy(args.scorer, dst / "scorer.txt")
    cfg = dict(runs=runs, cut=args.cut, link_thr=args.link_thr, tta=args.tta)
    if args.cnn:  # deney 33: CNN aday kalite ağı (tohum başına bir resnet18)
        seeds = sorted(Path(args.cnn).glob("seed*.pt"))
        (dst / "cnn_selector").mkdir(exist_ok=True)
        for sp in seeds:
            sd = torch.load(sp, map_location="cpu")
            torch.save({k: v.half() if v.is_floating_point() else v for k, v in sd.items()}, dst / "cnn_selector" / sp.name)
        cfg.update(cnn=[f"cnn_selector/{sp.name}" for sp in seeds], cnn_mix=args.cnn_mix, cnn_cut=args.cnn_cut)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    pack(dst)


def read_joined(path):
    """Dosyanın baytları; bölünmüşse parçaları sırayla birleştirir."""
    if path.exists():
        return path.read_bytes()
    return b"".join(p.read_bytes() for p in sorted(path.parent.glob(path.name + ".part*")))


def pack(dst):
    """Büyük dosyaları parçalara böler, her (birleşik) dosyanın sha256'sını MANIFEST.json'a yazar."""
    dst = Path(dst)
    manifest = {}
    files = {q.with_name(q.name.split(".part")[0]) for q in dst.rglob("*") if q.is_file() and q.name != "MANIFEST.json"}
    for p in sorted(files):  # yeniden çalıştırılırsa önceden bölünmüş dosyalar da manifeste girer
        data = read_joined(p)
        n = len(list(p.parent.glob(p.name + ".part*")))
        if p.exists() and len(data) > 50 * 2**20:
            step = PART_MB * 2**20
            for n, i in enumerate(range(0, len(data), step)):
                p.with_name(f"{p.name}.part{n:02d}").write_bytes(data[i:i + step])
            n += 1
            p.unlink()
        manifest[str(p.relative_to(dst))] = dict(sha256=hashlib.sha256(data).hexdigest(), parts=n)
    (dst / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    size = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 2**20
    print(f"paket hazır: {dst} ({size:.0f} MB, {sum(v['parts'] > 0 for v in manifest.values())} dosya bölündü)")


def verify(bundle):
    """MANIFEST.json'daki her dosya var ve sha256'sı tutuyor olmalı (eksik ya da bozuk parçada durur)."""
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    for rel, m in manifest.items():
        if hashlib.sha256(read_joined(bundle / rel)).hexdigest() != m["sha256"]:
            raise SystemExit(f"{bundle / rel}: SHA-256 mismatch or missing file/part")
    print(f"bundle verified: {len(manifest)} files")


def load_cnn(path):
    import timm
    m = timm.create_model("resnet18", pretrained=False, in_chans=3, num_classes=1)
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in load_state(path).items()})
    return m.cuda().eval()


def score_candidates(prob, img, path, scorer, cnns, cfg):
    """Aday etiket haritası ve aday başına skor tablosu (inst, lgb, cnn, score = mix·LightGBM + (1-mix)·CNN)."""
    from cnn_selector import crop_one, norm_img, predict_cnn
    lab = prob_to_instances(prob, **CAND)
    feats = instance_features(prob, img, lab, CAND["thr"])
    if not feats:
        return lab, pd.DataFrame(columns=["inst", "lgb", "cnn", "score"])
    X = pd.DataFrame(feats)
    X["n_inst_img"] = len(feats)
    prob_u8, img_n = np.round(prob * 255).astype(np.uint8), norm_img(path)
    C = np.stack([crop_one(img_n, prob_u8, lab == k) for k in X.inst])
    X["cnn"] = sum(predict_cnn(m, C) for m in cnns) / len(cnns)
    X["lgb"] = scorer.predict(X[FEATS])
    mix = cfg["cnn_mix"]
    X["score"] = (mix * X.lgb if mix > 0 else 0) + (1 - mix) * X.cnn
    return lab, X


def keep_instances(lab, keep):
    """Yalnız keep'teki etiketleri bırakır, 1..n olarak yeniden numaralar."""
    remap = np.zeros(lab.max() + 1, np.int32)
    remap[keep] = np.arange(1, len(keep) + 1)
    return remap[lab]


def select_cnn(prob, img, path, scorer, cnns, cfg):
    """Deney 33: skor > cnn_cut olan adaylar."""
    lab, X = score_candidates(prob, img, path, scorer, cnns, cfg)
    return keep_instances(lab, X.inst.to_numpy()[X.score.to_numpy() > cfg["cnn_cut"]]) if len(X) else lab


def load_bundle(bundle):
    """Paketi doğrular ve yükler: (ayarlar, encoder başına fold modelleri, LightGBM, CNN'ler)."""
    bundle = Path(bundle)
    verify(bundle)
    cfg = json.loads((bundle / "config.json").read_text())
    if cfg["link_thr"] is not None:
        CAND.update(join=0, link_thr=cfg["link_thr"])
    groups = [[load_model(bundle / f) for f in folds] for folds in cfg["runs"]]
    scorer = lgb.Booster(model_file=str(bundle / "scorer.txt"))
    cnns = [load_cnn(bundle / f) for f in cfg.get("cnn", [])]
    return cfg, groups, scorer, cnns


def ensemble_prob(img, groups, tta):
    """Encoder başına fold ortalaması, sonra encoder'lar eşit ağırlıkla (predict.py ile aynı); uint8'e nicemlenmiş."""
    with torch.no_grad():
        prob = sum(sum(predict(m, img, tta=tta) for m in g) / len(g) for g in groups) / len(groups)
    # uint8 nicemleme. OOF ile birebir aynı değil: OOF'ta her üye aşağı kesilip (floor) ortalaması yuvarlanıyor,
    # burada ortalama yuvarlanıyor (fark ≤ 1/255). OOF haritasını ±1/255 kaydırmak düz hat PQ'sunu ≤ 0,0005
    # oynatıyor (ölçüm deney 45 günlüğünde).
    return np.round(prob * 255).astype(np.uint8).astype(np.float32) / 255


def predict_image(path, cfg, groups, scorer, cnns):
    """Tek görüntü: olasılık haritası ve seçilmiş örnek etiketleri."""
    img = read_image(path)
    prob = ensemble_prob(img, groups, cfg["tta"])
    lab = select_cnn(prob, img, path, scorer, cnns, cfg) if cnns else select_instances(prob, img, path.stem, scorer, cfg["cut"])
    return prob, lab


def predict_folder(images, cfg, groups, scorer, cnns):
    """Klasördeki bütün görüntüler -> submission tablosu (filament_id, segmentation_rle)."""
    paths = sorted(Path(images).glob("*.jpeg")) + sorted(Path(images).glob("*.jpg"))
    rows, times = [], []
    for i, p in enumerate(paths):
        t = time.time()
        rows += instances_to_rows(p.stem, predict_image(p, cfg, groups, scorer, cnns)[1])
        times.append(time.time() - t)
        if i % 20 == 0:
            print(f"{i + 1}/{len(paths)} {times[-1]:.1f} s/image", flush=True)
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    check_submission(df)
    print(f"{len(df)} filaments, {len(paths)} images | mean {np.mean(times):.1f} s/image")
    return df


def run(args):
    t0 = time.time()
    cfg, groups, scorer, cnns = load_bundle(args.bundle)
    print(f"models loaded: {[len(g) for g in groups]} ({time.time() - t0:.1f} s)")
    predict_folder(args.images, cfg, groups, scorer, cnns).to_csv(args.out, index=False)
    print(f"-> {args.out}, total {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--runs", nargs="+", required=True)
    e.add_argument("--scorer", required=True)
    e.add_argument("--cut", type=float, required=True)
    e.add_argument("--link_thr", type=float, default=None)
    e.add_argument("--tta", type=int, default=2)
    e.add_argument("--dst", required=True)
    e.add_argument("--cnn", default=None, help="CNN seçici ağırlık klasörü (seed*.pt), deney 33")
    e.add_argument("--cnn_mix", type=float, default=0.0)
    e.add_argument("--cnn_cut", type=float, default=0.22)
    k = sub.add_parser("pack")
    k.add_argument("--bundle", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bundle", required=True)
    r.add_argument("--images", required=True)
    r.add_argument("--out", default="submission.csv")
    args = ap.parse_args()
    {"export": export, "pack": lambda a: pack(a.bundle), "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
