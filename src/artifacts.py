"""OOF ara ürünlerinin (aday önbelleği, seçici skorları) güncellik ve kapsama denetimi.

Her tüketici (scorer, cnn_selector, rubric_eval, error_atlas) aynı yoldan okur:
  - candidates_meta.json, olasılık dosyalarının şu anki içeriğiyle karşılaştırılır; uyuşmazlıkta durulur.
  - candidates_scored{tag}.csv, üretildiği aday manifestine bir yan dosyayla bağlıdır.
  - Tablolar anahtar kümeleri birebir aynı değilse birleştirilmez (eksik ya da fazla satır sessizce düşmez).
"""
import hashlib
import json
from pathlib import Path

import pandas as pd


def probs_sha1(oof_dir):
    """fold*/val_probs/*.npy dosyalarının yol + içerik özeti."""
    h = hashlib.sha1()
    for p in sorted(Path(oof_dir).glob("fold*/val_probs/*.npy")):
        h.update(str(p.relative_to(oof_dir)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def verify_candidates(oof_dir):
    """candidates.pkl'in manifesti var ve olasılık dosyaları o manifestten beri değişmemiş olmalı.
    Dönüş: manifest dosyasının sha1'i (aday kümesinin kimliği)."""
    meta = Path(oof_dir) / "candidates_meta.json"
    if not meta.exists():
        raise SystemExit(f"{meta} yok: aday önbelleğinin üretim ayarı doğrulanamıyor (önce scorer.py çalıştır).")
    saved = json.loads(meta.read_text())
    if saved["probs_sha1"] != probs_sha1(oof_dir):
        raise SystemExit(f"{oof_dir} olasılık dosyaları aday önbelleği üretildikten sonra değişmiş; scorer.py'yi yeniden çalıştır.")
    return hashlib.sha1(meta.read_bytes()).hexdigest()


def write_scored_link(oof_dir, tag=""):
    """candidates_scored{tag}.csv'yi üretildiği aday manifestine bağlar (scorer.py çağırır)."""
    oof_dir = Path(oof_dir)
    link = {"candidates_meta_sha1": hashlib.sha1((oof_dir / "candidates_meta.json").read_bytes()).hexdigest()}
    (oof_dir / f"candidates_scored{tag}.json").write_text(json.dumps(link, indent=1))


def load_candidates_scored(oof_dir, tag=""):
    """Aday tablosu (rle dahil) + LightGBM skoru; yalnız ikisi aynı aday kümesine aitse."""
    oof_dir = Path(oof_dir)
    cand_id = verify_candidates(oof_dir)
    link = oof_dir / f"candidates_scored{tag}.json"
    if not link.exists() or json.loads(link.read_text())["candidates_meta_sha1"] != cand_id:
        raise SystemExit(f"{link.name} yok ya da başka bir aday kümesine ait: skorlar güncel adaylarla eşleşmiyor.")
    cand = pd.read_pickle(oof_dir / "candidates.pkl")
    sc = pd.read_csv(oof_dir / f"candidates_scored{tag}.csv", usecols=["stem", "inst", "score"])
    return merge_exact(cand, sc, ["stem", "inst"], "aday", "skor")


def merge_exact(a, b, on, name_a="sol", name_b="sağ"):
    """Anahtar kümeleri birebir aynı olan iki tabloyu birleştirir; eksik/fazla/tekrarlı anahtarda durur."""
    m = a.merge(b, on=on, how="outer", validate="one_to_one", indicator=True)
    bad = m._merge.value_counts()
    if bad.get("left_only", 0) or bad.get("right_only", 0):
        raise SystemExit(f"anahtar uyuşmazlığı: yalnız {name_a} {bad.get('left_only', 0)}, yalnız {name_b} {bad.get('right_only', 0)}")
    return m.drop(columns="_merge")


def mean_cnn(oof_dir, seeds=(0, 1, 2)):
    """Tohum başına CNN skorlarının anahtarla birleştirilmiş ortalaması (satır sırasına güvenmez)."""
    out = None
    for s in seeds:
        d = pd.read_csv(Path(oof_dir) / f"cnn_scores_s{s}.csv", usecols=["stem", "inst", "cnn"]).rename(columns={"cnn": f"cnn{s}"})
        out = d if out is None else merge_exact(out, d, ["stem", "inst"], "önceki tohumlar", f"tohum {s}")
    out["cnn"] = out[[f"cnn{s}" for s in seeds]].mean(1)
    return out[["stem", "inst", "cnn"]]
