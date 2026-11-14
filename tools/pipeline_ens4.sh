#!/usr/bin/env bash
# Final hat: ens4 (EffNetV2-M + ConvNeXt-T + HRNet-W32 + EffNetV2-S rastgele hedef) 5 fold
#   1) eksik model OOF'ları  2) ensemble OOF (mimari başına eşit ağırlık)
#   3) post ayarı + aday seçici (iç içe CV)  4) rubrik  5) test tahmini (gönderilmez)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
P=.venv/bin/python
RUNS="effv2m_mean cnxt_mean hrnet_mean effv2s_random"
for r in effv2m_mean hrnet_mean effv2s_random; do
  $P src/oof.py --runs outputs/$r --tta 2 --folds 2 3 4 --out outputs/oof/${r}_tta2 2>&1 | grep -v -i warn
done
echo "$(date +%H:%M) model OOF'ları hazır"
$P - <<'EOF'
import numpy as np
from pathlib import Path
runs = ["effv2m_mean", "cnxt_mean", "hrnet_mean", "effv2s_random"]
src = [Path(f"outputs/oof/{r}_tta2") for r in runs]
dst = Path("outputs/oof/ens4_tta2")
n = 0
for p in sorted(src[0].glob("fold*/val_probs/*.npy")):
    rel = p.relative_to(src[0])
    arr = np.mean([np.load(s / rel).astype(np.float32) for s in src], 0)
    (dst / rel.parent).mkdir(parents=True, exist_ok=True)
    np.save(dst / rel, arr.round().astype(np.uint8))
    n += 1
print("ens4 OOF:", n, "görüntü")
EOF
echo "=== el ayarlı post (5 fold)"
$P experiments/tune_post.py outputs/oof/ens4_tta2 2>&1 | grep -v -i warn | head -4
echo "=== aday seçici"
rm -f outputs/oof/ens4_tta2/candidates.pkl
$P src/scorer.py outputs/oof/ens4_tta2 2>&1 | grep -v -i warn
echo "$(date +%H:%M) test tahmini"
CUT=$($P -c "import pandas as pd; d=pd.read_csv('outputs/oof/ens4_tta2/scorer_cut.csv'); print(d.loc[d.pq.idxmax(),'cut'])")
$P experiments/predict.py --runs $(for r in $RUNS; do echo -n "outputs/$r "; done) --per_run_mean 1 --tta 2 \
  --scorer outputs/oof/ens4_tta2/scorer.txt --cut $CUT --save_probs outputs/test_probs/ens4_tta2 \
  --out outputs/sub_ens4_scorer.csv 2>&1 | grep -v -i warn | grep -v "^model:"
echo "$(date +%H:%M) bitti (kesme $CUT)"
