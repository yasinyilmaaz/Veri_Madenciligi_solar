#!/usr/bin/env bash
# Tur 3 eğitimleri bitince: ens3 5 fold OOF -> post ayarı + aday seçici -> test submission (gönderilmez)
cd "$(dirname "$0")/.."
export PYTHONPATH=src
P=.venv/bin/python
RUNS="outputs/tgt_mean outputs/effv2s_mean outputs/cnxt_mean"
while pgrep -f "tools/queue.sh outputs/jobs_round3.txt" > /dev/null; do sleep 120; done
echo "$(date +%H:%M) eğitimler bitti"
$P src/oof.py --runs $RUNS --tta 2 --out outputs/oof/ens3_mean_tta2 2>&1 | grep -v -i warn
echo "=== el ayarlı post-processing"
$P experiments/tune_post.py outputs/oof/ens3_mean_tta2 2>&1 | grep -v -i warn | head -4
echo "=== aday seçici"
rm -f outputs/oof/ens3_mean_tta2/candidates.pkl
$P src/scorer.py outputs/oof/ens3_mean_tta2 2>&1 | grep -v -i warn
CUT=$($P -c "import pandas as pd; d=pd.read_csv('outputs/oof/ens3_mean_tta2/scorer_cut.csv'); print(d.loc[d.pq.idxmax(),'cut'])")
echo "=== test tahmini (kesme $CUT)"
$P experiments/predict.py --runs $RUNS --tta 2 --scorer outputs/oof/ens3_mean_tta2/scorer.txt --cut $CUT \
  --out outputs/sub_ens3_mean_scorer.csv 2>&1 | grep -v -i warn
echo "$(date +%H:%M) bitti"
