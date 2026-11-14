# Solar Filament Instance Segmentation (IEEE BigData Cup 2026)

Our solution for the Kaggle competition `filament-segmentation-2026`: instance segmentation of solar filaments in 2048×2048 GONG H-alpha full-disk images, scored by Panoptic Quality (PQ, IoU > 0.5) against every annotator separately.

## Result
Pooled 5-fold out-of-fold PQ on the 707 training images (the only metric used for decisions; the public leaderboard was used for record keeping only):

| Step | OOF PQ | Public LB |
|---|---|---|
| U-Net ResNet34, union target, D4 TTA | 0.4048 | 0.35 (fold 0 only) |
| Mean-of-annotators soft target | 0.4099 | – |
| 3-encoder ensemble + learned instance selector (LightGBM) | 0.4379 | 0.38 |
| Saddle linking of fragments | 0.4456 | – |
| 1536 px crops, EfficientNetV2-S + EfficientNetV2-M | 0.4539 | 0.39 |
| **+ CNN candidate-quality network (final system)** | **0.4581** | **0.39–0.40** |

Rubric metrics of the final system (OOF): mean IoU of overlapping GT–prediction pairs 0.600, mean Dice 0.731, ground-truth filaments split into ≥ 2 predictions 2.9%, predictions merging ≥ 2 filaments 3.1% (`docs/figures/fig_rubric_dist.pdf`). For reference, one annotator scored against the others reaches PQ 0.336.

## Pipeline (`docs/figures/fig_pipeline.pdf`)
1. **Normalisation:** each image is standardised by the mean/std of its on-disk pixels.
2. **Segmentation:** U-Net with EfficientNetV2-S and EfficientNetV2-M encoders, trained on 1536 px crops against the mean of the annotators' masks (BCE + Dice, 40 epochs). Inference on the full image with D4 test-time augmentation, averaged over 5 folds × 2 encoders.
3. **Instance candidates:** p > 0.4; fragments in the same connected component of p > 0.15 form one instance (saddle linking); area ≥ 100 px.
4. **Selection:** each candidate's expected PQ contribution is predicted by LightGBM (20 shape/probability/contrast features, computed from the image only) and by a ResNet-18 on a 160×160 crop (image, probability, mask; 3 seeds), both trained with nested cross-validation. A candidate is kept when 0.5·LightGBM + 0.5·CNN > 0.25.
5. **Output:** disjoint masks as COCO RLE (`filament_id`, `segmentation_rle`).

## Reproduce the submission
```bash
uv venv --python 3.11 .venv && uv pip install --index-strategy unsafe-best-match -r requirements.txt
# or: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python src/infer.py run --bundle weights/ens1536 \
    --images MAGFiLO_1.0_Kaggle_2026/test/test_images --out submission.csv
```
or run `notebooks/pipeline.ipynb`:
- it shows every stage (pre-processing, probability map, candidates, selector scores) on one image and writes `submission.csv`;
- it contains an optional retraining section (`TRAIN = True`, runs the commands below) and evaluates the pipeline out-of-fold on the training images with the competition metric;
- the trained weights are in the repository (`weights/ens1536`, 808 MB); files above 50 MB are stored as 45 MB parts and joined in memory, and every file is checked against the SHA-256 in `weights/ens1536/MANIFEST.json` before inference;
- it needs a CUDA GPU with ≥ 12 GB (~6 s per image on an RTX 3090 Ti) and uses only the given test images for inference.

## Retrain from scratch
The competition data is expected in `MAGFiLO_1.0_Kaggle_2026/` (not redistributed). All commands run from the repository root with `PYTHONPATH=src`.
```bash
# 1. segmentation models, 5 folds each (~1.6 h per fold on an RTX 3090 Ti)
for f in 0 1 2 3 4; do
  python src/train.py --fold $f --name crop1536_effv2s --encoder tu-tf_efficientnetv2_s --target mean --crop 1536 --bs 2 --lr 2e-4
  python src/train.py --fold $f --name effv2m_c1536b  --encoder tu-tf_efficientnetv2_m --target mean --crop 1536 --bs 2 --lr 2e-4
done
# 2. out-of-fold probabilities (D4 TTA) and their ensemble average (see tools/pipeline_ens1536.sh)
python src/oof.py --runs outputs/crop1536_effv2s --tta 2 --out outputs/oof/crop1536_effv2s_tta2
python src/oof.py --runs outputs/effv2m_c1536b  --tta 2 --out outputs/oof/effv2m_c1536b_tta2
# 3. LightGBM selector (nested CV), CNN selector (3 seeds) and the final selector models
python src/scorer.py outputs/oof/ens1536_tta2 --link_thr 0.15
for s in 0 1 2; do python src/cnn_selector.py --oof outputs/oof/ens1536_tta2 --seed $s; done
python src/cnn_selector.py --oof outputs/oof/ens1536_tta2 --final 3 --mix 0.5 --cut 0.25
# 4. portable bundle (fp16 weights, large files split into parts, MANIFEST.json with SHA-256)
python src/infer.py export --runs outputs/crop1536_effv2s outputs/effv2m_c1536b --scorer outputs/oof/ens1536_tta2/scorer.txt \
    --cut 0.24 --link_thr 0.15 --cnn outputs/oof/ens1536_tta2/cnn_selector --cnn_mix 0.5 --cnn_cut 0.25 --dst weights/ens1536
# report metrics and figures
python src/rubric_eval.py && python src/report_figs.py
```
Models trained with the current `src/train.py` record their fold scheme and exact train/validation image lists in `args.json` (the stored bundle predates this; its folds are the fixed `outputs/folds.csv` split, which `src/data.py` regenerates); `oof.py` refuses to write out-of-fold predictions for images a model was trained on. Cached candidates and selector scores are bound to the probability files they were computed from (`src/artifacts.py`).

## Repository layout
```
src/                 final pipeline: data, training (train.py, augment.py, losses.py), OOF, selectors (scorer.py,
                     cnn_selector.py), inference (infer.py), artifact checks, rubric metrics and figures
experiments/         scripts of the ablations and diagnostics summarised in the report (run with PYTHONPATH=src)
tools/               development scripts (experiment queue, submission queue, historical ensemble pipelines)
notebooks/           pipeline.ipynb — step-by-step walk-through and end-to-end inference
weights/ens1536/     trained model bundle used for the submission
docs/figures/        report figures
```

## Rules and data
- The public MAGFiLO release (whose labels overlap the test set) and anything derived from it were not used. No hand labelling. Inference uses only the test images.
- Inference reads only the pixels of each test image: no file-name metadata (observation time, station), no neighbouring frames and no other test images. The final selector features are computed from the image only.
- The encoders start from ImageNet weights downloaded by `timm` during training; inference needs only the weights in `weights/ens1536`.
- External labelled datasets (SWEFil, CHASE) and pseudo-labels on unlabelled GONG archive images (test days ±3 and training days excluded) were tried and rejected because they lowered the OOF score; none of them is used by the final system.

## License
Code: MIT (`LICENSE`). The trained weights in `weights/ens1536` are released under the same licence; the ImageNet initialisation comes from `timm`.
