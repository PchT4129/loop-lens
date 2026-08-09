# Visual Place Recognition with CNN Global Descriptors

This mini project implements a simple visual place recognition (VPR) pipeline in
PyTorch. Given a query image, the system retrieves visually similar places from a
database using pretrained CNN features and cosine similarity.

The project is motivated by loop closure and relocalization in SLAM. Traditional
SLAM systems often rely on handcrafted local features and geometric verification,
while learning-based VPR methods learn compact scene-level representations that
can be more robust to viewpoint and appearance changes.

## Overview

The system is a four-stage loop-closure front end:

```text
                 ┌──────────────────────────────────────────────────┐
 query image ──► │ ① Retrieval                                      │
                 │    ResNet18 -> GAP -> 512-D -> L2 norm -> cosine  │
                 └───────────────────────┬──────────────────────────┘
                                         │ Top-K candidates
                 ┌───────────────────────▼──────────────────────────┐
                 │ ② Sequence re-ranking                            │
                 │    aggregate along the diagonal of the           │
                 │    similarity matrix (simplified SeqSLAM)        │
                 └───────────────────────┬──────────────────────────┘
                                         │ re-ranked candidates
                 ┌───────────────────────▼──────────────────────────┐
                 │ ③ Geometric verification (ORB + RANSAC)          │
                 │    used as an accept/reject gate, not a ranker   │
                 └───────────────────────┬──────────────────────────┘
                                         │ inlier count
                 ┌───────────────────────▼──────────────────────────┐
                 │ ④ Accept / reject by inlier threshold            │
                 └──────────────────────────────────────────────────┘
```

Stage ① alone is the original baseline. Stages ②–④ were added in v2, together with a
validation-driven training loop. See [`docs/experiments.md`](docs/experiments.md) for the
full ablation, including the negative results.

The final classifier layer of ResNet18 is removed, so the model is used as a
feature extractor rather than an ImageNet classifier. Since the output features
are L2-normalized, dot product between two feature vectors is equivalent to
cosine similarity.

## Project Structure

```text
vpr-loop-closure/
  data/
    gardens_point/
      database/
      query/
  outputs/
    visualizations/
  src/
    dataset.py
    evaluate.py
    extract_features.py
    models.py
    retrieve.py
    train_triplet.py
    triplet_dataset.py
    visualize.py
  README.md
```

## Dataset

This project uses a subset of the
[Gardens Point Walking dataset](https://huggingface.co/datasets/medwa126/GardensPointWalking).
It contains images from the same route captured under different conditions:

- `day_left`: daytime traversal from the left side of the path
- `day_right`: daytime traversal from the right side of the path
- `night_right`: nighttime traversal from the right side of the path

In the current setup:

```text
data/gardens_point/
  database/
    day_left/
      Image000.jpg ... Image099.jpg
  query/
    day_right/
      Image000.jpg ... Image099.jpg
    night_right/
      Image000.jpg ... Image099.jpg
```

The image index is used as a pseudo ground truth location. For example,
`query/day_right/Image023.jpg` is expected to match a nearby database image such
as `database/day_left/Image023.jpg`, allowing a small index tolerance.

## Installation

Create and activate a conda environment:

```bash
conda create -n vpr-loop-closure python=3.11
conda activate vpr-loop-closure
```

Install PyTorch with CUDA support, then install the remaining dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pillow matplotlib tqdm scikit-learn
```

Verify CUDA:

```bash
python - <<'PY'
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
PY
```

## Usage

### 1. Extract Database Features

```bash
python -m src.extract_features \
  --image-dir data/gardens_point/database \
  --output outputs/gardens_database_features.pt
```

### 2. Extract Query Features

```bash
python -m src.extract_features \
  --image-dir data/gardens_point/query \
  --output outputs/gardens_query_features.pt
```

Each `.pt` file stores:

```python
{
    "features": Tensor[N, 512],
    "paths": list[str],
}
```

To extract features with a triplet fine-tuned checkpoint, pass `--checkpoint`:

```bash
python -m src.extract_features \
  --image-dir data/gardens_point/query \
  --output outputs/gardens_query_features_triplet.pt \
  --checkpoint outputs/checkpoints/resnet18_triplet.pt
```

### 3. Retrieve Top-K Matches

```bash
python -m src.retrieve \
  --database outputs/gardens_database_features.pt \
  --query outputs/gardens_query_features.pt \
  --top-k 5
```

### 4. Evaluate Retrieval

```bash
python -m src.evaluate \
  --database outputs/gardens_database_features.pt \
  --query outputs/gardens_query_features.pt \
  --top-k 10 \
  --tolerance 3
```

The tolerance means that a match is treated as correct if the database image
index is within `±3` frames of the query image index.

`--top-k` must be at least `max(--recall-ks)`; the evaluator now fails fast instead of
silently truncating the slice and reporting a wrong `recall@10`.

Sequence re-ranking and geometric verification are opt-in:

```bash
# 序列匹配：causal 只用过去帧（在线 SLAM 条件），不加则用前后帧
python -m src.evaluate ... --seq-window 15 --seq-causal

# 几何验证：gate = 接受/拒绝（默认），rerank = 按内点数重排（实测有害，见 docs/experiments.md）
python -m src.evaluate ... --geometric-verify --geometric-mode gate --inlier-threshold 20
```

To evaluate a held-out segment, use `--split-name`, `--min-index`, and
`--max-index`:

```bash
python -m src.evaluate \
  --database outputs/gardens_database_features_triplet_split.pt \
  --query outputs/gardens_query_features_triplet_split.pt \
  --top-k 10 \
  --tolerance 3 \
  --split-name night_right \
  --min-index 70 \
  --max-index 99
```

### 5. Visualize Retrieval Results

```bash
python -m src.visualize \
  --database outputs/gardens_database_features.pt \
  --query outputs/gardens_query_features.pt \
  --query-index 123 \
  --top-k 5 \
  --output outputs/visualizations/night_top5_success_123.png
```

### 6. Train with Triplet Loss

The triplet training stage uses nighttime images as anchors, nearby daytime
images as positives, and far-away daytime images as negatives:

```bash
python -m src.train_triplet \
  --anchor-dir data/gardens_point/query/night_right \
  --database-dir data/gardens_point/database/day_left \
  --output outputs/checkpoints/resnet18_triplet_train_000_069.pt \
  --epochs 5 \
  --batch-size 16 \
  --lr 1e-4 \
  --margin 0.2 \
  --min-index 0 \
  --max-index 69
```

The v2 recipe adds a validation split so that model selection is driven by real retrieval
metrics rather than by training loss:

```bash
python -m src.train_triplet \
  --anchor-dir data/gardens_point/query/night_right \
  --database-dir data/gardens_point/database/day_left \
  --output outputs/checkpoints/resnet18_v2.pt \
  --epochs 8 --batch-size 16 --lr 1e-4 --margin 0.2 \
  --min-index 0 --max-index 59 \
  --db-max-index 56 \
  --val-min-index 60 --val-max-index 69 \
  --early-stop-patience 4
```

- `--val-*` enables per-epoch Recall@K evaluation, best-checkpoint saving and early stopping
- `--db-max-index` truncates the database during training so it does not touch the
  validation/test region (purged split)
- `--no-augment` / `--no-schedule` disable ColorJitter / cosine decay for ablations

## Results

Using 100 database images from `day_left` and 200 query images from `day_right`
and `night_right`, the pretrained ResNet18 baseline obtains:

| Query split | Recall@1 | Recall@5 | Recall@10 | Precision@5 |
| --- | ---: | ---: | ---: | ---: |
| `day_right` | 0.9600 | 1.0000 | 1.0000 | 0.8020 |
| `night_right` | 0.5100 | 0.7400 | 0.8700 | 0.3920 |

These results show that pretrained ResNet18 descriptors handle moderate lateral
viewpoint changes well, but performance drops under stronger day-night
appearance changes.

### Held-out ablation

All rows below are evaluated on the **held-out segment `night_right` Image070–099**
(30 queries), which never participates in training. Training uses `000-059`,
validation `060-069`, and the database is truncated to `056` during training to
leave a purged gap.

Reproduce with `bash scripts/run_ablation.sh night_right`.

| # | Configuration | R@1 | R@5 | R@10 | P@5 | R@100%P |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ① | Baseline (pretrained, no training) | 0.500 | 0.767 | 0.900 | 0.433 | 0.000 |
| ② | v1 triplet (fixed 5 epochs, no validation/augmentation) | 0.567 | 1.000 | 1.000 | 0.493 | 0.133 |
| ③ | **v2 triplet** (BN fix + ColorJitter + AdamW + early stopping) | **0.633** | 0.967 | 1.000 | **0.607** | 0.033 |
| ④ | ③ + sequence matching (**causal**, online condition) | **1.000** | 1.000 | 1.000 | 0.893 | **1.000** |
| ⑤ | ③ + sequence matching (non-causal) | **1.000** | 1.000 | 1.000 | **0.913** | **1.000** |
| ⑥ | ⑤ + geometric verification (**gate** mode) | 1.000 | 1.000 | 1.000 | 0.913 | 1.000 |
| ⑦ | ⑤ + geometric verification (**re-rank** mode) — negative result | 0.667 | 1.000 | 1.000 | 0.647 | 0.033 |

**Important caveats — please read `docs/experiments.md` before citing these numbers:**

- The `R@1 = 1.000` in ④/⑤ comes **almost entirely from sequence matching, not from
  training**: applying the same sequence matching to the *untrained baseline* features
  also reaches 1.000. A control that shuffles the query temporal order drops it to 0.300,
  confirming the gain is genuinely temporal rather than an artifact.
- Gardens Point sequences are **frame-aligned** and traversed at near-constant speed, which
  perfectly satisfies the sequence-matching assumption. Real deployments vary in speed and
  direction, so the gain would be smaller.
- Row ③ trains on 10 fewer frames than ② (they became the validation split), so it is not a
  free improvement. Its `R@5` and `R@100%P` are slightly *worse*; with only 30 queries a
  single query is 3.3 points, so those differences are within noise.
- Row ⑦ is kept deliberately as a **negative result**. ORB inlier counts carry essentially no
  signal across day–night pairs (correct 25.1 vs incorrect 24.4 inliers), so re-ranking by
  them destroys a correct ranking. Geometric verification belongs in an accept/reject gate,
  not in a ranker.

### Training curve: loss does not track retrieval quality

```text
Epoch 1: loss 0.1246 | val R@1 0.8000   <- best
Epoch 2: loss 0.0308 | val R@1 0.7000
Epoch 3: loss 0.0085 | val R@1 0.6000
Epoch 4: loss 0.0052 | val R@1 0.5000
Epoch 5: loss 0.0020 | val R@1 0.3000   -> early stop
```

Training loss falls monotonically to 0.002 while validation Recall@1 falls monotonically
from 0.80 to 0.30. The original fixed 5-epoch schedule was training four epochs too long,
and without a validation split this is completely invisible.

## Qualitative Examples

### Night Success

`night_success_118.png` shows a nighttime query where the top retrievals are
nearby correct locations. The descriptor captures stable global layout cues such
as corridor direction, doorway position, lighting structure, and indoor geometry.

![Night success](outputs/visualizations/night_success_118.png)

### Top-1 Failure, Top-5 Success

`night_top5_success_123.png` shows a case where the top-1 result is wrong, but a
correct nearby place appears in the top-5 candidates. This is important for SLAM:
VPR can propose loop-closure candidates, while geometric verification can later
accept or reject them.

![Night top-5 success](outputs/visualizations/night_top5_success_123.png)

### Night Failure

`night_failure_103.png` shows a failure case where the correct place does not
appear in the top-5 results. The pretrained descriptor is confused by strong
illumination changes and visually similar corridor-like structures.

![Night failure](outputs/visualizations/night_failure_103.png)

## Discussion

This baseline demonstrates that off-the-shelf CNN global descriptors are already
useful for visual place recognition. However, the performance gap between
`day_right` and `night_right` highlights the difficulty of appearance changes.
Triplet-loss fine-tuning improves day-night retrieval, but the held-out split
shows why train/test separation is necessary when judging generalization.

In a SLAM system, this type of VPR module would typically be used as a candidate
retrieval stage:

```text
query image -> top-k place candidates -> geometric verification -> loop closure
```

High recall is important because the correct place must appear among the
candidates. High precision is also important because false loop closures can
damage the pose graph or map.

Three findings from the v2 ablation sharpen this picture:

1. **Validation-driven model selection mattered more than any modelling change.** The fixed
   5-epoch schedule was training four epochs past the optimum, which is invisible if you only
   watch the loss.
2. **Sequence matching dominates on this dataset** — zero training, zero learned parameters,
   and it saturates Recall@1 on the held-out segment. Its gain, however, rests on the
   trajectory being continuous and traversed at a stable speed.
3. **ORB-based geometric verification does not transfer across day–night pairs**
   (correct 25.1 vs incorrect 24.4 inliers, i.e. chance level), while on same-domain pairs it
   separates cleanly (134.7 vs 26.1). This independently confirms why learned descriptors are
   needed here in the first place.

## Next Steps

- Replace ORB with learned local features (SuperPoint + LightGlue) so that geometric
  verification also works across day–night pairs.
- Add hard negative mining (semi-hard) — 24% of randomly sampled triplets already produce
  zero gradient at the start of training.
- Evaluate in an open-set setting with distractors and "no loop closure" negatives, so that
  the accept/reject gate can actually be measured.
- Multi-velocity sequence search instead of assuming a fixed 1:1 frame alignment.
- Swap GAP for GeM pooling and add PCA-whitening.
- Validate on a standard benchmark (Nordland / Pitts30k / MSLS) with metric ground truth.

## Notes

Large downloaded datasets and generated `.pt` feature files should usually not
be committed to Git. The repository should contain the code, documentation, and a
small number of representative visualization images.
