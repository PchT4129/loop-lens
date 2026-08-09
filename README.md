# Visual Place Recognition with CNN Global Descriptors

This mini project implements a simple visual place recognition (VPR) pipeline in
PyTorch. Given a query image, the system retrieves visually similar places from a
database using pretrained CNN features and cosine similarity.

The project is motivated by loop closure and relocalization in SLAM. Traditional
SLAM systems often rely on handcrafted local features and geometric verification,
while learning-based VPR methods learn compact scene-level representations that
can be more robust to viewpoint and appearance changes.

## Overview

The system is a four-stage loop-closure front end. Each stage has a classic and a
modern implementation, selectable from the CLI, so the two can be compared directly:

```text
                 ┌──────────────────────────────────────────────────────┐
 query image ──► │ ① Retrieval                                          │
                 │    classic: ResNet18 -> GAP/GeM -> 512-D -> L2        │
                 │    modern:  DINOv2 ViT-S/14 -> patch pooling -> 384-D │
                 └───────────────────────┬──────────────────────────────┘
                                         │ Top-K candidates
                 ┌───────────────────────▼──────────────────────────────┐
                 │ ② Sequence re-ranking                                │
                 │    aggregate along the similarity matrix diagonal,    │
                 │    optionally searching several velocity ratios       │
                 └───────────────────────┬──────────────────────────────┘
                                         │ re-ranked candidates
                 ┌───────────────────────▼──────────────────────────────┐
                 │ ③ Geometric verification                             │
                 │    classic: ORB + RANSAC                              │
                 │    modern:  DISK + LightGlue + RANSAC                 │
                 └───────────────────────┬──────────────────────────────┘
                                         │ inlier count
                 ┌───────────────────────▼──────────────────────────────┐
                 │ ④ Accept / reject by threshold                       │
                 │    measurable only under the open-set protocol        │
                 └──────────────────────────────────────────────────────┘
```

Stage ① alone is the original baseline; stages ②–④ and the validation-driven
training loop were added in v2; the modern variants and the open-set protocol
in v3. See [`docs/experiments.md`](docs/experiments.md) for the full ablation,
including the negative results.

Training supports both triplet loss and InfoNCE, with diagnostics that report
how much of each batch still produces gradient — which turned out to matter
more than the choice of loss.

## Project Structure

```text
vpr-loop-closure/
  data/
    gardens_point/
      database/
      query/
  outputs/
    visualizations/
  scripts/
    run_ablation.sh              # v2 ablation
    run_ablation_v3.sh           # v3 ablation (classic vs modern)
  docs/
    experiments.md               # full ablation with negative results
    interview/                   # study notes written alongside the code
  src/
    backbones.py                 # ResNet18 / DINOv2 factory, one contract
    dataset.py
    evaluate.py                  # metrics, sequence re-rank, gate, open-set
    extract_features.py
    geometric_verification.py    # ORB + RANSAC
    learned_matching.py          # DISK + LightGlue + RANSAC
    losses.py                    # InfoNCE + gradient diagnostics
    models.py                    # ResNet18 + GAP/GeM pooling
    retrieve.py
    sequence_match.py            # simplified SeqSLAM, multi-velocity
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

All rows are evaluated on the **held-out segment `night_right` Image070–099**
(30 queries), which never participates in training. Reproduce with
`bash scripts/run_ablation_v3.sh`.

The project was built in two passes: first a classic pipeline (ResNet18 +
triplet + ORB) to establish where each stage breaks, then each stage swapped
for a modern replacement, with the earlier measurement as the stated motive.

| # | Configuration | R@1 | R@5 | P@5 | R@100%P |
| --- | --- | ---: | ---: | ---: | ---: |
| ① | ResNet18-GAP, pretrained | 0.500 | 0.767 | 0.433 | 0.000 |
| ② | ResNet18-**GeM**, zero training | 0.533 | 0.800 | 0.407 | 0.000 |
| ③ | ResNet18-GAP + **triplet** | 0.633 | 0.933 | 0.527 | 0.000 |
| ④ | ResNet18-GAP + **InfoNCE** | 0.633 | 0.900 | 0.593 | 0.000 |
| ⑤ | **DINOv2 CLS**, zero training | **0.933** | 0.967 | 0.753 | 0.167 |
| ⑥ | **DINOv2 patch-mean**, zero training | **0.933** | 0.967 | **0.800** | **0.900** |
| ⑦ | ⑥ + causal sequence matching | **1.000** | 1.000 | 0.933 | **1.000** |

**Zero-training DINOv2 beats the fine-tuned ResNet18 by 30 points.** With only
60 training frames, swapping in a feature that already generalises beats
fine-tuning one that does not — consistent with the 34.8-point generalisation
gap measured for the fine-tune. On `day_right` (viewpoint change only) every
configuration sits at 0.94–0.98, so the gain is specific to the cross-domain
case.

Rows ⑤ and ⑥ are worth a second look: identical Recall@1, but
Recall@100%Precision differs by 5.4x. Patch aggregation is not just as
accurate as the CLS token, its similarity scores are far better calibrated —
which matters more than average accuracy when a single false loop closure can
tear the map apart.

### Geometric verification: closing a negative result

v2 found that ORB inlier counts carry no signal across day–night pairs and
argued the cause was the descriptor rather than RANSAC. v3 tests that by
swapping only the feature and matcher, keeping the RANSAC criterion identical.
Measured over 32 sample points scored on exactly the same pairs:

| Method | Domain | Correct | Incorrect | Ratio | Separable |
| --- | --- | ---: | ---: | ---: | ---: |
| ORB | day_right | 126.5 | 26.9 | 4.7x | 97% |
| ORB | night_right | 27.0 | 25.7 | **1.1x** | **56%** |
| DISK+LightGlue | day_right | 446.4 | 9.6 | 46.7x | 100% |
| DISK+LightGlue | night_right | 62.5 | 9.2 | **6.8x** | **72%** |

Chance level is 50%, so ORB at 56% is effectively random. The failure was in
the descriptor. Still, 72% remains well short of the 100% seen in-domain, and
LightGlue's night distribution is heavily skewed (mean 62.5, median 18).

### Open-set evaluation

Removing database frames 30–49 leaves 14 of 100 night queries with no valid
match, so they *should* be rejected. This is what finally makes the gate and
Recall@100%Precision measurable — under the closed-set protocol used earlier,
rejecting was always wrong.

| Configuration | best F1 | R@100%P |
| --- | ---: | ---: |
| DINOv2, single frame | 0.874 | 0.233 |
| + causal sequence matching | **0.912** | **0.686** |
| + ORB gate | 0.859 | 0.000 |
| + LightGlue gate | 0.849 | 0.326 |

Sequence matching nearly triples Recall@100%Precision, so it improves not just
ranking but the reliability of the confidence score. Adding a geometric gate
then *hurts*, because the current implementation replaces the similarity
confidence with the inlier count instead of requiring both to pass. Combining
the two signals is the obvious next step.

**Caveats worth reading before citing any of this** — see
[`docs/experiments.md`](docs/experiments.md):

- Held-out is 30 queries, so one query is 3.3 points; the validation split is
  10 frames, so one query is 10 points. Several differences here are noise.
- Gardens Point sequences are frame-aligned and traversed at near-constant
  speed, which perfectly satisfies the sequence-matching assumption. Widening
  the velocity search to 0.5–1.5x drops night Recall@1 from 0.890 to 0.780, so
  the assumption is doing real work.
- InfoNCE's mechanism advantage is measurable (gradient diagnostics) but does
  not translate into a measurable retrieval gain at this dataset size.

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
