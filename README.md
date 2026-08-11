# Visual Place Recognition for SLAM Loop Closure

A loop-closure front end for SLAM: given the current frame, retrieve the same
place from a database of past keyframes, then decide whether to trust the match.

The interesting part of the problem is that the two failure modes pull in
opposite directions. The same place can look completely different across day and
night, while two different corridors can look nearly identical. And the cost is
asymmetric: a missed loop closure just delays a correction, but a false one
welds two unrelated places together and can distort the whole map irreversibly.

The project is built as a **classic-versus-modern comparison**. Each stage has
two implementations — hand-crafted or CNN on one side, self-supervised
transformer or learned matcher on the other — and every swap is motivated by a
measurement from the previous stage rather than by "this one is newer".

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
pip install -r requirements.txt
```

`opencv-python` is needed for ORB geometric verification and `kornia` for the
DISK + LightGlue path. DINOv2 weights (~85 MB for ViT-S/14) are pulled from
`torch.hub` on first use.

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

Each `.pt` file stores the features, their source paths, and the configuration
they were produced with — earlier versions stored only the first two, which meant
guessing from the filename which checkpoint and resolution a file came from:

```python
{
    "features": Tensor[N, D],        # D = 512 (ResNet18) or 384 (DINOv2 ViT-S/14)
    "paths": list[str],
    "meta": {"backbone": ..., "pooling": ..., "image_size": ..., "checkpoint": ...},
}
```

Backbone, pooling and resolution are all flags. The backbone is fully
convolutional with adaptive pooling (and DINOv2 is a ViT), so changing
resolution needs no model changes:

```bash
# GeM pooling instead of GAP
python -m src.extract_features ... --pooling gem

# DINOv2, zero training. Note patch size 14, so use multiples of 14 (224, 448)
python -m src.extract_features ... --backbone dinov2_vits14 --dinov2-aggregation mean
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

Sequence re-ranking, geometric verification and the open-set protocol are opt-in:

```bash
# Sequence matching. --seq-causal uses only past frames, matching the online
# SLAM constraint. --seq-velocities searches several speed ratios; widening it
# is not free, see docs/experiments.md
python -m src.evaluate ... --seq-window 15 --seq-causal --seq-velocities 0.9,1.0,1.1

# Geometric verification. gate = accept/reject (default), rerank = reorder by
# inlier count (measurably harmful). orb is fast, lightglue works across
# day-night where orb is at chance level
python -m src.evaluate ... --geometric-verify --verifier lightglue --geometric-mode gate

# Open-set: drop a database frame range so some queries have no valid match and
# *should* be rejected. Without this, rejecting is always wrong and the gate
# cannot be measured at all
python -m src.evaluate ... --db-exclude-range 30 49
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
- `--seed` matters here: the validation split is 10 frames, so Recall@1 moves by
  0.1 per query and ablations are not comparable without it

InfoNCE is available as an alternative loss. It uses every valid in-batch
negative rather than one sampled negative, and weights each by its softmax
probability, so hard negatives get more gradient with no explicit mining:

```bash
python -m src.train_triplet ... --loss infonce --batch-size 60 --tau 0.2
```

Both losses report how much of each batch still produces gradient, which turned
out to matter more than the choice of loss — triplet reaches 98% zero-gradient
triplets by epoch 6, and InfoNCE saturates too when the batch is small enough
that the in-batch task becomes trivial. See `docs/experiments.md`.

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

In a SLAM system this module sits at the front of a funnel that tightens stage
by stage:

```text
query image -> top-k candidates -> geometric verification -> consistency check -> loop closure
```

The first stage should favour recall: a correct place only has to reach the
candidate list, because later stages can reject the wrong ones — but anything
missed here is unrecoverable. Final precision is the later stages' job. That is
why Recall@K is the headline retrieval metric, while Recall@100%Precision is the
one that reflects what SLAM actually needs.

Five findings shaped how this project ended up:

1. **Validation-driven model selection mattered more than any modelling change.**
   The fixed 5-epoch schedule was training four epochs past the optimum —
   invisible if you only watch the loss, which fell to 0.002 while validation
   Recall@1 fell to 0.30.

2. **The fine-tuning result was better read as a diagnosis than as a result.**
   A 34.8-point generalisation gap says the model memorised 60 frames rather
   than learning something transferable. That pointed at the feature, not the
   training recipe — and swapping in zero-training DINOv2 gained 30 points where
   fine-tuning had gained 13.

3. **Sequence matching is the single largest lever on this dataset**, and it
   improves confidence calibration as much as ranking: Recall@100%Precision goes
   from 0.233 to 0.686. But the gain rests on the trajectory being continuous and
   traversed at a stable speed, which Gardens Point satisfies unusually well.

4. **A negative result was worth as much as a positive one.** ORB inlier counts
   sit at chance level across day–night pairs. The claim that this was the
   descriptor's fault rather than RANSAC's was testable, and swapping in
   DISK + LightGlue under an identical RANSAC criterion recovered separation from
   1.1x to 6.8x.

5. **Recall@1 hides things.** DINOv2's CLS and patch-mean aggregations score
   identically on Recall@1 (0.933) but differ 5.4x on Recall@100%Precision. When
   one false loop closure can tear the map apart, how trustworthy the top
   prediction is matters more than how often it is right on average.

## Next Steps

Ordered by how much they would change the conclusions rather than by effort:

- **Validate on a standard benchmark** (Nordland / Pitts30k / MSLS) with metric
  ground truth. Every number here comes from 30 held-out queries on one campus
  route, where a single query is worth 3.3 points.
- **Combine the confidence signals instead of substituting them.** The geometric
  gate currently replaces the sequence similarity with the inlier count, which
  is why it lowers Recall@100%Precision; requiring both to pass should not.
- **Multi-velocity sequence search with a proper velocity prior.** The search
  exists but widening it blindly costs accuracy, because it also gives wrong
  matches more chances to score high.
- **Open-set with real distractors**, not just a removed frame range — a real
  database contains mostly places the robot has never seen.
- Semi-hard negative mining, PCA-whitening, and FAISS/HNSW indexing: standard
  and cheap, but none of them are what currently limits this system.

## Notes

Large downloaded datasets and generated `.pt` feature files should usually not
be committed to Git. The repository should contain the code, documentation, and a
small number of representative visualization images.
