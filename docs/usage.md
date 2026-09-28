# Usage and Reproduction

> Moved out of the top-level README so that it can stay short. Content is unchanged;
> run every command from the repository root. Back to the [README](../README.md).

## Project Structure

```text
loop-lens/
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
    sequence_matching.md         # temporal constraint derivation and Q&A
    v4_evaluation_and_gating.md  # frozen-gate protocol walkthrough
  slam/
    configs/                     # ORB-SLAM3 yaml, loop closing on and off
    runs/                        # trajectories and logs
  src/
    backbones.py                 # ResNet18 / DINOv2 factory, one contract
    dataset.py
    evaluate.py                  # metrics, sequence re-rank, gate, open-set
    confidence.py                # joint/logistic gate fitting and inference
    ground_truth.py              # metric-distance benchmark manifests
    extract_features.py
    geometric_verification.py    # ORB + RANSAC
    learned_matching.py          # DISK + LightGlue + RANSAC
    losses.py                    # InfoNCE + gradient diagnostics
    models.py                    # ResNet18 + GAP/GeM pooling
    retrieve.py
    rgbd_pose.py                 # PnP + RANSAC relative pose from depth
    pose_graph.py                # Gauss-Newton pose graph on SE(3)
    se3.py                       # Lie group utilities, self-tested
    slam_loop_closure.py         # end-to-end SLAM loop closure pipeline
    sequence_match.py            # simplified SeqSLAM, multi-velocity
    stereo_depth.py              # SGBM disparity -> depth, for KITTI
    train_triplet.py
    triplet_dataset.py
    visualize.py
    visualize_proposal.py        # render gate evidence and decisions
  tests/                         # confidence, metrics and causality tests
  deploy/                        # inference profiling & deployment (see deploy/README.md)
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

After downloading the three Gardens Point traversals, normalize them into the
expected layout with:

```bash
python scripts/prepare_gardens_point.py \
  --day-left /path/to/day_left \
  --day-right /path/to/day_right \
  --night-right /path/to/night_right
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

Run the CPU-safe smoke suite:

```bash
bash scripts/run_smoke.sh
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

# Open-set validation: fit a similarity gate and save it
python -m src.evaluate ... --split-name night_right \
  --min-index 0 --max-index 49 --db-exclude-range 30 39 \
  --gate-mode similarity --fit-thresholds \
  --thresholds-out outputs/gates/similarity.json

# Disjoint test segment: apply the frozen gate, emit CIs and proposals
python -m src.evaluate ... --split-name night_right \
  --min-index 50 --max-index 99 --db-exclude-range 70 79 \
  --thresholds-in outputs/gates/similarity.json \
  --bootstrap-samples 1000 --proposals-out outputs/proposals.json

# Joint sequence + geometry gate. Use the same pipeline flags in fit and test.
python -m src.evaluate ... --seq-window 15 --seq-causal \
  --geometric-verify --verifier lightglue --gate-mode joint \
  --split-name night_right --min-index 0 --max-index 49 \
  --db-exclude-range 30 39 \
  --fit-thresholds --thresholds-out outputs/gates/joint.json
```

`oracle/*` metrics scan thresholds on the reported split and describe score
separation only. `deployed/*` metrics use a gate frozen on a different split;
these are the numbers to cite as an operational accept/reject result.

For MSLS, RobotCar or another metric-ground-truth benchmark, provide a CSV with
`query_path,database_path,distance_m` rows. Paths must match the strings stored
in the feature artifacts; queries with no pair inside the distance threshold
are treated as open-set negatives:

```bash
python -m src.evaluate ... \
  --ground-truth-manifest data/benchmark/validation_pairs.csv \
  --distance-threshold-m 25 \
  --gate-mode similarity --fit-thresholds \
  --thresholds-out outputs/gates/benchmark.json
```

The adapter removes the Gardens Point frame-index assumption, but this
repository does not claim external-benchmark numbers until the corresponding
data and feature artifacts have actually been evaluated.

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

### 7. Close Loops on a SLAM Trajectory

Run ORB-SLAM3 with loop closing disabled to get an odometry trajectory, then let
this pipeline detect the loops and optimise the pose graph:

```bash
# Odometry only. slam/configs/*_loop_off.yaml appends `loopClosing: 0`.
# Note this disables the loop-closing thread but not map-point reuse, which on
# short indoor sequences already absorbs most of the drift — see experiments §6.1
cd slam/runs/kitti_off && stereo_kitti ORBvoc.txt ../../configs/kitti_loop_off.yaml \
  ~/datasets/KITTI/odometry/sequences/00

# Detect loops, recover metric relative poses, optimise
python -m src.slam_loop_closure \
  --dataset kitti \
  --trajectory slam/runs/kitti_off/CameraTrajectory.txt \
  --sequence ~/datasets/KITTI/odometry/sequences/00 \
  --output slam/runs/vpr_kitti.txt \
  --min-gap 100 --top-k 1 --min-inliers 60 --max-reproj 1.5 \
  --rot-info 1e4 --loop-info 10

evo_ape kitti ~/datasets/KITTI/dataset/poses/00.txt slam/runs/vpr_kitti.txt -a
```

- `--min-gap` is the causal constraint: only keyframes this many frames older are
  eligible, so temporally adjacent frames cannot masquerade as loops
- `--rot-info` weights the rotation block of the information matrix. It is not
  cosmetic: with an identity matrix, one metre of translation error costs the
  same as one radian (57 degrees) of rotation error, and even 2000 ground-truth
  loop edges only reach 4.335 -> 3.17 m. At `1e4` the same graph reaches 1.97 m
- `--dataset tum` works the same way with `--associations`, taking depth from the
  RGB-D channel instead of stereo SGBM

## Notes

Large downloaded datasets and generated `.pt` feature files should usually not
be committed to Git. The repository should contain the code, documentation, and a
small number of representative visualization images.
