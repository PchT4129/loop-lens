# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **loop-closure front end for SLAM**, plus a from-scratch pose-graph back end.
Given the current frame, retrieve the same place from a database of past
keyframes, decide whether to trust the match, recover a metric relative pose,
and optimise the trajectory.

The repository is deliberately structured as a **classic-versus-modern
comparison**: every stage has a hand-crafted/CNN implementation and a
self-supervised-transformer/learned-matcher implementation, and each swap is
justified by a measurement from the previous stage. Negative results are kept,
not deleted — they are part of the argument.

Two asymmetries drive the design:

- Same place looks different across day/night; different corridors look alike.
- A missed loop closure only delays a correction; a **false** one welds two
  unrelated places together and can destroy the map. So recall is cheap and
  precision must come from later stages.

## Architecture

### Five-stage pipeline

```
query image
  ① global descriptor retrieval   classic: ResNet18 + GAP/GeM
                                  modern:  DINOv2 ViT-S/14 (zero training)
  ② sequence re-ranking           aggregate along the similarity-matrix diagonal,
                                  optionally over several velocity ratios
  ③ geometric verification        classic: ORB + RANSAC
                                  modern:  DISK + LightGlue + RANSAC
  ④ accept / reject gate          similarity + inlier ratio, fitted on validation,
                                  frozen before test
  ⑤ pose-graph optimisation       stereo/RGB-D depth -> PnP -> SE(3) loop edge;
                                  Gauss-Newton on SE(3), hand-written (no g2o/GTSAM)
```

Stage ① alone is the original v1 baseline. Stages ②–④ and the validation-driven
training loop are v2; the modern variants and the open-set protocol are v3; the
leakage-safe evaluation/frozen-gate protocol is v4; stage ⑤ and the KITTI
integration are v5.

### Key modules

Retrieval front end:
- **`backbones.py`**: `build_backbone()` factory + `DINOv2FeatureExtractor`.
  One contract for ResNet18 and DINOv2 (`cls` / `mean` / `gem` / `cls+gem`
  patch aggregation), so downstream code never branches on backbone.
- **`models.py`**: `ResNet18FeatureExtractor` (classifier removed, L2-normalised
  output) and `GeMPooling`.
- **`dataset.py`**: `ImageFolderDataset`, `get_default_transform()`,
  `get_train_transform()` (ColorJitter augmentation).
- **`extract_features.py`**: batch inference -> `.pt` artifact carrying
  `features`, `paths` **and `meta`** (see Implementation Details).
- **`retrieve.py`**: `retrieve_top_k()` — dot product on L2-normalised features.
- **`sequence_match.py`**: `sequence_rerank()`, simplified SeqSLAM.
  Supports causal mode (past frames only) and multi-velocity search.

Verification and gating:
- **`geometric_verification.py`**: ORB + RANSAC (`verify_pair`,
  `verify_candidates`, `rerank_by_inliers`).
- **`learned_matching.py`**: DISK + LightGlue + RANSAC via kornia, same
  interface as above.
- **`confidence.py`**: `fit_gate()` / `apply_gate()` for the four gate modes
  (`similarity`, `geometry`, `joint`, `logistic`). Gate configs are JSON and
  test-ready.
- **`evaluate.py`** (largest module, ~900 lines): metrics, sequence re-rank
  wiring, open-set protocol, bootstrap CIs, `LoopClosureProposal` emission.
- **`ground_truth.py`**: `PairManifestGroundTruth` — metric-distance ground
  truth from a CSV, for benchmarks without frame-index alignment.

Training:
- **`triplet_dataset.py`**: `TripletPlaceDataset` with index-range filtering.
- **`losses.py`**: `info_nce_loss()`, `build_invalid_negative_mask()`,
  `infonce_diagnostics()` (how much of each batch still produces gradient).
- **`train_triplet.py`**: training loop for both losses, with per-epoch
  validation Recall@K, best-checkpoint saving and early stopping.

SLAM back end:
- **`se3.py`**: Lie group utilities (`se3_exp/log`, `right_jacobian_inv`,
  adjoint). Self-tested.
- **`pose_graph.py`**: `Edge`, `optimize()` — Gauss-Newton on SE(3).
- **`rgbd_pose.py`**: `CameraModel`, `relative_pose()` — PnP + RANSAC.
- **`stereo_depth.py`**: SGBM disparity -> depth, for KITTI stereo.
- **`slam_loop_closure.py`**: end-to-end driver — load odometry trajectory,
  detect loops, verify, recover pose, optimise, write trajectory.

Visualisation:
- **`visualize.py`**: query + top-K montage.
- **`visualize_proposal.py`**: renders a `LoopClosureProposal`'s gate evidence
  and accept/reject decision.

### Data dependencies

Gardens Point Walking (retrieval experiments), 100 frames per traversal:

```
data/gardens_point/
  database/day_left/Image000.jpg ... Image099.jpg
  query/day_right/...
  query/night_right/...
```

Normalise a fresh download with `scripts/prepare_gardens_point.py`.
The image index is the pseudo ground-truth location; a match is correct when
indices are within `±tolerance` frames.

KITTI odometry 00 (stereo) and TUM RGB-D (`--dataset tum --associations`) are
used for the SLAM integration; paths are passed on the command line and the
datasets live outside the repo.

## Development Commands

### Setup

```bash
conda create -n vpr-loop-closure python=3.11
conda activate vpr-loop-closure
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt   # pinned ranges; includes opencv + kornia
```

DINOv2 weights (~85 MB for ViT-S/14) are pulled from `torch.hub` on first use.

### Tests and smoke

```bash
python -m compileall -q src
python -m unittest discover -s tests -v   # confidence, metrics, causality, retrieval
bash scripts/run_smoke.sh                 # CPU-safe; runs tests + a tiny eval if features exist
```

CI (`.github/workflows/test.yml`) runs compileall + unittest on CPU PyTorch for
every push and PR. Keep tests CPU-only and dataset-free.

### Feature extraction

```bash
# classic baseline
python -m src.extract_features --image-dir data/gardens_point/database \
  --output outputs/gardens_database_features.pt

# GeM pooling instead of GAP
python -m src.extract_features ... --pooling gem --gem-p 3.0

# DINOv2, zero training. Patch size 14 -> use multiples of 14 (224, 448)
python -m src.extract_features ... --backbone dinov2_vits14 \
  --dinov2-aggregation mean --image-size 224

# with a fine-tuned checkpoint
python -m src.extract_features ... --checkpoint outputs/checkpoints/v3_triplet.pt
```

### Retrieval and evaluation

```bash
python -m src.retrieve --database DB.pt --query Q.pt --top-k 5

# closed-set metrics
python -m src.evaluate --database DB.pt --query Q.pt \
  --top-k 10 --tolerance 3 --recall-ks 1 5 10 --precision-k 5

# held-out segment only
python -m src.evaluate ... --split-name night_right --min-index 70 --max-index 99

# sequence matching (causal = the real online SLAM constraint)
python -m src.evaluate ... --seq-window 15 --seq-causal --seq-velocities 0.9,1.0,1.1

# geometric verification
python -m src.evaluate ... --geometric-verify --verifier lightglue --geometric-mode gate
```

### Frozen-gate open-set protocol (the v4 contribution)

Fit thresholds on validation, freeze, then apply to a **disjoint** test segment.
`--fit-thresholds` and `--thresholds-in` are mutually exclusive by design.

```bash
# validation: fit and save
python -m src.evaluate ... --split-name night_right --min-index 0 --max-index 49 \
  --db-exclude-range 30 39 --gate-mode similarity \
  --fit-thresholds --thresholds-out outputs/gates/similarity.json

# test: apply the frozen gate, emit CIs and per-query proposals
python -m src.evaluate ... --split-name night_right --min-index 50 --max-index 99 \
  --db-exclude-range 70 79 --thresholds-in outputs/gates/similarity.json \
  --bootstrap-samples 1000 --proposals-out outputs/proposals.json

# metric-ground-truth benchmarks (MSLS/RobotCar): CSV of query_path,database_path,distance_m
python -m src.evaluate ... --ground-truth-manifest pairs.csv --distance-threshold-m 25
```

### Training

```bash
# v2+ recipe: validation-driven model selection is mandatory
python -m src.train_triplet \
  --anchor-dir data/gardens_point/query/night_right \
  --database-dir data/gardens_point/database/day_left \
  --output outputs/checkpoints/resnet18_v2.pt \
  --epochs 8 --batch-size 16 --lr 1e-4 --margin 0.2 \
  --min-index 0 --max-index 59 --db-max-index 56 \
  --val-min-index 60 --val-max-index 69 --early-stop-patience 4 --seed 0

# InfoNCE instead of triplet
python -m src.train_triplet ... --loss infonce --batch-size 60 --tau 0.2
```

`--no-augment` / `--no-schedule` are for ablations. **Always pass `--seed`**:
the validation split is 10 frames, so Recall@1 moves by 0.1 per query.

### Ablations

```bash
bash scripts/run_ablation_v3.sh            # extract + closed + openset
bash scripts/run_ablation_v3.sh extract    # just features/checkpoints (idempotent)
bash scripts/run_ablation_v3.sh closed
bash scripts/run_ablation_v3.sh openset
```

`run_ablation.sh` is the older v2 ablation, kept for provenance.

### SLAM integration

```bash
# 1. ORB-SLAM3 with loop closing disabled (slam/configs/*_loop_off.yaml)
cd slam/runs/kitti_off && stereo_kitti ORBvoc.txt ../../configs/kitti_loop_off.yaml \
  ~/datasets/KITTI/odometry/sequences/00

# 2. detect loops, recover metric poses, optimise
python -m src.slam_loop_closure --dataset kitti \
  --trajectory slam/runs/kitti_off/CameraTrajectory.txt \
  --sequence ~/datasets/KITTI/odometry/sequences/00 \
  --output slam/runs/vpr_kitti.txt \
  --min-gap 100 --top-k 1 --min-inliers 60 --max-reproj 1.5 \
  --rot-info 1e4 --loop-info 10

evo_ape kitti ~/datasets/KITTI/dataset/poses/00.txt slam/runs/vpr_kitti.txt -a
```

### Visualisation

```bash
python -m src.visualize --database DB.pt --query Q.pt --query-index 123 --top-k 5 \
  --output outputs/visualizations/night_top5_success_123.png

python -m src.visualize_proposal --proposals outputs/proposals.json --index 30 \
  --output outputs/visualizations/frozen_gate_false_negative_080.png
```

## Important Implementation Details

1. **Feature normalisation**: L2 normalisation happens inside the extractor's
   `forward()`, so dot product == cosine similarity everywhere downstream.

2. **Feature artifacts carry `meta`**: `{features, paths, meta}` where `meta`
   records `backbone`, `pooling`, `gem_p`, `image_size`, `checkpoint`,
   `feature_dim`. Earlier versions stored only the first two, which meant
   guessing from the filename which config a `.pt` came from. **Do not drop
   `meta` when adding a new extraction path.** Dim is 512 (ResNet18) or 384
   (DINOv2 ViT-S/14).

3. **Image index parsing**: regex `Image(\d+)\.jpg` in `evaluate.py` and
   `triplet_dataset.py`. Critical for ground truth and triplet construction.
   For datasets without this convention use `--ground-truth-manifest` instead of
   renaming files.

4. **`oracle/*` vs `deployed/*` metrics**: `oracle/*` sweeps thresholds on the
   split being reported — it describes score separation only. `deployed/*` uses
   a gate frozen on a different split. **Only `deployed/*` may be cited as an
   operational accept/reject result.** Historical oracle tables in the docs stay
   labelled as such.

5. **Gate configs are pipeline-bound**: the JSON records the pipeline flags
   (`seq_window`, `seq_causal`, `seq_velocities`, `verifier`, `geometric_mode`,
   ground-truth protocol) **and the feature `meta`** it was fitted under. Fit
   and test must match exactly, or `--thresholds-in` errors out. Likewise the
   database and query artifacts must share the same `meta`, or `evaluate.py`
   raises — no more silently comparing DINOv2 queries against ResNet18
   database features.

6. **Causality**: `--seq-causal` restricts sequence aggregation to past frames.
   In `slam_loop_closure.py`, `--min-gap` plays the same role — only keyframes
   that many frames older are eligible, so temporal neighbours cannot pose as
   loops. There is a unit test for this; keep it passing.

7. **`--db-exclude-range` builds the open-set**: removing a contiguous database
   range leaves some queries with no correct answer, which is what makes
   rejection measurable and `Recall@100%Precision` meaningful. It also
   compresses the sequence matrix so array adjacency no longer implies temporal
   adjacency — a known confound, documented in `docs/sequence_matching.md`.

8. **`--top-k` must be >= `max(max(--recall-ks), --precision-k)`**; the
   evaluator fails fast rather than silently truncating the slice (Python
   slicing does not raise on over-range) and printing a `recall@10` that is
   really `recall@5`.

   Other guardrails in `evaluate.py`'s arg validation, all deliberate:
   `--fit-thresholds` requires `--thresholds-out`, an explicit `--split-name`
   and an open-set definition (`--db-exclude-range` or a manifest); it cannot
   be combined with `--thresholds-in`; `--gate-mode geometry|joint|logistic`
   requires `--geometric-verify`; a metric manifest cannot be combined with
   `--db-exclude-range` or with `--min-index/--max-index`.

9. **Geometric verification mode**: `gate` (accept/reject, the ORB-SLAM usage)
   is the default; `rerank` is measurably harmful and kept only for the
   ablation. In gate mode only Top-1 is verified (~23 ms/query ORB, ~1.0 s
   LightGlue on CPU).

10. **Information matrix is not cosmetic**: with identity weighting, one metre
    of translation error costs the same as one radian of rotation error, and
    even ground-truth loop edges only reach 4.335 -> 3.17 m. `--rot-info 1e4`
    reaches 1.96 m.

11. **Checkpoint format**: `{model_state_dict, epochs, lr, margin, val_metrics}`.
    Load with `torch.load(..., map_location="cpu")` + `.load_state_dict()`.

12. **Seeds and sample size**: held-out is 30 queries (1 query = 3.3 points);
    validation is 10 frames (1 query = 10 points). Many differences in the
    tables are noise — report bootstrap CIs (`--bootstrap-samples`) rather than
    point estimates when making a claim.

## Known Results and Limitations

Held-out `night_right` Image070–099 (30 queries), Recall@1:

| Configuration | R@1 | R@5 | R@100%P |
| --- | ---: | ---: | ---: |
| ResNet18-GAP pretrained | 0.500 | 0.767 | 0.000 |
| ResNet18-GeM, zero training | 0.533 | 0.800 | 0.000 |
| ResNet18 + triplet | 0.633 | 0.933 | 0.000 |
| ResNet18 + InfoNCE | 0.633 | 0.900 | 0.000 |
| DINOv2 CLS, zero training | 0.933 | 0.967 | 0.167 |
| DINOv2 patch-mean, zero training | 0.933 | 0.967 | 0.900 |
| + causal sequence matching | 1.000 | 1.000 | 1.000 |

KITTI 00, ORB-SLAM3 stereo with its own loop closing disabled:
odometry 4.335 m ATE -> **1.956 m** with this front end + hand-written pose
graph (DBoW2/ORB-SLAM3 built-in reaches 1.204 m). 97% of 763 accepted loop
edges land within 5 m of ground truth.

Findings that shaped the project — preserve these framings when editing docs:

1. **Validation-driven model selection mattered more than any modelling
   change.** The fixed 5-epoch schedule trained four epochs past the optimum:
   loss fell to 0.002 while validation R@1 fell 0.80 -> 0.30.
2. **The fine-tuning result is a diagnosis, not a result.** A 34.8-point
   generalisation gap says 60 frames were memorised. Zero-training DINOv2 then
   gained 30 points where fine-tuning gained 13.
3. **Sequence matching is the single largest lever** on this dataset, and it
   improves confidence separation as much as ranking — but it depends on
   continuous trajectories at stable speed, which Gardens Point satisfies
   unusually well. Widening velocity search to 0.5–1.5x drops night R@1 from
   0.890 to 0.780.
4. **A negative result was worth as much as a positive one.** ORB inlier counts
   are at chance level across day–night (1.1x separation, 56% separable). The
   claim that this was the descriptor's fault, not RANSAC's, was testable:
   DISK + LightGlue under an identical RANSAC criterion recovered 6.8x.
5. **Recall@1 hides things.** DINOv2 CLS and patch-mean tie at 0.933 R@1 but
   differ 5.4x on Recall@100%Precision.

Open limitations: no standard-benchmark validation (Nordland/Pitts30k/MSLS);
the joint geometry gate still shows no gain over the similarity gate; the
open-set is synthetic (removed frame range, not real distractors); the pose
graph has no map points and runs no bundle adjustment, which is where the
remaining KITTI error lives.

## Deployment Subproject (`deploy/`)

Inference profiling and deployment optimisation of the DINOv2 VPR front end,
started 2026-09-26. Stages 0–3 are done; stage 4 (optimisation ladder:
`torch.compile` → ONNX → TensorRT FP16 → INT8 → FP8/FP4) and stage 5
(per-layer quantisation sensitivity) are the main line. Public summary:
`deploy/README.md`; full record: `deploy/EXPERIMENTS.md` (Chinese).

**The question** is not latency ("loop closure is slow" is false — it is off the
real-time path) but the accuracy cost and knee of each optimisation under a
shared GPU compute/memory/power budget. Accuracy is always measured with the
unmodified `src/evaluate.py` frozen-gate protocol.

### Environment

A separate conda env, `vpr-deploy` (python 3.11), cloned from
`vpr-loop-closure` so the PyTorch build is identical (torch 2.11.0+cu128,
native sm_120). Adds onnx, onnxscript and `tensorrt-cu12==10.16.1.11`
(cu12 to share the CUDA 12.x runtime with torch; 10.x over 11.x for
documentation coverage). The pip TensorRT wheel ships no `trtexec`: engines
are built through the Python API. Not installed on purpose: onnxruntime-gpu,
nvidia-modelopt, Nsight Systems/Compute — ask before adding any of them.

### How `deploy/` relates to `src/`

It consumes `src/` only through public contracts and does not modify it:
`DINOv2FeatureExtractor` (model source), the `{features, paths, meta}`
artifact (so `src/evaluate.py` scores any runtime's features unchanged), and
`get_default_transform` (preprocessing must be bit-identical, calibration
data included). `deploy/export/wrapper.py::VPRDescriptor` bakes the
interpolated positional embedding into a buffer and returns a tensor; it is
**bit-identical** to the original on all 300 images
(`python -m deploy.export.wrapper --self-check` is a hard gate).

One planned `src/` change, approved but not yet made: an
`--allow-runtime-mismatch` flag in `src/evaluate.py` that ignores only
`feature_meta.runtime` when comparing a frozen gate's pipeline, so a gate
fitted on FP32 features can be applied to TensorRT/INT8 features.

### Reference numbers (do not re-derive)

- FP32 reference features: `deploy/results/ref/fp32_{database,query}.pt`
  (gitignored; regenerate with `src.extract_features`, dinov2_vits14, mean, 224).
- Frozen gate for every later variant:
  `deploy/results/protocol/ref_fp32/gate_similarity.json` (threshold 0.6841).
- Primary accuracy metric: **deployed F1 = 0.8478 [0.7473, 0.9263]** (5000
  bootstrap samples). Oracle best F1 0.874 is a separability number only.
- Machine: bandwidth 574 GB/s; FP32/TF32/FP16/BF16 17.6/28.1/54.5/59.8 TFLOP/s.
- Batch 1 is CPU-launch-bound (~2 ms floor, GPU busy ~30% at BF16).

### Measurement discipline (each rule exists because its absence produced a wrong number)

1. Time with CUDA events, warm up, report median/p90/p99; use
   `deploy/bench/timing.py::time_cuda`, never `time.time()` around a call.
2. `ramp_clocks()` before a sweep; discard the first measurement of a session.
3. Record peak allocated/reserved against free VRAM — under WSL2,
   oversubscription silently pages to system memory instead of raising OOM.
4. Never compare two configs measured in size/time order: run a second pass in
   reverse, or interleave. For batch-1 comparisons use paired interleaved
   rounds (`_paired_rounds` in `bench_torch.py`) — an intermittent CPU-side
   slow state can double latency.
5. Compare memory with **one model resident at a time**.
6. `torch.profiler` slows the CPU: take kernel time from the profiler and wall
   clock from an unprofiled run; exclude `record_function` annotations, which
   are mirrored onto the GPU timeline.
7. Cosine similarity has a ~2e-7 noise floor in FP32.
8. Write the expected value in `EXPERIMENTS.md` **before** measuring; append
   expectation, measurement, explanation and conditions. Corrections are
   appended, never silently overwritten.
9. Every claim in the ledger must have committed code that regenerates it
   (`deploy/diagnostics/` holds the one-off experiments).

### Working agreement with the user

Stop and report at the end of each stage; ask before large downloads or
installs. The user is new to the systems side: `DEPLOY_WALKTHROUGH_CN.md` at the
repo root (local only, gitignored) explains every concept in
"what / why / what we are doing" form, assuming no prior knowledge. Keep it in
sync when a conclusion changes.

## Documentation Map

Analysis docs are written in **Chinese**; the README is in English. Match the
language of the file you are editing.

- `README.md` — the public narrative, classic-vs-modern framing, all results.
- `deploy/README.md` — public summary of the deployment subproject (English).
- `deploy/EXPERIMENTS.md` — deployment experiment ledger (Chinese, append-only).
- `docs/experiments.md` — full ablation including negative results and the
  KITTI error decomposition.
- `docs/v4_evaluation_and_gating.md` — frozen-gate protocol walkthrough.
- `docs/sequence_matching.md` — derivation, assumptions, complexity, failure
  modes, and the open-set adjacency confound.
- `docs/learning-notes/` — chronological study notes (01–07).
- `docs/interview/` — structured Q&A bank (numbered topics + A/B/C/D/E
  appendices + glossary).
- `*_CN.md` at repo root — interview prep drafts.

When a result changes, the README table, `docs/experiments.md` and the relevant
learning note all need updating; they are cross-referenced.

## File Organization

- `src/` — all modules are runnable as `python -m src.<module>`.
- `deploy/` — deployment subproject, runnable as `python -m deploy.<pkg>.<module>`; compiled by CI but not unit-tested (it needs a GPU).
- `tests/` — CPU-only unittest suite, no dataset required.
- `scripts/` — dataset prep, ablations, smoke suite.
- `slam/configs/` — ORB-SLAM3 YAML pairs (`*_loop_on/off.yaml`);
  `slam/runs/` — trajectories and logs (committed, they are small).
- `outputs/` — features, checkpoints, gates, visualisations. `.pt` files and
  **all `outputs/**/*.json` (so the fitted gates too)** are gitignored and must
  be regenerated by `run_ablation_v3.sh openset`; only a few representative
  PNGs are committed.
- `docs/interview/` is gitignored on purpose — study notes stay local. The root
  `*_CN.md` prep drafts and `docs/learning-notes/` are untracked for the same
  reason; do not commit them without asking, the GitHub remote is public.
- `.gitignore` excludes `data/`, `*.pt`, checkpoints, Python cache.
- The GitHub remote was renamed to `PchT4129/loop-lens` (public); the local directory and the `vpr-loop-closure` conda env keep the old name.
