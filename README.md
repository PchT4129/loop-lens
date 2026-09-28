# LoopLens — Visual Place Recognition for SLAM Loop Closure

**English** | [简体中文](README.zh-CN.md)

[![test](https://github.com/PchT4129/loop-lens/actions/workflows/test.yml/badge.svg)](https://github.com/PchT4129/loop-lens/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A loop-closure front end for SLAM, plus a pose-graph back end written from scratch.
Given the current frame, it retrieves the same place from past keyframes, decides
whether to trust the match, recovers a metric relative pose and corrects the
trajectory.

Two things make the problem hard, and they pull in opposite directions: the same
place can look completely different by day and by night, while two different
corridors can look nearly identical. The cost is also asymmetric — a missed loop
only delays a correction, but a false one welds two places together and can
distort the whole map.

## Results at a glance

| What | Baseline | This project |
| --- | --- | --- |
| Night-time place recognition, held-out Recall@1 | 63.3% (fine-tuned ResNet18) | **93.3%** zero-shot DINOv2 → **100%** with causal sequence matching |
| Day–night geometric verification (correct vs incorrect inlier separation) | 1.1× (ORB — chance level) | **6.8×** (DISK + LightGlue, same RANSAC) |
| KITTI 00 trajectory error, ATE RMSE over 3.7 km | 4.335 m (odometry only) | **1.956 m** (ORB-SLAM3's own loop closing: 1.204 m) |
| Accuracy of accepted loop edges on KITTI 00 | — | **97%** of 763 within 5 m of ground truth |
| Accept/reject with a threshold frozen before test | — | deployed F1 **0.848** [0.747, 0.926] |

![KITTI 00 trajectories](outputs/visualizations/kitti00_loop_closure.png)

These are small-sample results: 30–50 held-out queries on one campus route, where
a single query moves Recall@1 by 3.3 points. So every accuracy claim carries a
bootstrap interval, thresholds are fitted on a validation segment and frozen
before test, and negative results are kept rather than deleted.

## How it works

Five stages. Each has a classic and a modern implementation behind the same CLI,
and every swap is justified by a measurement from the previous stage — not by
"this one is newer".

| Stage | Classic | Modern |
| --- | --- | --- |
| ① Global retrieval | ResNet18 + GAP / GeM | DINOv2 ViT-S/14, zero training |
| ② Sequence re-ranking | — | causal aggregation along the similarity-matrix diagonal |
| ③ Geometric verification | ORB + RANSAC | DISK + LightGlue + RANSAC |
| ④ Accept / reject | — | similarity (+ inlier-ratio) gate fitted on validation, frozen for test |
| ⑤ Pose graph | — | depth → PnP → SE(3) loop edges; Gauss–Newton on SE(3), no g2o / GTSAM |

## What the measurements changed

1. **Validation-driven model selection mattered more than any model change.**
   Training loss fell to 0.002 while validation Recall@1 fell from 0.80 to 0.30 —
   the fixed five-epoch schedule trained four epochs too long, invisibly.
2. **The fine-tuning result was a diagnosis, not a result.** A 34.8-point
   generalisation gap said 60 frames were memorised. Swapping in a feature that
   already generalises — zero-training DINOv2 — gained 30 points where
   fine-tuning had gained 13.
3. **A negative result, made testable.** ORB inlier counts were at chance level
   across day and night. Replacing only the descriptor and matcher, under an
   identical RANSAC criterion, recovered the separation from 1.1× to 6.8× — the
   failure was the descriptor's, not RANSAC's.
4. **Recall@1 hides confidence.** DINOv2's CLS token and patch-mean pooling tie at
   0.933 Recall@1 but differ 5.4× in Recall@100%Precision — the number that
   matters when one false loop can tear the map apart.
5. **Oracle numbers are upper bounds.** Sweeping the threshold on the evaluated
   split itself allowed 100% precision (at 0.686 recall). Fitted on a validation
   segment and frozen, the sequence-matching gate reached 0.958 precision on the
   test segment and let one false positive through.
6. **The remaining KITTI error is in the back end.** Substituting ground-truth
   loop pairs or ground-truth relative poses does not help (1.956 → 2.036 /
   2.029 m). And the information matrix is not cosmetic: with identity weighting
   even ground-truth loops only reach 3.17 m.

## In progress: inference profiling and deployment

A follow-up in [`deploy/`](deploy/README.md) asks what each inference
optimisation — low-precision quantisation in particular — costs in VPR accuracy
when the front end shares a robot's GPU budget. Profiling is done; the TensorRT /
INT8 ladder is next.

- **Batch 1 is CPU-launch-bound.** From FP32 to BF16 the GPU work gets 3.5× faster,
  but latency improves only 1.29× — the GPU idles ~70% of the time waiting for
  ~170 kernel launches.
- **A layout bug in DINOv2's positional-embedding interpolation** makes one bicubic
  kernel 7.8× slower than necessary, ~46% of all GPU work at BF16 batch 1. Baking
  the embedding removes it, bit-exact.
- **Accuracy is scored with this repository's protocol unchanged:** the FP32
  reference reproduces deployed F1 0.848 bit-for-bit.

![One forward pass, batch 1 vs batch 32](deploy/results/timeline_b1_vs_b32.png)

## Quick start

```bash
conda create -n vpr-loop-closure python=3.11 && conda activate vpr-loop-closure
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

bash scripts/run_smoke.sh             # CPU-only unit tests, no dataset needed
python scripts/prepare_gardens_point.py \
  --day-left /path/to/day_left --day-right /path/to/day_right \
  --night-right /path/to/night_right
bash scripts/run_ablation_v3.sh       # reproduce the held-out ablation
```

Every command, flag and protocol is documented in [`docs/usage.md`](docs/usage.md).

## Where to read more

| Document | Contents |
| --- | --- |
| [`docs/results.md`](docs/results.md) | All result tables, qualitative examples, discussion, next steps |
| [`docs/usage.md`](docs/usage.md) | Installation, dataset, every command, SLAM integration |
| [`docs/experiments.md`](docs/experiments.md) | Full ablation including negative results (Chinese) |
| [`docs/v4_evaluation_and_gating.md`](docs/v4_evaluation_and_gating.md) | The frozen-gate evaluation protocol (Chinese) |
| [`docs/sequence_matching.md`](docs/sequence_matching.md) | Sequence matching: derivation, assumptions, failure modes (Chinese) |
| [`deploy/README.md`](deploy/README.md) | Inference profiling and deployment subproject |

## Limitations

- One campus route and 30–50 held-out queries; no standard benchmark (Nordland,
  Pitts30k, MSLS) yet — a metric-ground-truth adapter exists but has not been run.
- The open-set protocol removes a contiguous database range rather than adding
  real distractors.
- On the frozen test segment, the joint sequence + geometry gate shows no gain over
  the similarity gate.
- The pose graph has no map points and runs no bundle adjustment, which is where
  the remaining gap to ORB-SLAM3 lives.

## License

[MIT](LICENSE)
