# Results in Detail

> Moved out of the top-level README so that it can stay short. Content is unchanged.
> Back to the [README](../README.md); the full ablation with negative results is in
> [`experiments.md`](experiments.md) (Chinese).

## Results

### SLAM integration: KITTI odometry 00

The front end is wired to a real trajectory. ORB-SLAM3 stereo runs with its own
loop closing disabled to produce odometry; the pipeline then detects loops with
DINOv2, verifies them with DISK + LightGlue, recovers a metric SE(3) relative
pose by PnP on stereo depth, and optimises a pose graph written from scratch.

![KITTI 00 trajectories](../outputs/visualizations/kitti00_loop_closure.png)

| Configuration | ATE RMSE | Max |
| --- | ---: | ---: |
| Odometry, loop closing disabled | 4.335 m | 8.54 m |
| **Ours: DINOv2 + LightGlue + pose graph** | **1.956 m** | 3.82 m |
| DBoW2, ORB-SLAM3 built-in loop closing | 1.204 m | 3.28 m |

Loop edges are accurate: of the 763 accepted, **97% lie within 5 m** of ground
truth (median 0.89 m, median 1213 PnP inliers, median reprojection error
0.84 px). This holds even though retrieval alone is weak on this sequence —
AUROC 0.896 and only 17% top-1 hit rate — which is the project's recurring
finding that recall is cheap and precision must come from the second stage.

Substituting ground truth for either the detected loop pairs or the estimated
relative poses does **not** improve the result (1.956 m measured, 2.029 m with
ground-truth relative poses, 2.036 m with ground-truth pairs). The remaining
error therefore comes from the pose-graph formulation, not the front end: unlike
ORB-SLAM3, this back end has no map points and runs no bundle adjustment. See
[`docs/experiments.md`](experiments.md) §6 for the full decomposition, the
information-matrix ablation and the caveats.

### Retrieval

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
accurate as the CLS token, its similarity scores provide far better confidence separation —
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

### Historical open-set oracle evaluation

Removing database frames 30–49 leaves 14 of 100 night queries with no valid
match, so they *should* be rejected. This is what finally makes the gate and
Recall@100%Precision measurable — under the closed-set protocol used earlier,
rejecting was always wrong.

The following v3 table swept thresholds on the same 100-query split. It is kept
to document how the next failure was discovered, but it is an **oracle
separation result**, not a deployable threshold result. The current
`run_ablation_v3.sh openset` instead fits on frames 000–049 and applies the
frozen gate to frames 050–099.

| Configuration | best F1 | R@100%P |
| --- | ---: | ---: |
| DINOv2, single frame | 0.874 | 0.233 |
| + causal sequence matching | **0.912** | **0.686** |
| + ORB gate | 0.859 | 0.000 |
| + LightGlue gate | 0.849 | 0.326 |

Sequence matching nearly triples Recall@100%Precision, so it improves not just
ranking but the reliability of the confidence score. Adding a geometric gate
then *hurt* because the old implementation replaced similarity confidence with
the inlier count.

The replacement has now been implemented and evaluated with disjoint gate
fitting: validation uses queries 000–049 with database frames 30–39 removed;
test uses queries 050–099 with frames 70–79 removed. Thresholds are selected
only on validation and frozen before test.

| Frozen gate | Score threshold | Inlier-ratio threshold | Test precision | Test recall | Test F1 | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv2 single frame | 0.684 | — | 0.848 | 0.848 | **0.848** | 7 |
| Sequence only | 0.792 | — | 0.958 | 0.500 | 0.657 | 1 |
| Sequence + ORB | 0.792 | 0.042 | 0.958 | 0.500 | 0.657 | 1 |
| Sequence + LightGlue | 0.792 | 0.000 | 0.958 | 0.500 | 0.657 | 1 |

With 5,000 query bootstraps, the 95% F1 intervals are `[0.747, 0.926]` for the
single-frame gate and `[0.516, 0.775]` for the sequence gate. The intervals are
wide, reinforcing that this remains a diagnostic result on a small route.

The honest result is more nuanced than the oracle table. Sequence matching
improves test Recall@1 from 0.82 to 0.92 and open-set AUPRC from 0.791 to 0.947,
but its validation-fitted threshold transfers poorly to the later route segment:
precision rises while recall and deployed F1 fall. Geometry does **not** repair
that transfer—validation makes the ORB condition permissive and ignores
LightGlue entirely. The frozen test also produces one false positive, showing
why the oracle 100%-precision number was optimistic.

This synthetic open-set protocol also removes a contiguous database range,
compressing the sequence matrix so that array adjacency no longer always means
temporal adjacency. The drop therefore cannot yet be attributed purely to route
distribution shift; see [`docs/sequence_matching.md`](sequence_matching.md)
for the confound and the timestamp-aware fix.

In gate mode only Top-1 now receives geometric verification; on the current CPU
environment this costs about 23 ms per query for ORB and 1.0 s for LightGlue.
Runtime depends strongly on hardware.

**Caveats worth reading before citing any of this** — see
[`docs/experiments.md`](experiments.md):

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

![Night success](../outputs/visualizations/night_success_118.png)

### Top-1 Failure, Top-5 Success

`night_top5_success_123.png` shows a case where the top-1 result is wrong, but a
correct nearby place appears in the top-5 candidates. This is important for SLAM:
VPR can propose loop-closure candidates, while geometric verification can later
accept or reject them.

![Night top-5 success](../outputs/visualizations/night_top5_success_123.png)

### Night Failure

`night_failure_103.png` shows a failure case where the correct place does not
appear in the top-5 results. The pretrained descriptor is confused by strong
illumination changes and visually similar corridor-like structures.

![Night failure](../outputs/visualizations/night_failure_103.png)

### Frozen-gate decision evidence

The evaluator can emit a `LoopClosureProposal` JSON for every query. This view
shows a correct geometric candidate rejected by the validation-fitted sequence
threshold—a concrete example of threshold transfer, rather than just another
Top-K retrieval montage.

![Frozen gate false negative](../outputs/visualizations/frozen_gate_false_negative_080.png)

```bash
python -m src.visualize_proposal \
  --proposals outputs/proposals.json --index 30 \
  --output outputs/visualizations/frozen_gate_false_negative_080.png
```

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
   improves confidence separation as much as ranking: Recall@100%Precision goes
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
- **Evaluate the new joint gate on a larger benchmark.** The implementation now
  fits similarity and inlier-ratio thresholds on validation and freezes them for
  test, but Gardens Point is too small to establish how well those thresholds
  transfer.
- **Multi-velocity sequence search with a proper velocity prior.** The search
  exists but widening it blindly costs accuracy, because it also gives wrong
  matches more chances to score high.
- **Open-set with real distractors**, not just a removed frame range — a real
  database contains mostly places the robot has never seen.
- Semi-hard negative mining, PCA-whitening, and FAISS/HNSW indexing: standard
  and cheap, but none of them are what currently limits this system.
