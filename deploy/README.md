# Inference Profiling and Deployment of the DINOv2 VPR Front End

> **Status: complete.** Stages 0–3 (environment, machine limits, baseline,
> profiling), the optimisation ladder (`torch.compile` → ONNX → TensorRT FP16 →
> TensorRT INT8), stage 5 (per-group quantisation sensitivity, explicit Q/DQ INT8)
> and stage 6 (FP8 engine, NVFP4 in simulation) are done.
> The experiment ledger [`EXPERIMENTS.md`](EXPERIMENTS.md) is written in Chinese,
> following the repository's convention for analysis documents.

## Why

The loop-closure front end is **not** on the real-time critical path of SLAM, so
"it is too slow" is not the motivation. On a robot, however, SLAM, loop closure,
planning and perception share one GPU's **compute, memory and power budget**.
The question this subproject answers is:

> What does each optimisation — and low-precision quantisation in particular —
> actually cost in downstream VPR accuracy, and where is the knee?

Accuracy is always measured with the repository's own frozen-gate protocol
(`src/evaluate.py`, unchanged), never with a new metric.

## Findings

**Headline: the knee has two layers — FP16 is the last free step at the feature
level; selective INT8 and FP8 are the last steps whose task-level cost is within
noise.**

| Variant (224 px) | batch 1 | batch 32 | Process VRAM | Held-out R@1 | Deployed F1 [95% CI] |
| --- | ---: | ---: | ---: | ---: | --- |
| FP32 eager (reference) | 3.4 ms | 45 ms | 162 MB | 0.933 | 0.848 [0.747, 0.926] |
| FP16 + `torch.compile` (CUDA Graph) | 0.86 ms | 11.5 ms | 142 MB | 0.933 | 0.848 (bit-identical) |
| **FP16 TensorRT + CUDA Graph** | **0.51 ms** | **7.6–7.9 ms** | 106–118 MB | **0.933** | **0.848** (bit-identical) |
| **Selective INT8** (explicit Q/DQ, SmoothQuant) + CUDA Graph | **0.46 ms** | 7.3 ms | **90 MB** | 0.900 | 0.882 [0.796, 0.949] |
| **FP8** (E4M3, every Linear, explicit Q/DQ) + CUDA Graph | **0.46 ms** | 8.3 ms | **50 MB** | 0.900 | 0.826 [0.719, 0.915] |
| INT8 TensorRT (implicit PTQ, whole model) + CUDA Graph | 0.47 ms | 6.9 ms | 98 MB | 0.633 | 0.667 [0.533, 0.791] |
| NVFP4 (every Linear; PyTorch simulation only) | — | — | — | 0.867 | 0.727 [0.600, 0.839] |

FP16 TensorRT is 6.6× faster than FP32 at batch 1 with no measurable accuracy
cost. Whole-model implicit INT8 buys another 8% and gives back all of DINOv2's
advantage. Quantising only qkv/proj/fc1 (SmoothQuant), with `fc2` and attention in
FP16, is 14% faster and 24% smaller than FP16 with task metrics inside FP32's
interval — but not free: its maximum score shift is 2.5× the gate's decision
margin (FP16: 0.34×). (Selective-INT8 batch-1 and FP16 figures come from the same
paired run: 0.524 → 0.459 ms; FP8's from another: 0.520 → 0.461 ms.) FP8 on every
Linear is the smallest (50 MB) at the same speed, with a larger shift (4.5×).
NVFP4 falls past the knee already in simulation.

**Deployment recommendation.** Tightest memory → FP8 on every Linear; best accuracy
among low-precision variants → selective INT8; zero tolerance for false positives →
FP16. Whichever it is, run it inside a CUDA Graph.

**1. Batch 1 is CPU-launch-bound, so lower precision barely helps latency.**
ViT-S/14 issues 154–178 kernels per forward. At BF16 batch 1 the GPU is busy only
~30% of the time; the rest is waiting for the CPU to launch the next kernel.

| batch 1, 224 px | GPU kernel time | Wall clock (no profiler) |
| --- | ---: | ---: |
| FP32 | 2.96 ms | 3.37 ms |
| BF16 | 0.85 ms | 2.61 ms |
| **ratio** | **3.50×** | **1.29×** |

The GPU work does get 3.5× faster — matching the measured Tensor Core ratio — but
the user sees 1.29×. This predicts that CUDA Graphs, not precision, are the lever
at batch 1 (to be tested in stage 4).

![One forward pass, batch 1 vs batch 32](results/timeline_b1_vs_b32.png)

**2. A layout bug in DINOv2's positional-embedding interpolation.**
In the original model, one bicubic interpolation kernel took **~720 µs** — about
46% of all GPU work at BF16 batch 1. The same interpolation on a contiguous tensor
takes 92 µs. DINOv2 feeds it `reshape(...).permute(0, 3, 1, 2)`, a non-contiguous
view whose neighbouring pixels are 1.5 KB apart; changing only the layout makes it
**7.8× slower** with bit-identical output. Since the inference resolution is fixed,
the wrapper bakes the interpolated embedding into a constant (bit-exact against the
original on all 300 images), saving 3–17% latency where the effect is measurable. If resolution must stay dynamic,
a single `.contiguous()` recovers most of it.

**3. Measured limits, not spec-sheet limits.**

| | Measured | Note |
| --- | ---: | --- |
| Bandwidth (triad) | 574 GB/s | 85% of the 672 GB/s spec |
| FP32 / TF32 | 17.6 / 28.1 TFLOP/s | TF32 speed-up 1.60×; the model at batch 32 gets 1.58× |
| FP16 / BF16 | 54.5 / 59.8 TFLOP/s | BF16's micro-benchmark edge does **not** transfer to the model |

![Measured roofline](results/roofline.png)

**4. Measurement traps found and guarded against.**
- Under WSL2, oversubscribing VRAM does not raise OOM: it silently pages to system
  memory (≈500 → 35 GB/s). Every measurement records peak memory against free memory.
- The memory clock drops by a third under load. A probe with a memory-bound control
  shows ViT latency is insensitive to it (±4% vs +52% for the control).
- An intermittent CPU-side slow state doubles batch-1 latency with identical GPU
  kernel time. Its trigger was not identified; batch-1 comparisons therefore use
  paired, interleaved rounds.
- Three of my own measurement errors (profiler annotations counted as kernels, a
  truncated timeline window, a single A/B/A that reversed a sign) are documented in
  the ledger together with how they were caught.

**5. Batch 1 stays launch-bound even under TensorRT.**
`torch.compile` with CUDA Graphs brings FP16 batch 1 from ~2.3 ms to 0.86 ms —
exactly the 0.89 ms of GPU work the profiler measured, as predicted. TensorRT
then cuts the GPU work itself by 37% (it re-fuses the attention that the ONNX
export had split into MatMul/Softmax/MatMul, and times kernels for the exact
shapes), but on its own its wall clock is unchanged at 0.87 ms: its C++ launches
still cannot keep the GPU fed. Only TensorRT inside a CUDA Graph reaches 0.51 ms.
At batch 32, where the GPU is saturated, TensorRT is 1.47× faster than
`torch.compile`.

**6. BF16 loses to FP16 on this model — on both axes.** Task metrics are identical,
but BF16 features deviate 62× more from FP32 than FP16 features do (theory from the
mantissa widths: 64×), and BF16 is not faster in the model. BF16's wider exponent
buys nothing here because FP16 never overflows.

**7. INT8 fails because of a few outlier channels.** With TensorRT's implicit INT8
and post-training calibration (entropy, 50 `day_right` frames disjoint from the
test segment), mean feature cosine to FP32 drops to 0.74 and the maximum score shift
is 50× the gate's decision margin. The calibration cache itself is sane (input
range decodes to ImageNet-normalised pixels), and it matches per-layer maxima
measured independently in PyTorch. The cause: a few fixed channels in DINOv2's
LayerNorm outputs carry activations up to 19× the typical channel maximum.
Per-tensor INT8 either clips them (entropy) or coarsens everything else (min-max).
Fake-quantising only the LayerNorm outputs in PyTorch, with TensorRT's own scales,
reproduces entropy being an order of magnitude worse than min-max. That ordering
is the reverse of what I predicted before measuring.

**8. Which layers can be INT8: `fc2` is the problem, not the residual stream.**
Per-group fake quantisation with NVIDIA ModelOpt (one group at a time, everything
else FP32; an all-off group is bit-identical to FP32 as a sanity check) ranks the
feature error: patch conv 2.7e-4, attention proj 4.4e-4, qkv 6.8e-4, fc1 3.3e-3,
LayerNorm inputs 7.4e-3, **fc2 1.6e-2** — the GELU output, concentrated in 5 of
12 blocks. Per-group errors do not add: seven blocks' fc2 that are each cheap
alone triple the error together. SmoothQuant (divide activation outlier channels
by a factor, multiply the weight columns back — mathematically equivalent) cuts
the qkv/proj/fc1 error 5.2× to 8.9e-4. Exported as Q/DQ ONNX, TensorRT follows the
graph exactly (36 INT8 Linears, attention FP16), and the engine's error matches
the PyTorch simulation (9.0e-4 vs 8.9e-4). Zero-false-positive recall still drops
0.900 → 0.100: one query flips to a neighbour 4 frames away, one frame outside
the ±3 tolerance, with a high score — a reminder of how fragile a metric decided by
a single query out of 30 is.

**9. FP8 and INT8 fail in opposite places; NVFP4 is past the knee.**
Same groups, same calibration, only the number format changed. On Linear
*activations* FP8 (E4M3) is 17× better than INT8 (1.4e-3 vs 2.3e-2): a floating-point
grid keeps outlier and ordinary values at the same relative precision, and `fc2`
stops being special (1.9e-3). On *weights* INT8 is 7× better (5.9e-4 vs 4.2e-3):
weights have no large outliers, and 255 evenly spaced levels beat a 3-bit mantissa;
per-channel FP8 weight scales barely help (3.85e-3), so the error is the format's,
not the scaling's. The FP8 engine (all 48 GEMMs FP8, attention FP16) is 13% faster
than FP16 at batch 1 inside a CUDA Graph but **11% slower without one** — the
quantise/dequantise nodes add 24 kernel launches to a launch-bound pass. At batch 32
the profiler shows 14% less GPU work but wall clock is unchanged, and the FP8
execution context needs only 4 MB; neither is attributed yet. NVFP4 (1-bit
mantissa, one FP8 scale per 16 values) isolates outliers but its weight error alone
is 5.5e-2; held-out R@1 0.867 in simulation. No NVFP4 engine was built: export on CPU
asserts, and export on GPU *silently drops every quantiser* — the ONNX has zero
Q/DQ nodes, and an engine built from it would be FP16 labelled NVFP4. The exporter
now refuses to continue in that case.

**Accuracy reference.** FP32 features extracted through the repository's own entry
point reproduce the README bit-for-bit: deployed F1 **0.848 [0.747, 0.926]**, oracle
best F1 0.874. The frozen gate fitted here is the one every lower-precision variant
is evaluated against; applying it to another runtime's features uses
`src/evaluate.py --allow-runtime-mismatch`, which relaxes only the recorded
runtime and nothing else.

## Layout

```text
deploy/
  EXPERIMENTS.md        experiment ledger: expectation → measurement → explanation
  env/                  check_env.py (self-checks, incl. VRAM-spill probe), requirements
  bench/                timing.py (CUDA-event timer), machine limits, roofline, baseline sweep,
                        torch.compile / TensorRT / INT8 latency, isolated-process memory
  export/wrapper.py     VPRDescriptor: baked pos-embed, tensor in/out, equivalence gate
  export/               ONNX export, TensorRT engine build (FP32/FP16/INT8/explicit INT8/FP8/FP4),
                        INT8 calibrator, runner
  quant/                ModelOpt per-group fake quantisation (INT8/FP8/NVFP4), sensitivity sweep,
                        Q/DQ export with a silent-drop guard
  eval/                 runtime registry, feature extraction per runtime, feature-level comparison,
                        run_protocol.sh (src/evaluate.py's frozen-gate protocol on any feature set)
  profile/              torch.profiler analysis, timelines, CPU-affinity experiments
  diagnostics/          one script per ledger claim (layout, spill, clock, kernels, sampler)
  results/              JSON / CSV / PNG evidence (engines, ONNX, .pt are gitignored)
```

`deploy/` only consumes `src/` through its public contracts
(`DINOv2FeatureExtractor`, the `{features, paths, meta}` artifact, and
`get_default_transform`); it does not modify `src/`.

## Reproduce

```bash
# environment (python 3.11; torch must match the main project's build)
conda create -n vpr-deploy --clone vpr-loop-closure
pip install -r deploy/env/requirements-deploy.txt
python deploy/env/check_env.py

# stage 1: machine limits and roofline
python -m deploy.bench.machine_limits && python -m deploy.bench.roofline

# stage 2: equivalence gate, baseline sweep, FP32 accuracy reference
python -m deploy.export.wrapper --self-check
python -m deploy.bench.bench_torch --sweep && python -m deploy.bench.bench_torch --side
bash deploy/eval/run_protocol.sh ref_fp32 deploy/results/ref/fp32_database.pt deploy/results/ref/fp32_query.pt

# stage 3: profiling and diagnostics
python -m deploy.profile.profile_torch
python -m deploy.diagnostics.bicubic_layout    # the 7.8x layout effect

# stage 4a: half precision and torch.compile
python -m deploy.eval.runtimes extract torch-fp16 torch-bf16 compile-reduce-overhead-fp16
python -m deploy.bench.bench_compile && python -m deploy.bench.memory_isolated

# stage 4b: ONNX -> TensorRT FP16 (one command, stops at the first failure)
bash deploy/run_stage4b.sh

# stage 4c: TensorRT INT8
python -m deploy.export.to_trt --precision int8 --calib-algo entropy --calib-split day --batch 1 32 \
  --out deploy/results/trt_build_int8_entropy_day.json
python -m deploy.eval.runtimes extract trt-int8-entropy-day
python -m deploy.eval.compare_features deploy/results/runtime/trt_int8_entropy_day
python -u -m deploy.bench.bench_int8
python -m deploy.diagnostics.activation_outliers   # the outlier-channel mechanism

# stage 5: per-group sensitivity, then the chosen config as an explicit-INT8 engine
python -m deploy.quant.sensitivity
python -m deploy.quant.export_qdq --group qkv,proj,fc1 --algo smoothquant --tag lin_nofc2_sq --batch 1 32 --build
python -m deploy.eval.runtimes extract trt-qdq-lin_nofc2_sq
python -u -m deploy.bench.bench_int8 --int8 qdq_lin_nofc2_sq
python -m deploy.diagnostics.zero_fp_breakpoint    # which query breaks zero-false-positive recall

# stage 6: FP8 and NVFP4 (same groups, only the format changes)
python -m deploy.quant.sensitivity fp8:weights fp8:linear_in fp8:fc2 fp8:linear nvfp4:weights nvfp4:linear
python -m deploy.quant.export_qdq --fmt fp8 --group linear --tag fp8_lin --batch 1 32 --build
python -m deploy.eval.runtimes extract trt-qdq-fp8_lin
python -u -m deploy.bench.bench_int8 --int8 qdq_fp8_lin
```

Every variant's accuracy goes through the same frozen-gate protocol:
`bash deploy/eval/run_protocol.sh <tag> <db.pt> <query.pt> deploy/results/protocol/ref_fp32/gate_similarity.json`.

Hardware used: RTX 5070 Ti Laptop (Blackwell, sm_120, 12 GB), WSL2, PyTorch
2.11+cu128, TensorRT 10.16 (cu12). All numbers were measured on this machine,
plugged in; absolute values will differ elsewhere, the method should not.
