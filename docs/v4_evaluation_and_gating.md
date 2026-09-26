# v4 讲解：从 Oracle 指标到可部署的回环门控

这轮升级没有继续更换 backbone，而是解决一个更接近真实 SLAM 的问题：

> **模型在一批数据上能找到“完美阈值”，不代表这个阈值到了下一段路线仍然可靠。**

v3 已经证明 DINOv2、因果序列匹配和 LightGlue 各自具有能力，但当时的开集
`best F1` 与 `Recall@100%Precision` 都是在同一个评测 split 上扫描阈值得到的。
它们可以说明分数是否具有可分性，却不能回答部署时真正的问题：

> 只看过去 validation 数据定下一个阈值，机器人继续往前走时还能否保持低误报？

v4 因此把评测重构成严格的两阶段协议，并加入联合门控、结构化输出、置信区间、
度量真值适配和逐阶段运行时测量。

序列约束本身的公式推导、因果语义、速度搜索、复杂度和高频面试问题见
[`sequence_matching.md`](sequence_matching.md)。

---

## 1. 系统现在如何工作

```text
query image
    │
    ├─► global descriptor retrieval
    │       └─ retrieval_score：被选中候选的原始单帧余弦相似度
    │
    ├─► causal sequence re-ranking
    │       └─ sequence_score：沿轨迹对角线聚合后的 Top-1 分数
    │
    ├─► geometric verification of Top-1
    │       ├─ num_matches
    │       ├─ num_inliers
    │       ├─ inlier_ratio
    │       └─ fundamental_matrix
    │
    └─► frozen acceptance gate
            ├─ accept：向 SLAM 后端提出回环约束
            └─ reject：不产生约束，继续运行
```

系统明确区分两个问题：

1. **排序问题**：正确地点有没有排到 Top-K，使用 Recall@K、AUPRC 衡量。
2. **决策问题**：Top-1 是否值得信任，使用冻结阈值下的 precision、recall、F1 衡量。

排序变好不必然意味着同一个决策阈值能跨路线迁移，这是本轮实验最重要的发现。

---

## 2. 为什么旧的开集数字不够

旧流程在 test split 上枚举所有可能阈值，然后选出：

- F1 最高的阈值；
- precision 仍为 100% 时 recall 最高的阈值。

这类指标仍然有价值，但它们使用了 test 标签选择阈值，因此属于 **oracle
score-separation metric**。如果把该阈值称为“部署阈值”，就发生了测试集泄漏。

当前输出通过命名强制区分两类结果：

| 输出前缀 | 含义 | 能否当部署结果引用 |
|---|---|---|
| `oracle/*` | 在当前 split 上扫描阈值的上界 | 否，只能说明可分性 |
| `validation_gate/*` | validation 上拟合 gate 的结果 | 否，用于模型选择 |
| `deployed/*` | 加载冻结 gate 后在 disjoint test 上的结果 | 是 |

### 当前 Gardens Point 切分

| 阶段 | Query | 从 database 移除 | 用途 |
|---|---|---|---|
| Validation | `night_right` 000–049 | 30–39 | 选择 gate 与阈值 |
| Test | `night_right` 050–099 | 70–79 | 只应用冻结 gate |

由于正确匹配允许 `±3` 帧，移除 10 帧并不会产生 10 个完全无匹配的 query；每个
split 实际包含 46 个有匹配 query 和 4 个真正的 open-set query。这也是为什么
代码根据过滤后的 database 重新计算 `has_match`，而不是简单把被移除区间内的
所有 query 都标成未知地点。

---

## 3. 两阶段阈值协议

### 3.1 Validation：拟合并保存 gate

以 DINOv2 + 因果序列分数为例：

```bash
python -m src.evaluate \
  --database outputs/v3/dino_mean_database.pt \
  --query outputs/v3/dino_mean_query.pt \
  --top-k 10 --tolerance 3 \
  --split-name night_right --min-index 0 --max-index 49 \
  --db-exclude-range 30 39 \
  --seq-window 15 --seq-causal \
  --gate-mode similarity \
  --fit-thresholds \
  --thresholds-out outputs/v3/gates/dino_sequence.json
```

这一步可以读取 validation 标签并搜索阈值，但不能报告为 test 性能。

### 3.2 Test：只加载冻结 gate

```bash
python -m src.evaluate \
  --database outputs/v3/dino_mean_database.pt \
  --query outputs/v3/dino_mean_query.pt \
  --top-k 10 --tolerance 3 \
  --split-name night_right --min-index 50 --max-index 99 \
  --db-exclude-range 70 79 \
  --seq-window 15 --seq-causal \
  --thresholds-in outputs/v3/gates/dino_sequence.json \
  --bootstrap-samples 5000 \
  --proposals-out outputs/v3/proposals_sequence.json
```

test 阶段不能重新优化阈值，只能执行 JSON 中已经冻结的决策规则。

### 3.3 阈值文件为什么绑定 pipeline

一个阈值只对产生它的特征和 pipeline 有意义。例如 DINOv2 分数阈值不能直接
拿去判断 ResNet18 特征，单帧阈值也不能无声地用于序列分数。

因此 gate JSON 同时保存：

```json
{
  "gate_mode": "similarity",
  "sequence_threshold": 0.7920109629631042,
  "pipeline": {
    "seq_window": 15,
    "seq_causal": true,
    "seq_velocities": "1.0",
    "geometric_verify": false,
    "ground_truth_protocol": "frame_index@3",
    "feature_meta": {
      "backbone": "dinov2_vits14",
      "pooling": "mean",
      "image_size": 224,
      "feature_dim": 384
    }
  }
}
```

测试时只要 backbone、聚合方式、序列窗口、验证器或真值协议不一致，程序就会
fail fast，而不是用一个语义已经变化的阈值继续产生看似正常的数字。

---

## 4. 四种 gate

所有 gate 都在 validation 上选择参数，并序列化为纯 JSON。test 推理不需要
保存或反序列化 scikit-learn 模型对象。

### 4.1 Similarity gate

```text
accept = sequence_score >= sequence_threshold
```

关闭序列匹配时，`sequence_score` 就是单帧 Top-1 分数。它是最简单、也是当前
Gardens Point 上迁移表现最好的 F1 基线。

### 4.2 Geometry gate

```text
accept = inlier_ratio >= inlier_ratio_threshold
```

v3 使用内点数作为几何置信度。v4 默认使用内点率，因为它能减少“检测到更多
特征点就天然得到更多内点”的影响，也对应之前测到的约 5% RANSAC 噪声地板。

### 4.3 Joint gate

```text
accept = (
    sequence_score >= sequence_threshold
    and inlier_ratio >= inlier_ratio_threshold
)
```

validation 穷举两个阈值，以 F1 为主目标；F1 相同时依次偏好更高 precision 和
更高 recall。这是一个可解释的 AND gate，适合检查几何验证是否提供了独立于
序列分数的拒绝信号。

完整命令：

```bash
python -m src.evaluate \
  --database outputs/v3/dino_mean_database.pt \
  --query outputs/v3/dino_mean_query.pt \
  --top-k 10 --split-name night_right \
  --min-index 0 --max-index 49 --db-exclude-range 30 39 \
  --seq-window 15 --seq-causal \
  --geometric-verify --verifier lightglue \
  --gate-mode joint --fit-thresholds \
  --thresholds-out outputs/v3/gates/dino_lightglue_joint.json
```

test 时必须保留相同的 `--seq-*`、`--geometric-verify` 和 `--verifier` 参数，再
传入 `--thresholds-in`。

### 4.4 Logistic gate

Logistic gate 使用四个特征：

```text
[retrieval_score, sequence_score, inlier_ratio, log(1 + num_matches)]
```

validation 上训练带 balanced class weights 的逻辑回归，然后把系数、截距和
概率阈值写入 JSON：

```text
p(accept) = sigmoid(wᵀx + b)
accept = p(accept) >= probability_threshold
```

它可以表达简单 AND gate 无法表达的权衡，但在当前只有 50 个 validation query
的条件下很容易过拟合。因此它是对照基线，不应在没有更大数据时成为主结果。

---

## 5. Open-set 混淆矩阵的语义

系统接受的不是“这张图属于已知类”，而是一个具体回环候选，因此 accepted 但
Top-1 错误同样属于 FP：

| | 存在正确回环 | 不存在正确回环 |
|---|---|---|
| 系统接受 | Top-1 正确为 TP；Top-1 错误为 FP | FP |
| 系统拒绝 | FN | TN |

指标定义为：

```text
precision = TP / (TP + FP)
recall    = TP / num_queries_with_match
F1        = harmonic_mean(precision, recall)
```

这种定义体现了 SLAM 的代价不对称：任何错误的已接受候选都可能向位姿图加入
错误约束，而拒绝只会少获得一次校正机会。

---

## 6. 新增指标和不确定性

### 6.1 Open-set AUPRC

`open_set/sequence_auprc` 不需要选择单个阈值，它衡量整条 precision-recall
曲线。它适合回答“分数排序是否整体改善”，而冻结 gate 的 F1 回答“这个具体
阈值能否迁移”。两者应该同时看。

### 6.2 Bootstrap 95% CI

`--bootstrap-samples N` 会对 query 做非参数 bootstrap，报告：

- Recall@K 的 95% CI；
- deployed precision、recall、F1 的 95% CI。

这不会增加数据量，但能阻止我们把一两个 query 的变化包装成稳定增益。

### 6.3 逐阶段运行时

评测器报告：

- `runtime/retrieval_ms_per_query`；
- `runtime/geometry_ms_per_query`。

运行时必须与硬件和候选数量一起解释。本项目当前 CPU 环境下，gate 模式的
ORB 约为 23 ms/query，LightGlue 约为 1.0 s/query；GPU 结果会不同。

---

## 7. 几何验证接口的变化

v3 的 `verify_candidates` 只返回内点数。v4 返回完整字典：

```python
{
    "num_keypoints_query": int,
    "num_keypoints_database": int,
    "num_matches": int,
    "num_inliers": int,
    "inlier_ratio": float,
    "fundamental_matrix": list[list[float]] | None,
}
```

### Top-1 优化

`gate` 模式不改变检索排序，只判断当前 Top-1 是否可信，因此没有必要验证
Top-10。v4 只对 Top-1 运行 ORB/LightGlue；只有显式选择 `rerank` 时才验证全部
候选。该修改使当前 CPU 上 LightGlue 从约 6.2 s/query 降到约 1.0 s/query。

### 为什么还不是完整 SLAM 回环模块

当前输出的是 Fundamental Matrix，而不是相对位姿。恢复有尺度约束的旋转和平移
方向需要可靠相机内参，随后还需要把约束接入 pose graph，并由后端做一致性检查
与优化。因此项目应准确称为 **loop-closure front end**，不能声称已经实现完整
SLAM 后端。

---

## 8. 结构化 LoopClosureProposal

传入 `--proposals-out` 后，每个 query 会输出：

```json
{
  "query_path": ".../Image080.jpg",
  "candidate_path": ".../Image082.jpg",
  "retrieval_score": 0.812,
  "sequence_score": 0.731,
  "num_matches": 12,
  "num_inliers": 7,
  "inlier_ratio": 0.583,
  "accepted": false,
  "fundamental_matrix": [
    [-0.000002, 0.000004, -0.000560],
    [0.000004, 0.000004, -0.000555],
    [0.000227, -0.006653, 1.0]
  ]
}
```

这个接口把实验脚本变成了一个可与 SLAM 后端衔接的前端输出，而不是只在终端
打印平均指标。可视化命令：

```bash
python -m src.visualize_proposal \
  --proposals outputs/v3/proposals_sequence.json \
  --index 30 \
  --output outputs/visualizations/gate_decision.png
```

当前 README 中的 false-negative 示例正是由该接口生成：候选在 `±3` 帧真值下
正确、几何内点率为 0.583，但冻结的序列阈值仍将它拒绝。这比单纯展示 Top-5
图片更能解释系统为什么犯错。

---

## 9. 标准 benchmark 真值适配

Gardens Point 没有本项目可直接使用的米制位姿真值，所以早期实验使用帧号差。
v4 新增 CSV manifest 协议，以支持 MSLS、RobotCar 或其他带位置真值的数据：

```csv
query_path,database_path,distance_m
queries/q0001.jpg,database/db0042.jpg,7.4
queries/q0001.jpg,database/db0198.jpg,31.6
```

```bash
python -m src.evaluate \
  --database outputs/benchmark_database.pt \
  --query outputs/benchmark_query.pt \
  --ground-truth-manifest data/benchmark/validation_pairs.csv \
  --distance-threshold-m 25 \
  --gate-mode similarity --fit-thresholds \
  --thresholds-out outputs/gates/benchmark.json
```

距离大于阈值的配对被忽略；在过滤后的 database 中没有任何有效配对的 query
自动成为 open-set negative。接口已经可用，但仓库没有在未实际下载和运行数据
前声称任何外部 benchmark 数字。

---

## 10. 新实验结果应该如何解读

Validation 与 test 严格分开后得到：

| Frozen gate | Test R@1 | AUPRC | Precision | Recall | F1 | FP |
|---|---:|---:|---:|---:|---:|---:|
| DINOv2 single frame | 0.820 | 0.791 | 0.848 | 0.848 | **0.848** | 7 |
| + causal sequence | 0.920 | **0.947** | **0.958** | 0.500 | 0.657 | 1 |
| + sequence + ORB | 0.920 | 0.947 | 0.958 | 0.500 | 0.657 | 1 |
| + sequence + LightGlue | 0.920 | 0.947 | 0.958 | 0.500 | 0.657 | 1 |

5,000 次 bootstrap 的 F1 95% CI：

- 单帧 gate：`[0.747, 0.926]`；
- 因果序列 gate：`[0.516, 0.775]`。

### 结论 1：排序改善不等于冻结决策改善

因果序列将 R@1 从 0.82 提升到 0.92，AUPRC 从 0.791 提升到 0.947，说明它
确实让正确候选整体排得更靠前、置信度排序更好。

但 validation 学到的 `0.792` 阈值到了路线后半段明显过严，只接受 24 个 query：
其中 23 个正确、1 个错误。因此 precision 很高，但 recall 降到 0.50。

这里不能把差异全部归因于自然的 score distribution shift：合成开集协议删除了
连续 database 帧，删除后数组列被压缩，跨缺口的序列轨迹不再对应真实时间间隔。
因此当前结果同时混合了阈值迁移和 database-gap 两个因素。完整分析见
[`sequence_matching.md`](sequence_matching.md#84-合成-open-set-删除会制造-database-gap)。

### 结论 2：联合几何门控没有提供额外增益

validation 为 ORB 学到的内点率阈值只有 `0.042`，LightGlue 则直接学到 `0.0`。
也就是说最优 validation gate 分别把几何条件设得非常宽松或完全忽略，最终与
sequence-only 得到相同 test 结果。

这不是“几何验证没用”的证明。几何模块仍提供特征对应与 Fundamental Matrix，
是未来位姿估计必需的接口。这个结果只说明：在当前小样本 split 上，几何统计
没有改善 accept/reject threshold transfer。

### 结论 3：原来的 100% precision 是 oracle 上界

同集扫描可以找到零假阳性阈值，但该阈值迁移到 test 后仍产生 1 个 FP。项目的
叙事因此从“我们达到 100% precision”升级为更严谨的：

> 分数在同集上具有较强可分性，但冻结决策到了 disjoint test 后失效；当前实验
> 还需用不破坏 database 时间轴的真实 distractor 解耦分布漂移与 gap 混杂因素。

---

## 11. 测试与复现入口

CPU-safe smoke test：

```bash
bash scripts/run_smoke.sh
```

完整 v4 开集表：

```bash
bash scripts/run_ablation_v3.sh openset
```

当前测试覆盖：

- joint gate 必须同时通过两个信号；
- logistic gate 可脱离 sklearn estimator 从 JSON 推理；
- 显式 accept/reject 的 open-set confusion matrix；
- retrieval 与 deployed 指标的 bootstrap CI；
- gate 模式不重排候选且只使用 Top-1 统计；
- 因果序列不会读取未来帧；
- metric-distance manifest 与真正的 open-set query；
- feature artifact 元数据与路径数量检查；
- `LoopClosureProposal` JSON 的关键证据字段。

---

## 12. 面试时怎么讲这轮升级

推荐用下面这条主线，而不是逐个介绍 CLI 参数：

> v3 的开集结果是在同一 split 上扫描阈值得到的，所以它只能说明分数可分性。
> 我把协议改成 validation 定阈值、test 冻结应用，并把特征配置和 pipeline 一起
> 写进 gate artifact，防止阈值被错误复用。新实验发现序列匹配虽然将 R@1 从
> 82% 提到 92%、AUPRC 从 0.791 提到 0.947，但阈值迁移后 recall 只有 50%；
> ORB 和 LightGlue 联合门控也没有改善。这说明当前瓶颈不是再堆一个融合公式，
> 而是小样本下的 threshold transfer；同时 synthetic database gap 仍是待解耦的
> 评测混杂因素。

如果面试官追问工程贡献，再补充：

- gate 输出被结构化为可传给 SLAM 后端的 proposal；
- 几何验证只在 Top-1 上执行，避免 10 倍无效计算；
- 引入 AUPRC、bootstrap CI 和运行时，让精度与系统成本同时可比较；
- 增加米制度量真值适配，但没有在未跑数据时虚构 benchmark 结果。

### 仍然存在的边界

- validation/test 各只有 50 个 query，置信区间仍然较宽；
- Gardens Point 路线逐帧对齐，对序列方法过于友好；
- open-set 仍由移除连续帧段构造，不等于真实大规模 distractor；
- 尚未在标准 benchmark 上实际跑数；
- 没有相机内参、相对位姿和 pose-graph 后端。

这些边界不是需要隐藏的缺陷，而是下一轮实验应该直接针对的研究问题。
