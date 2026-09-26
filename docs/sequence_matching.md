# 序列约束讲义：从单帧 VPR 到因果轨迹匹配

这篇讲义对应项目中的第二级模块 `src/sequence_match.py`。目标不是只会调用
`--seq-window 15`，而是能够回答下面这些面试问题：

- 为什么序列信息能缓解 perceptual aliasing？
- 相似度矩阵上的对角线从哪里来？
- `velocity` 到底代表物理速度还是索引速度？
- causal 为什么没有使用未来帧？它是否已经是在线实现？
- 多速度搜索为什么可能让结果变差？
- 删除一段 database 后，序列约束是否仍然成立？
- 这个实现与完整 SeqSLAM、HMM/Viterbi、位姿图约束有什么区别？
- 为什么闭集 R@1 达到 100%，却不能宣称问题已经解决？

---

## 1. 问题背景：单帧检索丢掉了什么

单帧 VPR 对每个 query 独立计算：

```text
query frame q_i -> global descriptor -> nearest database frame d_j
```

这样做忽略了机器人运动的连续性。如果当前帧位于 database 的 `j` 附近，下一帧
通常不会突然跳到完全无关的 `j+80`，而应该落在 `j+1` 或某个由速度比决定的
邻近位置。

单帧错误通常具有两种形态：

1. **孤立误匹配**：某一帧恰好与错误地点纹理相似，但前后帧不支持该位置。
2. **重复结构误匹配**：长走廊、树木、门窗等让多个地点都产生较高相似度。

序列约束利用的不是“某张图更像”，而是：

> 一串 query 是否能在 database 中找到一条运动连续、整体得分都高的轨迹。

这相当于用时间上下文给单帧外观证据增加一个运动一致性先验。

---

## 2. 从相似度矩阵理解对角线

设 query 有 `M` 帧，database 有 `N` 帧，全局描述子已经 L2 归一化：

```math
Q ∈ R^{M×D},  D ∈ R^{N×D}
```

余弦相似度矩阵为：

```math
S = QD^T,  S(i,j) = q_i^T d_j
```

矩阵的行是 query 时间，列是 database 时间。假设两次遍历的采样速率和行进速度
相近，而且 query 第 `i` 帧对应 database 第 `j` 帧，那么：

```text
q_i     -> d_j
q_{i+1} -> d_{j+1}
q_{i+2} -> d_{j+2}
```

正确匹配会在 `S` 上形成连续高分对角线：

```text
database index j ─────────────────────────►

query i       · · █ · · · ·
      i+1     · · · █ · · ·
      i+2     · · · · █ · ·
      i+3     · · · · · █ ·
                correct trajectory
```

错误的单帧高分通常只是孤立亮点：

```text
query i       · · █ · · · ●   <- accidental high score
      i+1     · · · █ · · ·
      i+2     · · · · █ · ·
```

沿候选轨迹聚合后，连续对角线被增强，孤立亮点被周围低分稀释。

---

## 3. 项目中的数学形式

对于候选匹配 `(i,j)`、窗口 `D` 和索引速度比 `v`，项目计算：

```math
S_v^{seq}(i,j)
= 1 / |D_{valid}|
  Σ_{δ ∈ D_{valid}}
  S(i+δ, j+round(vδ))
```

然后在多个速度假设上取最大值：

```math
S^{seq}(i,j) = max_{v ∈ V} S_v^{seq}(i,j)
```

这里有四个需要说准确的细节。

### 3.1 `v` 是索引速率比

代码中的 `v` 表示：

```text
database index displacement / query index displacement
```

也就是 query 每前进一帧，database 轨迹预计前进多少个索引。只有在两边采样率
一致、帧间运动近似均匀时，它才能近似解释为物理速度比。

例如：

- `v=1.0`：query 前进 10 帧，database 也前进约 10 帧；
- `v=0.5`：query 前进 10 帧，database 前进约 5 帧；
- `v=1.5`：query 前进 10 帧，database 前进约 15 帧。

如果视频帧率不同、存在丢帧或按距离而非时间采样，`v` 与真实线速度不能直接
画等号。面试时把它称为 **index-rate ratio** 最严谨。

### 3.2 为什么需要 `round`

矩阵列是离散索引，而 `vδ` 可能不是整数，所以代码用：

```python
col_shift = int(round(velocity * offset))
```

这是一种最近邻离散化。它简单，但会产生速度 aliasing：当窗口较短时，`v=0.9`
与 `v=1.0` 可能在多个 offset 上得到完全相同的整数位移。

### 3.3 为什么边界要除以真实计数

矩阵开头和结尾无法获得完整窗口。例如 causal window 为 15 时，第 3 个 query
只有 4 帧历史。如果仍除以固定的 16，边界分数会被系统性压低。

代码为每个元素维护 `counts`，最后计算：

```python
accumulated / counts.clamp(min=1)
```

因此边界位置使用实际参与聚合的项数。

### 3.4 多速度为什么取最大值

每个 `(i,j)` 都允许选择最支持自己的速度假设：

```math
max_v S_v^{seq}(i,j)
```

它相当于对速度做离散 latent-variable search，避免预先知道两次遍历的速度比。
但速度候选越多，错误地点也有越多机会“碰巧找到一条高分轨迹”。因此多速度
搜索增加覆盖范围的同时，也增加 multiple-hypothesis bias。

---

## 4. 因果与非因果窗口

### 4.1 非因果模式

```python
offsets = range(-window, window + 1)
```

对于当前 query `i`，同时使用过去和未来：

```text
i-window ... i-1, i, i+1 ... i+window
```

优点是上下文对称、信息更多；缺点是必须等待未来 `window` 帧才能输出当前结果，
不符合实时 SLAM 的即时决策条件。

### 4.2 因果模式

```python
offsets = range(-window, 1)
```

只使用：

```text
i-window ... i-1, i
```

因此 query `i` 的得分不依赖 `i+1` 之后的任何图像。项目测试会修改未来行并确认
过去输出完全不变，以防未来信息泄漏。

### 4.3 因果不等于已经实现在线系统

这是最容易被追问的地方。

当前 `sequence_rerank` 接收完整的 `[M,N]` 相似度矩阵，并一次性返回完整矩阵。
它具有 **causal semantics**，因为每一行只读取过去；但实现方式仍是离线 batch
computation，不是持续接收单帧、维护状态的 streaming node。

真正的在线版本应该：

- 保存最近 `window+1` 行相似度的 ring buffer；
- 每来一个 query 只更新当前行；
- 只对稀疏候选或局部 database 区域计算轨迹分数；
- 维护时间戳、里程计和有效 database 索引；
- 输出当前帧 proposal，而不是保存完整 `M×N` 矩阵。

因果模式没有未来帧等待延迟，但前 `window` 帧处于 **history warm-up**，上下文
不完整；非因果模式才具有明确的未来帧等待延迟。

---

## 5. 代码逐段对应

### 5.1 单一速度的轨迹聚合

入口：`_aggregate_along_line(similarities, window, velocity, causal)`。

对于每个时间 offset：

```python
row_shift = offset
col_shift = int(round(velocity * offset))
```

然后计算同时满足 query 和 database 边界的目标区域：

```python
i_start = max(0, -row_shift)
i_end = min(num_queries, num_queries - row_shift)
j_start = max(0, -col_shift)
j_end = min(num_database, num_database - col_shift)
```

源块相对目标块发生位移：

```python
accumulated[target] += similarities[source_shifted_by_offset]
```

这里不能把 source 和 target 写成相同切片。否则只是把每个矩阵元素重复加多次，
所有候选的排序完全不变，代码表面运行正常但实际上没有加入任何序列约束。

### 5.2 多速度包装

`sequence_rerank` 对每个 `velocity` 调用一次聚合，然后：

```python
best = torch.maximum(best, scored)
```

返回的仍是 `[M,N]` 分数矩阵，所以后续 `torch.topk`、几何验证和 gate 都不需要
理解序列模块内部细节。

### 5.3 与 evaluator 的接口

```bash
python -m src.evaluate ... \
  --seq-window 15 \
  --seq-causal \
  --seq-velocities 0.9,1.0,1.1
```

- `--seq-window 0`：关闭序列约束；
- `--seq-causal`：只使用历史 query；
- `--seq-velocities`：逗号分隔的索引速率假设。

序列重排必须在 `topk` 之前作用于完整候选分数，否则如果先保留单帧 Top-K，真正
正确但单帧分数稍低的轨迹可能已经被不可逆地删掉。

---

## 6. 复杂度与工程代价

设：

- query 数量为 `M`；
- database 数量为 `N`；
- 速度假设数量为 `|V|`；
- causal 窗口项数约为 `W+1`，非因果约为 `2W+1`。

当前 dense 实现的时间复杂度为：

```math
O(|V| · W · M · N)
```

主要内存复杂度为：

```math
O(M · N)
```

虽然每次循环使用的是 PyTorch 块运算，而不是 Python 逐元素循环，但 database
扩大到百万级 keyframe 时，完整相似度矩阵和逐速度 dense 聚合都不可接受。

生产化方向包括：

1. 全局 ANN 先产生稀疏候选，再做局部序列验证；
2. 只维护最近 query 的 ring buffer；
3. 根据里程计给出速度和搜索区间先验；
4. 使用 HMM/Viterbi 或动态规划维护有限条候选轨迹；
5. 按时间戳或累计里程而不是数组位置定义运动。

---

## 7. 实验到底证明了什么

### 7.1 闭集结果

held-out `night_right` 070–099：

| 配置 | R@1 | R@5 | R@100%P |
|---|---:|---:|---:|
| DINOv2 patch-mean | 0.933 | 0.967 | 0.900 |
| + causal sequence | **1.000** | 1.000 | **1.000** |
| + non-causal sequence | **1.000** | 1.000 | **1.000** |

在这个数据集上，因果版已经获得全部增益，说明未来信息不是达到该结果的必要
条件。但样本只有 30 个 query，不能据此断言因果与非因果普遍等价。

### 7.2 打乱时序对照

历史实验中，相同 baseline 特征加入序列匹配后 R@1 达到 1.000；打乱 query
顺序后再做相同聚合，R@1 降到 0.300。

这个对照说明增益确实来自轨迹的时间结构，而不是单纯对相似度做平滑就会变好。

### 7.3 多速度负结果

将速度范围扩大到 `0.5–1.5` 后，night R@1 从 0.890 降到 0.780。

原因不是“多速度代码错误”，而是搜索空间扩大带来的统计代价：正确轨迹多了速度
容错，错误候选同样多了找到高分路径的机会。没有速度 prior 时，`max over v`
会系统性抬高部分误匹配。

### 7.4 冻结 gate 结果

新的 validation/test 分离实验得到：

| 配置 | Test R@1 | AUPRC | Frozen-gate precision | recall | F1 |
|---|---:|---:|---:|---:|---:|
| DINOv2 single frame | 0.820 | 0.791 | 0.848 | 0.848 | **0.848** |
| + causal sequence | **0.920** | **0.947** | **0.958** | 0.500 | 0.657 |

序列约束明显改善排序和整条 PR 曲线，但 validation 学到的阈值在 test 路线后半段
过严。这说明：

> temporal aggregation improves ranking，不等于它产生了跨地点稳定校准的分数。

序列平均会改变分数分布；窗口完整度、场景纹理、速度稳定性和轨迹位置都可能影响
分数尺度。因此部署阈值必须在更多路线和更多条件下验证。

---

## 8. 必须主动披露的实现边界

### 8.1 Gardens Point 对序列假设过于友好

两次遍历基本逐帧对齐、方向相同、速度稳定，正确轨迹天然接近 `v=1` 对角线。
真实机器人可能遇到：

- 不同帧率和大量丢帧；
- 停车、急转弯、倒车；
- 路线分叉或只部分重叠；
- database keyframe 非均匀采样；
- 动态物体与长期外观变化。

所以 100% R@1 更像是“该数据满足模型假设”的证据，不是通用性能上界。

### 8.2 当前版本不是完整 SeqSLAM

本项目只实现核心的沿轨迹聚合与速度搜索。经典 SeqSLAM 还包含局部对比度
归一化等步骤，用于抑制不同图像或路线区域的相似度尺度差异。

准确说法是：

> simplified SeqSLAM-style sequence re-ranking

而不是“完整复现 SeqSLAM”。

### 8.3 数组相邻不一定等于时间相邻

当前实现默认 query 行和 database 列都按时间排序，并且相邻数组位置代表相邻
时刻。它没有接收 timestamp 或真实 frame index。

这意味着：

- 中间丢一帧会被当作时间仍然连续；
- keyframe 采样不均匀时，固定 `v` 不再代表固定运动；
- 不同序列不能拼接后直接放进同一个相似度矩阵。

### 8.4 合成 open-set 删除会制造 database gap

当前 Gardens Point 开集协议会从 database 中移除连续帧。移除后 tensor 的列会
被压缩，例如 `Image069` 与 `Image080` 在数组中变成相邻列，但真实索引相差 11。

序列算法仍按“相邻列等于相邻时间”聚合，因此跨越删除区间的轨迹不再具有正确
的时间几何。这是当前 sequence open-set 实验的一个混杂因素，尤其会影响缺口
附近的 query。

因此冻结 gate 的结果应该被称为诊断实验，而不是最终 sequence benchmark。
更严谨的解决方式是：

1. 使用真实不重叠 query/distractor 构造 open set，不破坏 database 时间轴；或
2. 让序列模块显式接收 frame index/timestamp，按真实时间寻找轨迹位置；或
3. 将每个连续 database segment 独立聚合，禁止轨迹跨越缺口。

这也是下一轮最应该修复的序列评测问题。

### 8.5 相似度证据不是几何或位姿约束

序列模块只说明一条外观匹配轨迹是否连续。重复走廊可能在多帧上都保持错误但
连续，因此 sequence consistency 不能替代：

- 局部特征几何验证；
- Essential Matrix / 相对位姿恢复；
- pose graph 中的时空一致性检查。

序列约束属于候选排序与置信度证据，不是最终 SLAM 后端约束。

---

## 9. 高频面试问题

### Q1：为什么不直接对每一帧的 Top-1 分数做时间平均？

因为相邻 query 的单帧 Top-1 可能落在完全不同的 database 位置。只平均 Top-1
分数会保留“每帧都很自信，但位置不断跳变”的错误。

本项目针对每个候选 `(i,j)` 沿一条具体 database 轨迹聚合，约束的是位置演化，
不是仅约束分数随时间平滑。

### Q2：为什么正确匹配一定是直线？

它不一定。直线来自窗口内索引速率 `v` 近似恒定的局部模型。如果速度持续变化，
正确路径会弯曲。短窗口可以把平滑曲线局部近似成直线；更一般的方法需要动态
规划、HMM 或里程计条件下的非线性轨迹模型。

### Q3：`v=1` 为什么在 Gardens Point 上这么强？

因为两次遍历帧数相近、逐帧对齐且速度稳定，数据天然满足单位斜率假设。它说明
时序信息被正确利用，但也意味着该数据对序列方法偏友好。

### Q4：causal 模式有没有未来信息泄漏？

对单个输出行的数学依赖而言没有：offset 只取 `[-window,0]`，测试也验证修改
未来 query 不会改变过去输出。但整个函数当前一次接收完整矩阵，所以它是具有
因果语义的 batch 实现，不是完整 streaming deployment。

### Q5：为什么扩大速度范围反而下降？

因为系统对每个候选取所有速度中的最大分数。速度假设越多，正确候选得到更多
容错，错误候选也得到更多“试错次数”。没有速度先验或路径复杂度惩罚时，最大值
会产生 multiple-comparisons effect。

### Q6：如何避免多速度搜索抬高假阳性？

可以：

- 用里程计提供中心速度与窄搜索范围；
- 给偏离先验的速度加入 penalty；
- 在一段轨迹上要求速度状态连续，而不是每个 `(i,j)` 独立选速度；
- 用 validation 校准不同速度集合下的分数；
- 使用 HMM/Viterbi 联合估计位置和速度状态。

### Q7：窗口越大是否一定越好？

不是。大窗口能平均更多单帧噪声，但会：

- 增加冷启动长度和非因果等待延迟；
- 更难满足窗口内恒速假设；
- 跨过转弯、停车、路线分叉和 database gap；
- 增加计算量；
- 让短暂但真实的回环响应变慢。

窗口大小是 bias–variance 与响应速度之间的权衡。

### Q8：为什么 sequence score 的阈值迁移会失败？

序列平均后的尺度受窗口完整度、局部场景可辨识度、速度匹配程度和错误轨迹数量
影响。validation 前半段学到的 `0.792` 到 test 后半段变得过严，说明 score
distribution 随路线位置发生漂移。AUPRC 改善只能证明排序更好，不能证明固定
阈值已经校准。

### Q9：删除 database 帧构造 open set 有什么问题？

除了它不是自然 distractor，还会压缩 database 列，使数组相邻关系不再等于时间
相邻关系，直接破坏当前序列模型的几何假设。更好的实验应使用不重叠 query，或
按真实 timestamp/frame index 聚合。

### Q10：复杂度是多少，如何扩展？

当前 dense 算法约为 `O(|V|WMN)` 时间和 `O(MN)` 内存。扩展到大地图时应先用
ANN 得到稀疏候选，再对局部轨迹做 sequence verification；在线端只维护有限
历史窗口和少量候选轨迹。

### Q11：为什么不用 RNN 或 Transformer？

当前目标是建立一个行为可解释、无需额外训练的时序基线。对角线聚合能明确暴露
窗口、速度和因果性假设，适合小数据和消融。学习型时序模型可能表达更复杂运动，
但需要更多轨迹、严格防止路线记忆，并且更难解释错误来自外观还是时序模型。

### Q12：序列约束和 pose graph temporal consistency 有什么区别？

这里的序列约束工作在图像检索分数上，用来重排或判断回环候选。Pose graph
consistency 工作在相对位姿和图优化约束上，检查加入该回环后是否与已有轨迹一致。
前者是前端外观/时间证据，后者是后端几何一致性，不能互相替代。

---

## 10. 两分钟讲法

> 单帧 VPR 容易被重复走廊或瞬时光照误导，但机器人运动是连续的。我把 query
> 和 database 描述子的余弦相似度组成二维矩阵；如果两次遍历对应同一路线，正确
> 匹配会沿斜率为索引速率比 `v` 的对角线形成连续高分带。因此我沿候选轨迹做
> 窗口平均，并实现只读取过去帧的 causal 模式和多速度搜索。闭集上 DINOv2 的
> R@1 从 93.3% 提升到 100%，打乱 query 后降到 30%，证明增益来自时序结构。
> 但扩大速度范围会增加错误轨迹试错机会，而且 validation/test 分离后发现，序列
> 虽把 test R@1 从 82% 提到 92%、AUPRC 从 0.791 提到 0.947，冻结阈值 F1
> 反而下降，说明排序改善不等于阈值可迁移。当前实现还有一个明确边界：它按数组
> 位置而不是真实 timestamp 聚合，删除 database 帧构造 open set 会破坏时间轴。
> 下一步应改成 timestamp-aware sparse sequence tracking，并加入 odometry prior。

这段回答同时覆盖了动机、公式、实现、实验证据、负结果和下一步，通常比只说
“我实现了 SeqSLAM”更能经受追问。

---

## 11. 复现清单

### 单帧基线

```bash
python -m src.evaluate \
  --database outputs/v3/dino_mean_database.pt \
  --query outputs/v3/dino_mean_query.pt \
  --top-k 10 --tolerance 3 \
  --split-name night_right --min-index 70 --max-index 99
```

### 因果序列

```bash
python -m src.evaluate \
  --database outputs/v3/dino_mean_database.pt \
  --query outputs/v3/dino_mean_query.pt \
  --top-k 10 --tolerance 3 \
  --split-name night_right --min-index 70 --max-index 99 \
  --seq-window 15 --seq-causal
```

### 非因果序列

```bash
python -m src.evaluate \
  --database outputs/v3/dino_mean_database.pt \
  --query outputs/v3/dino_mean_query.pt \
  --top-k 10 --tolerance 3 \
  --split-name night_right --min-index 70 --max-index 99 \
  --seq-window 5
```

### 多速度搜索

```bash
python -m src.evaluate \
  --database outputs/v3/dino_mean_database.pt \
  --query outputs/v3/dino_mean_query.pt \
  --top-k 10 --tolerance 3 \
  --split-name night_right \
  --seq-window 15 --seq-causal \
  --seq-velocities 0.5,0.75,1.0,1.25,1.5
```

完整消融入口：

```bash
bash scripts/run_ablation_v3.sh closed
```

相关代码与测试：

- `src/sequence_match.py`：核心聚合与多速度搜索；
- `src/evaluate.py`：相似度矩阵、Top-K 和 gate 集成；
- `tests/test_sequence_match.py`：因果性与 identity 行为测试；
- `docs/experiments.md`：历史消融和负结果；
- `docs/v4_evaluation_and_gating.md`：序列分数与冻结 gate 的关系。
