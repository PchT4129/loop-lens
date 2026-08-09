# 术语速查表

> 覆盖 M0–M6 与附录 A1–A3 出现的全部术语。面试前扫一遍。
>
> **"出处"列**指向详细解释所在的章节。

---

## ⚠️ 先看这条：一个非标准用词

| 我用的词 | 状况 | 面试时该怎么说 |
|---|---|---|
| **判别间隙** | ⚠️ **这不是标准术语，是我为了讲解方便造的词**（指"同地点对的平均相似度 − 全体对的平均相似度"） | 学术上标准的表达是 **类内距离 vs 类间距离**（intra-class / inter-class distance）、或**可分性（separability）**。<br>建议说法：「我比较了**类内相似度和类间相似度的差距**——夜间是 0.081，白天是 0.157，差了一半」 |

概念本身完全成立，只是**别用"判别间隙"这四个字**，面试官可能听不懂。其余术语都是标准的。

---

## 1. 任务与场景

| 术语 | 英文 | 一句话 | 出处 |
|---|---|---|---|
| 视觉地点识别 | VPR (Visual Place Recognition) | 给一张图，从数据库找**同一地点**的图 | M1 |
| 回环检测 | Loop Closure Detection | VPR 在 SLAM 中的落地；识别"我以前来过这" | M1 |
| 漂移 | drift | 里程计相对位姿连乘导致误差累积、永不自愈 | M1 |
| 位姿图 | pose graph | 节点=位姿，边=约束；回环给它加一条跨时间的远距离边 | M1 |
| 视觉重定位 | camera relocalization | 不只知道"哪个地点"，还要 6-DoF 位姿。VPR 常是它第一步 | M1 |
| 图像检索 | CBIR | 找**视觉相似**的图。VPR 里相似只是手段不是目标 | M1 |
| 感知混叠 | perceptual aliasing | 不同地点长得几乎一样（相似走廊、重复立面） | M1 |
| 外观/条件变化 | appearance / condition change | 昼夜、季节、天气。本项目主攻的挑战 | M1 |
| 视角变化 | viewpoint change | 横向偏移、朝向变化。`day_right` 隔离的就是这个变量 | M1 |
| 伪真值 | pseudo ground truth | 用图像帧号当位置标签（省掉 GPS 真值） | M1 |
| 几何验证 | geometric verification | 局部特征匹配 + RANSAC，用来剔除 VPR 的误检 | M1/M8 |
| 干扰集 | distractor | 数据库里大量与 query 无关的图。本项目**没有**（闭集设定） | M1 |
| 时序排除 | temporal exclusion | 真实 SLAM 中必须排除最近 N 秒的关键帧，否则检索到的是"我还站在原地" | M5 |

---

## 2. 描述子与特征

| 术语 | 英文 | 一句话 | 出处 |
|---|---|---|---|
| 全局描述子 | global descriptor | 整张图 → 一个定长向量。匹配是 O(N) 内积 | M0/M2 |
| 局部特征 | local feature | ORB/SIFT/SuperPoint 等关键点描述子，用于几何验证 | M2/M8 |
| 骨干网络 | backbone | 负责提特征的主体网络（这里是 ResNet18） | M2 |
| 全局平均池化 | GAP (Global Average Pooling) | 把 512×7×7 的每个通道 49 个位置求平均 → 512 维 | M2 |
| 广义均值池化 | GeM | `(mean(xᵖ))^(1/p)`，p 可学习。**p=1 是 GAP，p→∞ 是 max** | M2 |
| 最大池化描述子 | MAC / R-MAC | 取通道最大值 / 分区域取最大再聚合 | M2 |
| 词袋 | BoW / DBoW2 | 手工特征 + 词汇树 + TF-IDF + 倒排索引。**ORB-SLAM2/3 用的方案** | M2/M8 |
| VLAD / NetVLAD | — | 聚合局部特征到聚类中心的**残差**；NetVLAD 是其可微版本，VPR 里程碑 | M2 |
| CosPlace | — | CVPR'22，把 VPR 当**分类**做（GPS+朝向分组 + 大间隔 softmax） | M2/M6 |
| MixVPR / EigenPlaces / SALAD | — | 更新的聚合方案（MLP-Mixer / 多视角 / 最优传输） | M2 |
| AnyLoc | — | DINOv2 自监督特征 + VLAD，**零训练**、跨域强 | M2 |
| L2 归一化 | L2 normalization | 除以模长，钉在单位球面上。**点积=余弦 + 堵死缩放作弊** | M2/M6 |
| 余弦相似度 | cosine similarity | 两向量夹角的 cos。归一化后 = 点积 | M2/M4 |
| 倒数第二层特征 | penultimate features | 去掉分类头后的表征，迁移学习标配 | M2 |
| 下采样率 | stride / downsampling | ResNet 家族总下采样 **32×**（224→7） | M2 |
| PCA 白化 | PCA-whitening | 去均值 + 去相关 + 方差归一。检索里近乎免费的提升 | M2/M4 |
| 孪生网络 | Siamese network | 多个分支**共享同一套权重**。triplet 的三次 forward 就是它 | M6/A1 |
| 类内/类间距离 | intra- / inter-class distance | ⚠️ 见开头：这才是"判别间隙"的标准说法 | M2 |

---

## 3. 检索与索引

| 术语 | 英文 | 一句话 | 出处 |
|---|---|---|---|
| 向量化 | vectorization | 把逐元素循环改写成整块张量运算。**实测快 1489 倍** | M4 |
| BLAS / SIMD | — | 底层线性代数库 / 一条指令算多个浮点数 | M4 |
| 花式索引 | fancy indexing | 用下标列表索引 tensor，按给定顺序挑行 | M5 |
| 近似最近邻 | ANN | 不和全部比，牺牲一点召回换速度/内存 | M4 |
| FAISS | — | Facebook 的向量检索工具箱，可组合 IVF+PQ 等 | M4 |
| 倒排文件 | IVF (Inverted File) | 先聚成 N 个簇，只搜最近的几个簇。**像图书馆先找书架** | M4 |
| 分层小世界图 | HNSW | 图上贪心搜索，上层高速公路下层小路。**支持在线插入**，适合 SLAM | M4 |
| 乘积量化 | PQ (Product Quantization) | 512 维切 8 段，每段用 256 个"代表"编号。**2048 字节 → 8 字节** | M4 |

---

## 4. 评测

| 术语 | 英文 | 一句话 | 出处 |
|---|---|---|---|
| Recall@K | — | Top-K 里**至少有一个**正确的 query 比例。⚠️ 不是严格 recall，实为 hit rate@K | M1/M5 |
| Precision@K | — | Top-K 里正确的**比例**，再对 query 取平均 | M1/M5 |
| 宏平均 / 微平均 | macro / micro average | 先算个体再平均 / 先加总再相除 | M5 |
| 容差 | tolerance | 帧号差 ≤3 判正确。承认"地点有空间外延" | M1/M5 |
| 零假阳性召回 | Recall@100% Precision | SLAM 回环的经典指标；本项目**做不了**（无拒绝机制） | M1 |
| PR 曲线 / AUC-PR | — | 扫阈值画曲线，信息量远大于单点指标 | M1 |
| 快速失败 | fail fast | 真值解析失败直接 `raise`，不返回默认值继续跑 | M3/M5 |
| 静默失效 | silent failure | 不崩不报警、输出看似正常但结果是错的。**本项目 `--top-k` 那个 bug** | M5 |

---

## 5. 训练与优化

| 术语 | 英文 | 一句话 | 出处 |
|---|---|---|---|
| 梯度 | gradient | "这个参数增大一点，loss 会怎么变" | A1 |
| 学习率 | learning rate | 步子迈多大 | A1 |
| 轮 / 批 / 步 | epoch / batch / iteration(step) | **step = 一次参数更新**。本项目共 **25 步** | A1 |
| 小批量随机梯度下降 | mini-batch SGD | 用 B 个样本的平均梯度更新。B 越大梯度越准，误差按 **1/√B** 降 | A1 |
| 动量 | Momentum | 累积历史梯度，像小球滚下山。抑制震荡 + 冲过平坦区 | A1 |
| AdaGrad / RMSProp | — | 自适应学习率；AdaGrad 分母只增会卡死，RMSProp 换成指数移动平均 | A1 |
| Adam | — | Momentum(管方向) + RMSProp(管步长)。**活跃参数迈小步** | A1 |
| AdamW | — | 把 weight decay 从梯度里**解耦**。现在的默认选择 | A1 |
| 偏差修正 | bias correction | m、v 初始为 0 导致前几步偏小，除以 `(1−βᵗ)` 校正 | A1 |
| 线性缩放规则 | linear scaling rule | batch 翻倍，学习率也翻倍 | A1 |
| 梯度累积 | gradient accumulation | 跑几个小 batch 累积梯度再更新，模拟大 batch | A1/M6 |
| 学习率调度 | lr schedule / warmup / cosine decay | 前期热身后期衰减。**本项目没有** | A1 |
| 灾难性遗忘 | catastrophic forgetting | 大学习率微调会破坏预训练学到的结构 | A1 |
| 度量学习 | metric learning | 不学"这是哪"，只学"哪两个该挨得近" | M6 |
| 三元组损失 | triplet loss | `max(0, d(a,p) − d(a,n) + margin)`。**满足即归零** | M6 |
| 锚/正/负样本 | anchor / positive / negative | 本项目：夜间图 / 同地点白天图 / 远处白天图 | M6 |
| 间隔 | margin | 要求"近至少这么多"。本项目 0.2 ≈ 把当前间隙撑到 **1.9 倍** | M6 |
| 难负样本挖掘 | hard negative mining | 挑难区分的负样本。**随机负样本实测 24% 零梯度** | M6 |
| 半难负样本 | semi-hard negative | 比 positive 远但仍在 margin 内。避免挑到噪声样本训崩 | M6 |
| 退化解 | degenerate solution | 不归一化时"把向量整体放大"就能降 loss 的作弊路径 | M6 |

---

## 6. 泛化与正则化

| 术语 | 英文 | 一句话 | 出处 |
|---|---|---|---|
| 泛化 | generalization | 在**没见过的数据**上表现好。ML 唯一的目标 | A3 |
| 过拟合 / 欠拟合 | overfitting / underfitting | 自由度 >> 约束 / 反之 | A3 |
| 泛化间隙 | generalization gap | 训练集与 held-out 表现之差。**本项目 1.4 点 → 34.8 点** | A3 |
| **留出集** | **held-out set** | **主动扣下来、训练时绝对不碰的数据**。test 和 validation 都属于它 | A3 |
| 训练/验证/测试集 | train / val / test | 更新参数 / 调超参与早停 / 最终评估一次 | A3 |
| 时序泄漏、邻近泄漏 | temporal leakage | 相邻帧几乎是同一张图，**随机划分会泄漏** | A3 |
| 净化划分 | purged split | 训练与测试之间留一段 gap 丢掉。金融时序标配 | A3 |
| 正则化 | regularization | **牺牲训练表现换泛化能力**的任何手段 | A3 |
| 权重衰减 | weight decay / L2 | 罚 `Σw²`，权重趋小 → 函数更平滑。**本项目为 0** | A3 |
| Dropout | — | 随机置零神经元，强迫信息分散存储 | A3 |
| 数据增强 | data augmentation | **本项目完全没有**。昼夜任务最该加 `ColorJitter` | A3 |
| 早停 | early stopping | 验证指标变差时停下。**本项目没有**（无验证集） | A3 |
| 迁移学习 / 预训练 | transfer learning | 借用别的数据集的信息。**预训练本身就是最强的正则化** | A3 |
| 双下降 | double descent | 超过插值阈值后继续增大模型，泛化**反而又变好** | A3 |
| 隐式正则化 | implicit regularization | SGD 本身倾向找平坦、低复杂度的解 | A3 |

---

## 7. PyTorch 工程

| 术语 | 一句话 | 出处 |
|---|---|---|
| `Dataset` / `DataLoader` | 定义"怎么拿一个样本" / 负责批处理、采样、并行加载 | M3 |
| map-style / iterable-style | 实现 `__len__`+`__getitem__` / 实现 `__iter__`（流式数据） | M3 |
| `collate_fn` | 把样本堆成 batch。**对 `str` 保持 `list[str]` 不转换** | M3 |
| `num_workers` | 多进程加载，让 CPU 解码与 GPU 计算重叠 | M3 |
| `pin_memory` / `persistent_workers` | 锁页内存加速 H2D / 避免反复启动 worker | M3 |
| **`model.eval()`** | 改变**模块行为**：BN 用 running stats、Dropout 关闭 | M3 |
| **`torch.no_grad()`** | 关闭 autograd：不建图、不存激活。**实测省 3.2 倍显存** | M3/A2 |
| BN running stats | BN 的 buffer，**in-place 滑动平均更新，`no_grad()` 拦不住** | M3 |
| **计算图** | 前向时记的账本：每步做了什么运算、用了哪些输入 | A2 |
| 自动微分 | 把已知的简单运算导数按链式法则乘回去 | A2 |
| `requires_grad` | 要不要为它算梯度。**有传染性** → 图能从 loss 连回参数 | A2 |
| `grad_fn` | "我是被哪个运算生出来的"。图的骨架 | A2 |
| `is_leaf` | 叶子（参数）才会被填 `.grad`，中间结果算完就扔 | A2 |
| **`.item()` / `.detach()`** | 切断图引用。不加会**攥着整张图**，实测线性涨到 OOM | A2 |
| `retain_graph` | `backward()` 默认销毁图；要反向两次得开这个 | A2 |
| 动态图 / 静态图 | 每次前向现场建图（可用普通控制流）/ 先定义后执行 | A2 |
| `state_dict` / `strict` | 权重字典 / `load_state_dict` 默认严格匹配。**`strict=False` + `pretrained=False` 是危险组合** | M3 |

---

## 8. 本项目关键数字（面试前速记）

### 8.-1 v3 消融表（held-out `night_right` 070-099，30 query）⭐⭐ 最新，优先报这组

| # | 配置 | R@1 | P@5 | R@100%P |
|---|---|---:|---:|---:|
| ① | ResNet18-GAP 预训练 | 0.500 | 0.433 | 0.000 |
| ③ | ResNet18 + triplet | 0.633 | 0.527 | 0.000 |
| ④ | ResNet18 + InfoNCE | 0.633 | 0.593 | 0.000 |
| ⑤ | **DINOv2 CLS 零训练** | **0.933** | 0.753 | 0.167 |
| ⑥ | **DINOv2 patch-mean 零训练** | **0.933** | 0.800 | **0.900** |
| ⑦ | ⑥ + 因果序列匹配 | **1.000** | 0.933 | **1.000** |

**头号结论**：零训练 DINOv2（0.933）比在 60 帧上微调的 ResNet18（0.633）**高 30 个点**。

**⑤ vs ⑥ 那一对**：R@1 完全相同，但 **R@100%P 差 5.4 倍**（0.167 vs 0.900）——patch 聚合的**置信度校准**远好于 CLS。

**几何验证判别力**（32 采样点，两方法同一批配对）：

| 方法 | night_right 正确 vs 错误 | 均值比 | 逐样本可分 |
|---|---|---:|---:|
| ORB | 27.0 vs 25.7 | **1.1×** | **56%**（随机=50%） |
| DISK+LightGlue | 62.5 vs 9.2 | **6.8×** | **72%** |

**开集评测**（移除 db 帧号 30-49，14/100 query 应被拒绝）：

| DINOv2 + | best F1 | R@100%P |
|---|---:|---:|
| 单帧 | 0.874 | 0.233 |
| **因果序列** | **0.912** | **0.686** |
| 序列 + ORB 门控 | 0.859 | 0.000 |
| 序列 + LightGlue 门控 | 0.849 | 0.326 |

**InfoNCE 诊断**（机制成立但收益进噪声）：

```
triplet 零梯度三元组: 10% → 67% → 78% → 92% → 97% → 98%
InfoNCE batch=16: 有效负样本仅 6.6，梯度>1% 的 6.6 → 0.9
InfoNCE batch=60 τ=0.2: 27.1 个负样本，梯度权重完全不衰减
→ 但 held-out 上 triplet 与 InfoNCE 都是 0.633
```

### 8.0 v2 消融表（held-out `night_right` 070-099，30 query）

| # | 配置 | R@1 | R@5 | P@5 | R@100%P |
|---|---|---:|---:|---:|---:|
| ① | Baseline 预训练 | 0.500 | 0.767 | 0.433 | 0.000 |
| ② | v1 triplet（原实现） | 0.567 | 1.000 | 0.493 | 0.133 |
| ③ | **v2**（BN修复+增强+AdamW+早停） | **0.633** | 0.967 | **0.607** | 0.033 |
| ④ | ③+序列匹配（因果/在线） | **1.000** | 1.000 | 0.893 | **1.000** |
| ⑦ | ③+序列+几何验证重排 ⚠️负面 | 0.667 | 1.000 | 0.647 | 0.033 |

**训练曲线（loss 与指标脱节的实证）**：

```
Epoch 1: loss 0.1246 | val R@1 0.8000  <- best
Epoch 5: loss 0.0020 | val R@1 0.3000  -> 早停
```

**几何验证适用边界**：同域 134.7 vs 26.1（可分）｜跨昼夜 25.1 vs 24.4（随机）

**关键对照**：baseline 特征 + 序列匹配也能到 R@1 1.000（说明 100% 来自序列而非训练）；
时序打乱后降到 0.300（说明增益确实来自时序约束）

### 8.1 性能结果（v1 时期，仍可用于讲动机）

| | R@1 | R@5 | R@10 | P@5 |
|---|---:|---:|---:|---:|
| Baseline `day_right`（仅视角变化） | **0.960** | 1.000 | 1.000 | 0.802 |
| Baseline `night_right`（昼夜变化） | **0.510** | 0.740 | 0.870 | 0.392 |

**严格 held-out（070-099，30 个 query）** ← 对外该报的就是这组

| | R@1 | R@5 | R@10 | P@5 |
|---|---:|---:|---:|---:|
| Baseline | 0.500 | 0.767 | 0.900 | 0.433 |
| Triplet（训于 000-069） | **0.567** | **1.000** | **1.000** | 0.493 |

**过拟合证据**

| Recall@1 | 训练区间 000-069 | held-out 070-099 | 间隙 |
|---|---:|---:|---:|
| 训练前 | 0.5143 | 0.5000 | **1.4 点** |
| 训练后 | 0.9143 | 0.5667 | **34.8 点** |

### 8.2 实测数据

| 项目 | 数值 | 出处 |
|---|---|---|
| 类内/类间相似度差距（白天 / 夜间） | **0.157 / 0.081** | M2 |
| 特征非负性 | 512 维全非负，最不相关图对余弦仍有 **0.35** | M2 |
| 归一化前模长（白天/夜间） | 23.34 / 25.61 | M2 |
| train 模式下同图不同 batch 的特征余弦 | **0.796** | M3 |
| 矩阵乘法 vs for 循环 | **快 1489 倍** | M4 |
| 100 万张图暴力检索 | **34 ms，1.9 GB** ← 先卡内存不是计算 | M4 |
| 单图存储 | 512 维 float32 = **2048 字节** | M4 |
| 随机负样本零梯度比例 | **24%**（最难负样本时 1%） | M6 |
| 梯度方向余弦（batch=1 / 16） | **0.163 / 0.620** | A1 |
| 总参数量 / 训练样本 | **1118 万 / 70** ≈ 16 万:1 | A1/A3 |
| 总参数更新次数 | **25 次** | A1 |
| 计算图显存开销（batch=16） | 108 MB → **344 MB**，3.2 倍 | A2 |

### 8.3 关键配置

```
backbone      ResNet18 (ImageNet 预训练), 512-D, L2 归一化
输入          960×540 原图 → Resize(224,224)（非等比，横向压 1.78 倍）
检索          q @ dbᵀ + topk，暴力精确搜索
评测          tolerance ±3 帧，recall_ks=[1,5,10]，precision_k=5
训练          TripletMarginLoss(margin=0.2, p=2), Adam lr=1e-4
              batch=16, epochs=5, positive_tolerance=3, negative_gap=20
划分          训练 Image000-069 / held-out Image070-099（按时序，不随机）
```
