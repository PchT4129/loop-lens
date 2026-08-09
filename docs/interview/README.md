# VPR 项目面试复习笔记

针对 `vpr-loop-closure` 这个 side project 的系统性复习材料。目标：面试中能从**概念**讲到**实现细节**，并且顶得住深挖追问。

每篇统一三段式：**概念原理 → 代码逐行 → 面试怎么答**。

## 速查

- 📖 [GLOSSARY.md](GLOSSARY.md) —— 全部术语速查表 + **本项目关键数字速记**（面试前扫这份）
- 🧪 [../experiments.md](../experiments.md) —— **v2 消融实验**（含负面结果与对照实验），最新数字以这份为准

## 目录

| # | 文件 | 主题 | 状态 |
|---|------|------|------|
| M0 | [00-overview.md](00-overview.md) | 项目全景、结果总表、简历措辞、60 秒电梯陈述 | ✅ |
| M1 | [01-problem-and-metrics.md](01-problem-and-metrics.md) | VPR / 回环检测是什么，数据集与伪真值，Recall@K & Precision@K | ✅ |
| M2 | [02-global-descriptor.md](02-global-descriptor.md) | ResNet18 全局描述子逐行拆解，与 NetVLAD/GeM/DBoW2 对比 | ✅ |
| M3 | [03-data-pipeline.md](03-data-pipeline.md) | Dataset/DataLoader、离线特征库、eval() 与 no_grad() | ✅ |
| M4 | [04-retrieval.md](04-retrieval.md) | 矩阵乘法检索、复杂度、FAISS/HNSW/PQ 大规模方案 | ✅ |
| M5 | [05-evaluation.md](05-evaluation.md) | 评测代码逐行、tolerance 的含义、结果解读与指标局限 | ✅ |
| M6 | [06-metric-learning.md](06-metric-learning.md) | Triplet loss 原理与几何、三元组采样、训练循环 | ✅ |
| **A1** | [A1-training-basics.md](A1-training-basics.md) | **PyTorch 训练基础**：为什么要 batch、mini-batch SGD、Adam、反向传播 | ✅ |
| **A2** | [A2-computation-graph.md](A2-computation-graph.md) | **计算图**：自动微分原理、为什么占显存、`no_grad()` 与 `.item()` | ✅ |
| **A3** | [A3-overfitting.md](A3-overfitting.md) | **过拟合与泛化**：泛化间隙、数据划分与泄漏、正则化五类武器 | ✅ |
| **A4** | [A4-metrics.md](A4-metrics.md) | **Recall@K / Precision@K 从零讲**：对着真实检索结果手算 | ✅ |
| **A5** | [A5-pytorch-basics.md](A5-pytorch-basics.md) | **PyTorch 基础构件**：Tensor/Dataset/DataLoader/nn.Module/怎么搭网络 | ✅ |
| **A6** | [A6-convolution.md](A6-convolution.md) | **卷积与感受野**：参数共享、感受野递推、为什么 512 维是"全局"的 | ✅ |
| **A7** | [A7-batchnorm.md](A7-batchnorm.md) ⭐ | **BatchNorm**：四步公式、γβ 的作用、为什么有效（含被推翻的原论文说法） | ✅ |
| **A8** | [A8-residual.md](A8-residual.md) | **残差连接与梯度消失**：连乘机制、梯度高速公路、为什么叫"残差" | ✅ |
| **A9** | [A9-softmax-crossentropy.md](A9-softmax-crossentropy.md) | **Softmax 与交叉熵**：梯度 `p−y`、PyTorch 的坑、与 triplet 的对照 | ✅ |

### 第一梯队必补（支撑项目叙事的地基）—— 已全部完成 ✅

| 顺序 | 主题 | 一句话 |
|---|---|---|
| ① | [卷积与感受野](A6-convolution.md) | layer4 感受野 435 > 输入 224，**每个位置都看遍全图** |
| ② | [BatchNorm](A7-batchnorm.md) | 原论文的 ICS 解释**已被推翻**，实为**让损失曲面更平滑** |
| ③ | [残差连接](A8-residual.md) | 30 层实验：第 1 层梯度 `2.1e-08` vs `1.39e+01`，**差 6.6 亿倍** |
| ④ | [Softmax 与交叉熵](A9-softmax-crossentropy.md) | 梯度 `p − y`；**交叉熵只保证可分，不保证类内紧凑** |

### 第二梯队（按求职方向补）

| # | 主题 | 状态 |
|---|---|---|
| B1 | [SLAM 几何](B1-slam-geometry.md) —— 相机模型 / 对极几何 / PnP / 位姿表示 / 非线性优化 | ✅ |
| B3 | [对比学习 InfoNCE](B3-infonce.md) —— 自带难样本挖掘 / 温度 τ / 怎么改造本项目 | ✅ |
| B2 | [注意力与 Transformer](B2-attention-transformer.md) —— √d_k / 排列等变 / 多头 / ViT / vs 卷积 | ✅ |
| B4 | [部署与 Python 进阶](B4-deployment-python.md) —— Conv+BN 融合 / 量化 / 装饰器·上下文管理器·生成器 | ✅ |

### 插播（项目无关的通用主题）

| # | 主题 | 状态 |
|---|---|---|
| C1 | [微调 Fine-tuning](C1-finetuning.md) —— 冻结策略 / LoRA / 低秩假设的反例 / 何时不该微调 | ✅ |
| C2 | [RAG 检索增强生成](C2-rag.md) —— 分块 / 混合检索 / **为什么必须两级（与本项目同构）** | ✅ |
| M7 | [07-pitfalls.md](07-pitfalls.md) ⭐ | 过拟合、数据泄漏、hard negative、BatchNorm 陷阱等 | ✅ |
| M8 | [08-slam-integration.md](08-slam-integration.md) | 在真实 SLAM 中的位置、几何验证、DBoW2 对比、SeqSLAM | ✅ |
| M9 | [09-qa-bank.md](09-qa-bank.md) | 分方向面试问答库 + 简历措辞 + 改进 roadmap + **面试前速览** | ✅ |

## 一句话记住这个项目

> 用预训练 CNN 的全局描述子做视觉地点识别（VPR），为 SLAM 提供回环检测候选；
> 用 triplet loss 做度量学习，严格 held-out 下 **Recall@5 从 76.7% 提到 100%**；
> 并通过泛化间隙对照定位出"**召回强、精排弱**"的能力特征，据此引入序列匹配后处理，
> **零训练**将 Recall@1 从 51% 提升至 89%。

## 怎么用这套材料

| 时机 | 看什么 |
|---|---|
| 系统复习 | M0 → M9 顺序过，遇到基础概念卡壳时跳去对应附录 |
| 补基础 | A1 训练机制 / A2 计算图 / A3 过拟合 / A4 指标 |
| **面试前 30 分钟** | **[M9 §7 速览清单](09-qa-bank.md)** + [GLOSSARY §8 关键数字](GLOSSARY.md) |
| 准备特定岗位 | [M9 §3](09-qa-bank.md) 按四个方向分类的问答索引 |
| 改简历 / 练自我介绍 | [M9 §1-2](09-qa-bank.md) |
