# LoopLens — 面向 SLAM 回环检测的视觉地点识别

[English](README.md) | **简体中文**

[![test](https://github.com/PchT4129/loop-lens/actions/workflows/test.yml/badge.svg)](https://github.com/PchT4129/loop-lens/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

一个 SLAM 回环检测前端，外加一个从零手写的位姿图后端。给定当前帧，它从历史关键帧中
检索出同一地点，判断这个匹配是否可信，恢复带尺度的相对位姿，并据此校正整条轨迹。

这个问题难在两个方向相反的失效模式：同一地点在白天和夜晚可能看起来完全不同，而两条不同的
走廊却可能几乎一模一样。代价也是不对称的——漏掉一次回环只是推迟一次校正，而一次错误的回环
会把两个不同的地方"焊"在一起，可能让整张地图变形。

## 结果一览

| 指标 | 基线 | 本项目 |
| --- | --- | --- |
| 夜间地点识别，留出集 Recall@1 | 63.3%（微调后的 ResNet18） | **93.3%** 零训练 DINOv2 → 加因果序列匹配后 **100%** |
| 昼夜几何验证（正确 / 错误匹配的内点数区分度） | 1.1×（ORB，等同随机） | **6.8×**（DISK + LightGlue，RANSAC 条件不变） |
| KITTI 00 轨迹误差（ATE RMSE，全程 3.7 km） | 4.335 m（仅里程计） | **1.956 m**（ORB-SLAM3 自带回环：1.204 m） |
| KITTI 00 上被接受的回环边的准确度 | — | 763 条中 **97%** 距真值 5 m 以内 |
| 阈值在测试前冻结的接受 / 拒绝决策 | — | deployed F1 **0.848** [0.747, 0.926] |

![KITTI 00 轨迹](outputs/visualizations/kitti00_loop_closure.png)

这些都是小样本结果：一条校园路线上 30–50 个留出查询，一个查询就能让 Recall@1 变动 3.3 个百分点。
因此每个精度结论都附带 bootstrap 置信区间，阈值在验证段上拟合、测试前冻结，负结果保留而不删除。

## 系统怎么工作

五个阶段。每个阶段都有一个经典实现和一个现代实现，共用同一套命令行接口；每一次替换都由
上一阶段的测量结果驱动，而不是因为"新的更好"。

| 阶段 | 经典 | 现代 |
| --- | --- | --- |
| ① 全局检索 | ResNet18 + GAP / GeM | DINOv2 ViT-S/14，零训练 |
| ② 序列重排序 | — | 沿相似度矩阵对角线做因果聚合 |
| ③ 几何验证 | ORB + RANSAC | DISK + LightGlue + RANSAC |
| ④ 接受 / 拒绝 | — | 相似度（+ 内点率）门控，在验证段拟合、测试时冻结 |
| ⑤ 位姿图 | — | 深度 → PnP → SE(3) 回环边；在 SE(3) 上做高斯-牛顿优化，不依赖 g2o / GTSAM |

## 测量改变了什么

1. **基于验证集的模型选择，比任何模型改动都重要。** 训练损失降到 0.002 的同时，验证集 Recall@1
   从 0.80 掉到 0.30——固定的 5 轮训练多练了 4 轮，而只看损失完全发现不了。
2. **微调结果是一个诊断，而不是一个结果。** 34.8 个百分点的泛化差距说明模型只是记住了 60 帧。
   换成一个本身就泛化得好的特征——零训练的 DINOv2——提升了 30 个点，而微调只提升了 13 个点。
3. **把一个负结果变成可检验的命题。** 昼夜之间，ORB 的内点数与随机无异。在 RANSAC 判据完全不变
   的前提下只替换描述子和匹配器，区分度从 1.1× 恢复到 6.8×——失效的是描述子，不是 RANSAC。
4. **Recall@1 会掩盖置信度。** DINOv2 的 CLS token 和 patch 均值池化在 Recall@1 上同为 0.933，
   但 Recall@100%Precision 相差 5.4 倍——而一次错误回环就可能撕裂地图时，后者才是关键。
5. **Oracle 数字只是上界。** 直接在被评测的那一段上扫阈值，可以做到 100% 精度（对应召回 0.686）；
   改为在验证段拟合、冻结后应用到测试段，序列匹配门控的精度是 0.958，并放进了一个误报。
6. **KITTI 上剩余的误差在后端。** 把检测到的回环对或估计的相对位姿换成真值都没有改善
   （1.956 → 2.036 / 2.029 m）。而且信息矩阵不是装饰：用单位矩阵时，即使回环边全是真值，
   误差也只能降到 3.17 m。

## 推理剖析与部署优化

[`deploy/`](deploy/README.md) 下的后续子项目回答的问题是：当这个前端和机器人上的其他模块共享
GPU 预算时，每一级推理优化——尤其是低精度量化——对 VPR 精度的真实代价是什么。从剖析一路做到
TensorRT FP16、INT8、FP8，NVFP4 做到模拟。

- **FP16 免费，全模型 INT8 不行，选择性 INT8 介于两者之间。** FP16 TensorRT 套上 CUDA Graph 后 batch 1
  只要 **0.51 ms，比 FP32 快 6.6 倍**，任务指标逐位不变。全模型 INT8 让 held-out Recall@1 从 0.933 掉到 0.633：
  DINOv2 有几个离群通道，最大达到典型通道的 19 倍，逐张量 INT8 扛不住。逐组研究发现 MLP 的第二层（GELU 之后）最敏感；
  其余 Linear 用 SmoothQuant 做 INT8，**比 FP16 再快 14%、省 24% 显存，任务指标落在 FP32 的置信区间内**，
  但分数漂移已是门控决策余量的 2.5 倍。
- **FP8 和 INT8 的弱点正好相反。** 在激活上 FP8 比 INT8 准 17 倍（浮点格点容得下离群值），在权重上反而差 7 倍
  （只有 3 位尾数）。所有 Linear 都用 FP8，速度与选择性 INT8 相同，显存只要 **50 MB，比 FP16 省 58%**，
  任务指标落在 FP32 的置信区间内。NVFP4 在模拟里就已越过拐点（Recall@1 0.867），而且它的 ONNX 导出会静默丢掉
  全部量化器——现在由一道检查拦住。
- **batch 1 是 CPU 发射受限的。** 从 FP32 换到 BF16，GPU 上的计算快了 3.5 倍，但延迟只改善 1.29 倍——
  GPU 约 70% 的时间在空等 CPU 逐个发射约 170 个 kernel。起作用的是 CUDA Graph 而不是精度——
  在 TensorRT 下也一样。
- **DINOv2 位置编码插值中的一个内存布局问题**，让一个 bicubic kernel 比必要的慢 7.8 倍，
  在 BF16 batch 1 下约占全部 GPU 工作的 46%。把插值结果固化为常量即可消除，且输出逐位一致。
- **精度评估完全复用本仓库原有的评测协议**：FP32 参考逐位复现了 deployed F1 0.848。

![一次前向的时间线：batch 1 vs batch 32](deploy/results/timeline_b1_vs_b32.png)

## 快速开始

```bash
conda create -n vpr-loop-closure python=3.11 && conda activate vpr-loop-closure
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

bash scripts/run_smoke.sh             # 仅 CPU 的单元测试，不需要数据集
python scripts/prepare_gardens_point.py \
  --day-left /path/to/day_left --day-right /path/to/day_right \
  --night-right /path/to/night_right
bash scripts/run_ablation_v3.sh       # 复现留出集消融
```

所有命令、参数和评测协议见 [`docs/usage.md`](docs/usage.md)（英文）。

## 延伸阅读

| 文档 | 内容 |
| --- | --- |
| [`docs/results.md`](docs/results.md) | 全部结果表、定性示例、讨论与后续计划（英文） |
| [`docs/usage.md`](docs/usage.md) | 安装、数据集、全部命令、SLAM 集成（英文） |
| [`docs/experiments.md`](docs/experiments.md) | 完整消融，包括负结果 |
| [`docs/v4_evaluation_and_gating.md`](docs/v4_evaluation_and_gating.md) | 冻结门控评测协议详解 |
| [`docs/sequence_matching.md`](docs/sequence_matching.md) | 序列匹配：推导、假设、失效模式 |
| [`deploy/README.md`](deploy/README.md) | 推理剖析与部署优化子项目（英文） |
| [`deploy/EXPERIMENTS.md`](deploy/EXPERIMENTS.md) | 部署子项目的完整实验台账 |

## 局限

- 只有一条校园路线、30–50 个留出查询；尚未在标准基准（Nordland、Pitts30k、MSLS）上验证——
  基于米制真值的适配器已经写好，但还没有跑。
- 开集评测是通过移除数据库中一段连续帧构造的，而不是加入真实的干扰项。
- 在冻结的测试段上，序列 + 几何的联合门控相比单独的相似度门控没有增益。
- 位姿图没有地图点，也不做光束法平差，这是与 ORB-SLAM3 剩余差距的来源。

## 许可证

[MIT](LICENSE)
