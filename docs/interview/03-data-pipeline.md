# M3 · 数据管线与特征库

> 对应 `src/dataset.py` + `src/extract_features.py`。
>
> **本章比 M1/M2 短**，因为三分之二的内容已被附录覆盖——这里只讲剩下的部分，
> 其中第一条是 **PyTorch 最高频的面试题之一**。

## 这章还剩什么

| 内容 | 状态 |
|---|---|
| Dataset / DataLoader 协议 | **已在 [A5](A5-pytorch-basics.md) 从零讲过** |
| 计算图为什么占显存 | **已在 [A2](A2-computation-graph.md) 讲过** |
| BatchNorm 的数学 | **已在 [A7](A7-batchnorm.md) 讲过** |
| **`eval()` 与 `no_grad()` 的配合** | ← **本章核心** |
| **离线特征库的工程设计** | ← 本章 |
| **`build_model` 里那个细节** | ← 本章 |

---

## 1. 为什么"抽特征"是独立的一步

pipeline 长这样：

```
extract_features.py  →  .pt 文件  →  retrieve.py / evaluate.py / visualize.py
     (抽特征)          (存到磁盘)          (下游全都读这个文件)
```

**为什么不端到端跑完，非要中间落个盘？** 三个理由。

### 1.1 对应真实系统的两个阶段

SLAM 里**建图和定位本来就是分开的**：

- **建图（离线/后台）**：把所有关键帧的特征算好存起来
- **定位（在线）**：只算当前这一帧，然后去库里查

**database 特征就是"地图"，只需算一次。** 每次评测都重新过一遍网络是纯粹的浪费。

### 1.2 实验迭代效率

换 backbone、换 checkpoint，**只需重跑抽特征这一步**，下游检索/评测/可视化**一行都不用改**。

> 💡 **这个好处是真实存在的**：M0 里补跑"baseline 在 070-099 上的指标"时，就是**直接复用已有特征文件**，几秒出结果。

### 1.3 关注点分离

下游三个模块只依赖一个契约：

```python
{"features": Tensor[N, 512], "paths": list[str]}
```

**它们完全不知道模型是什么。** 加新 backbone 不需要碰它们。

---

## 2. 本章核心：`eval()` 与 `no_grad()` ⭐

```python
model.eval()                    # ← 在 build_model 里
...
with torch.no_grad():           # ← 在抽特征循环外
    for images, paths in loader:
        features = model(images)
```

**很多人以为这两个是一回事，或以为加一个就够。它们完全正交——管的是两件毫不相干的事。**

### 2.1 `no_grad()` 管"要不要记账"

A2 讲过：前向时 PyTorch 会**一边算一边记账**（构建计算图），为的是之后能反向传播。**推理不需要反向传播，所以这个账可以不记。**

实测（batch=16 跑一次 ResNet18 前向）：

```
加 no_grad()  : 峰值显存 108.2 MB
不加          : 峰值显存 344.0 MB      ← 3.2 倍
```

**它只影响显存和速度，不影响计算结果。**

### 2.2 `eval()` 管"模块的行为"

有些层训练时和推理时行为不同，最典型的是 BatchNorm（A7 §5）：

| | 训练模式 `train()` | 推理模式 `eval()` |
|---|---|---|
| BatchNorm 用什么统计量 | **当前 batch 的均值方差** | **累积的 running_mean/var** |
| Dropout | 随机丢弃 | 关闭 |

**它只影响计算结果，不影响显存。**

### 2.3 缺哪个都不行

| 缺什么 | 后果 |
|---|---|
| 只有 `eval()`，没有 `no_grad()` | **结果正确**，但白白记账，显存 3 倍，大 batch 会 OOM |
| 只有 `no_grad()`，没有 `eval()` | **结果错误，而且是静默的** |

---

## 3. 只加 `no_grad()` 会错成什么样（实测）⭐

拿**同一张图** `Image000.jpg`，分别放进两个不同的 batch：

- **batch A**：这张图 + 7 张**白天**图
- **batch B**：这张图 + 7 张**夜间**图

```
model.eval()  + no_grad():  余弦 = 1.000000   → 完全一致
model.train() + no_grad():  余弦 = 0.796133   → 同一张图，特征不一样了
```

### 3.1 为什么

`train()` 模式下 BN 用**当前 batch 的均值方差**：

- batch A 全是白天图，整体偏亮 → 一套统计量
- batch B 全是夜间图，整体偏暗 → 另一套统计量

**同一张图被两套不同的归一化处理，出来的特征当然不一样。**

### 3.2 为什么这对检索是致命的

> **整个检索系统建立在一个前提上：同一张图必须映射到同一个向量。**

前提一破：

- database 特征在"全白天"的 batch 里抽的
- query 特征在"全夜间"的 batch 里抽的
- **两组特征根本不在同一个坐标系里**

**余弦相似度失去意义。而且不报任何错**——你会得到一堆看起来正常但完全不可信的数字。

---

## 4. 更隐蔽的坑：`no_grad()` 拦不住 BN ⚠️

很多人会想"我加了 `no_grad()`，应该什么都不会被改吧？"**错。** 实测：

```
train 模式 + no_grad()，BN 的 running_mean 变化 L2 = 0.227184
```

### 4.1 为什么

A7 §5 讲过，BN 的 `running_mean/var` 是 **buffer 不是 parameter**：

| | 怎么更新 | `no_grad()` 管得住吗 |
|---|---|---|
| **parameter** | 靠梯度 | **能** |
| **buffer** | **in-place 滑动平均**，不走 autograd | **管不着** |

```python
running_mean = (1 − momentum) × running_mean + momentum × 当前batch均值
                            ↑ 这行和梯度没有任何关系
```

### 4.2 后果比"当次结果错"更严重

**忘记 `eval()` 会永久污染模型的 BN 统计量。**

即使后来发现问题、切回 `eval()`，用的也是**已经被污染过的 running stats**。

> 📌 **这个坑是 M7 §4 那个域污染问题的机制来源**：训练时 anchor 全是夜间图、positive 全是白天图，三次 forward 就是在用三批不同域的数据轮流更新同一套 BN 统计量。

### 4.3 面试标准答法（值得背）

> "`eval()` 和 `no_grad()` 是正交的两件事。`eval()` 切换的是**模块行为**——BatchNorm 从用当前 batch 统计量切换到用累积的 running statistics，Dropout 关闭；`no_grad()` 关的是 **autograd**，不构建计算图、不保存中间激活，省显存也更快。
>
> **推理时两个都要。** 只加 `no_grad()` 不加 `eval()` 结果是错的，而且是静默的——我实测过，同一张图放进两个不同的 batch，train 模式下抽出的特征余弦相似度只有 0.796。对检索任务这是致命的，因为整个系统的前提就是同一张图必须映射到同一个确定的向量。
>
> 还有一个容易忽略的点：**BN 的 running stats 是 buffer，用 in-place 滑动平均更新，不走 autograd，所以 `no_grad()` 拦不住它被污染**——忘了 `eval()` 会永久改掉模型的 BN 统计量。"

---

## 5. `build_model` 里那个细节

```python
def build_model(checkpoint_path, device):
    use_pretrained = checkpoint_path is None          # ← 这一行
    model = ResNet18FeatureExtractor(pretrained=use_pretrained)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model
```

### 5.1 那行在说什么

> **"如果要加载 checkpoint，就别下载 ImageNet 权重了。"**

**因为 checkpoint 会把所有权重完全覆盖**，先加载一遍纯属浪费（下载 45 MB + 读盘 + 赋值）。

### 5.2 为什么安全

`load_state_dict` 默认 **`strict=True`**——**参数名必须完全对得上，少一个多一个都直接报错**。

所以"某些层没被覆盖、还留着随机初始化值"这种情况**不可能悄悄发生**。

### 5.3 ⚠️ 但它有个前提

**如果哪天为了兼容旧 checkpoint 改成 `strict=False`，这个设计就变危险了。**

```python
model = ResNet18FeatureExtractor(pretrained=False)   # 随机初始化
model.load_state_dict(ck, strict=False)              # 缺失的键被静默跳过
```

**缺失的层会保留随机初始化值，模型静默失效**——不报错，但输出是垃圾。

> 📌 **`pretrained=False` + `strict=False` 是个危险组合。**
>
> 被问"你读代码时会注意什么"，能说出这种"**当前安全但改一个默认值就会塌**"的依赖关系，是个不错的答案。

### 5.4 顺带：`map_location="cpu"`

checkpoint **记录了张量原来在哪个设备**。直接 `torch.load` 会尝试恢复到原设备——**在没有 GPU 的机器上会报错**。指定 `cpu` 保证总能加载，之后再 `.to(device)`。标准做法。

---

## 6. 特征库的设计与不足

### 6.1 存的是什么

```python
torch.save({"features": Tensor[N, 512], "paths": list[str]}, output_path)
```

**极简：一个特征矩阵 + 一个路径列表，两者按行对应。**

`paths` 必须存，因为**下游要靠路径知道"检索到的是哪张图"**——`evaluate.py` 从路径解析帧号，`visualize.py` 用路径读图。

### 6.2 三个不足（主动说出来）

| 不足 | 后果 | 生产做法 |
|---|---|---|
| **没有元数据** | 靠文件名后缀区分是哪个 checkpoint 抽的，**容易搞错**；也没记录归一化方式、输入分辨率、backbone 版本 | dict 里存 `model_name` / `checkpoint_hash` / `transform_config` |
| **全量加载进内存** | `torch.load` 一次读全部。100 万张 = 1.9 GB 还行，再大不行（M4 §7） | memmap / 分片 / 向量数据库 |
| **不支持增量插入** | **真实 SLAM 数据库随关键帧不断增长**，这是一次性批处理 | FAISS 的 `index.add()`，或 Milvus/Qdrant |

> 💡 被问"这个设计有什么问题"，说出**"没有元数据版本标记"**和**"不支持增量插入，而真实 SLAM 数据库是增长的"**这两条就够了。

---

## 7. 三个防御性细节

在 `dataset.py` 里，A5 提过，这里点明各自防什么：

| 代码 | 防什么 |
|---|---|
| `sorted(image_paths)` | `rglob` 顺序**依赖文件系统**。不排序则**每次运行特征的行顺序都可能不同**，结果无法复现，`features[i]` 与 `Image{i:03d}` 的对应关系也没了 |
| `.convert("RGB")` | 混入**灰度图（1 通道）、CMYK、带 alpha 的 PNG（4 通道）**时，不转会导致 batch 内通道数不一致直接崩 |
| `if len(...) == 0: raise` | **fail fast**——路径写错时立刻报错，而不是训练到一半才发现数据集是空的 |

### 7.1 一个刻意的设计

```bash
--image-dir data/gardens_point/query      # 一次把 day_right 和 night_right 都抽了
```

`rglob` 是**递归**的，所以一条命令把两个 split 抽进同一个文件。

**这不是偷懒，是刻意的**：

> **两个 split 的特征在同一个文件里，就保证了它们是用完全相同的模型状态抽出来的。**
>
> 分两次抽的话，中间万一模型状态有任何差异，横向对比就不严格了。

`evaluate.py` 靠路径里的 `/day_right/` 或 `/night_right/` 区分它们——这就是 M5 里 `select_query_indices` 的用途。

---

## 8. 面试怎么答

### Q1：PyTorch 的 Dataset 和 DataLoader 各负责什么？

（见 [A5](A5-pytorch-basics.md) §3-4）

### Q2 ⭐：`model.eval()` 和 `torch.no_grad()` 的区别？

（见 §4.3 的完整答法，务必背熟，尤其那个 0.796 的实测数字）

### Q3：为什么推理时 `shuffle=False`？

> 严格说不是正确性必需，因为路径和特征一起返回并同步 append，打乱也不会错位。但推理时 shuffle 没有任何好处——它的意义只在训练时打破样本顺序相关性。保持 `False` 让特征文件行顺序稳定可复现，而且顺序和 `sorted()` 一致，`features[i]` 就对应 `Image{i:03d}`，写分析脚本很方便。

### Q4：为什么把特征存到文件，而不是端到端跑完？

> 三个理由。一是对应真实系统架构：SLAM 里建图和定位本来就是分离的，database 特征是一次性离线成本，在线只需算当前帧。二是实验效率：换 backbone 或 checkpoint 只要重跑抽特征，检索、评测、可视化一行都不用改，评测秒级重跑。三是关注点分离，下游只依赖 `{features, paths}` 这个契约。
>
> 不足也有：没有存元数据，靠文件名后缀区分是哪个 checkpoint 抽的，容易搞错；而且是全量加载、不支持增量插入，但真实 SLAM 的数据库是随关键帧不断增长的，那种场景要换成 FAISS 或向量数据库。

### Q5：`build_model` 里为什么有 checkpoint 时 `pretrained=False`？

> 因为 checkpoint 会把所有权重完全覆盖，加载 ImageNet 权重是白费的下载和读盘。这里安全的原因是 `load_state_dict` 默认 `strict=True`，少任何一个键都会直接报错。**但如果哪天为了兼容旧 checkpoint 改成 `strict=False`，这个组合就危险了**——缺失的层会保留随机初始化的值，模型静默失效。`pretrained=False` 加 `strict=False` 是个危险组合。

### 分方向追问

| 方向 | 追问 | 落点 |
|---|---|---|
| **ML 工程** | 数据加载是瓶颈怎么定位、怎么优化？ | [A5](A5-pytorch-basics.md) §4 + profiler / GPU 利用率 |
| **ML 工程** | 特征库怎么支持在线增量？ | §6.2，FAISS `add()` / 向量数据库 |
| **DL 基础** | BN 在 train 和 eval 下的行为差异？ | §2.2 / [A7](A7-batchnorm.md) §5 |
| **DL 基础** | **`no_grad()` 能阻止 BN running stats 更新吗？** | §4，**不能**，buffer 不走 autograd |
| **SLAM** | 在线阶段的实时性预算怎么算？ | 只算 query 一帧 + 一次矩阵乘 |

---

## 9. 小结

| 要点 | 白话版 |
|---|---|
| **为什么特征要落盘** | ① 对应 SLAM 建图/定位的分离 ② 换模型后下游零改动 ③ 关注点分离 |
| **`no_grad()` 管什么** | **要不要记账**（建计算图）。只影响显存和速度，**不影响结果**。实测省 **3.2 倍显存** |
| **`eval()` 管什么** | **模块行为**（BN 用哪套统计量、Dropout 开关）。只影响结果，**不影响显存** |
| **两者正交** ⭐ | **推理时两个都要**。只有 `eval()` → 显存浪费；只有 `no_grad()` → **结果静默出错** |
| **静默出错有多严重** ⭐ | 实测同一张图在不同 batch 里余弦只有 **0.796**。**检索的前提"同图同向量"被破坏** |
| **`no_grad()` 拦不住 BN** ⚠️ | running stats 是 **buffer 不是 parameter**，**in-place 滑动平均**更新，不走 autograd。忘记 `eval()` 会**永久污染**统计量 |
| **`pretrained = checkpoint is None`** | checkpoint 会全覆盖，省一次下载。安全性依赖 **`strict=True`** |
| **危险组合** ⚠️ | **`pretrained=False` + `strict=False`** → 缺失的层保留随机值，**静默失效** |
| **`map_location="cpu"`** | checkpoint 记录了原设备，不指定的话在无 GPU 机器上会崩 |
| **特征库三个不足** | 无元数据 / 全量加载 / **不支持增量插入**（而真实 SLAM 数据库在增长） |
| **一次抽两个 split** | 刻意设计——**保证两个 split 用的是完全相同的模型状态** |

---

## 10. 本模块自测

1. 特征落盘的三个理由？举一个"它真的帮到了"的具体例子
2. **`no_grad()` 和 `eval()` 各管什么？** 只加其中一个各会怎样？
3. **train 模式下同一张图放不同 batch，特征余弦是多少？为什么？为什么这对检索是致命的？**
4. **`no_grad()` 能防止 BN running stats 被更新吗？为什么不能？** 后果比"当次结果错"严重在哪？
5. `pretrained = checkpoint is None` 为什么安全？什么改动会让它变危险？
6. `map_location="cpu"` 防的是什么？
7. 特征库的三个不足？哪两条最值得在面试里说？
8. `sorted()` / `.convert("RGB")` / 空目录 `raise` 各防什么？
9. 为什么一条命令把两个 split 抽进同一个文件是刻意设计？
