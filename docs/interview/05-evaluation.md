# M5 · 评测：你怎么知道自己做得好不好

> 对应 `src/evaluate.py`（182 行，项目里最长的模块）。
>
> 指标的**定义与取舍**在 M1 讲过；本章讲**实现**——代码怎么一步步算出来的，以及它藏着的一个静默失效。

---

## 1. 评测为什么值得单独讲一章

前面几章讲"怎么做"，这一章讲"**怎么知道做得对不对**"。

> **评测代码是整个项目里唯一不能出错的部分。**
>
> 模型差一点，结论是"效果一般"；**评测错一点，结论就是假的**——而且你不会知道。

M0 里那个例子是活的：README 原本拿"全序列 baseline 51%"比"held-out 57%"，两组 query 不是同一批。数字都真，对比无效。

面试官清楚这点。所以"你怎么评测的"考的不是会不会写循环，而是**你有没有意识到评测本身可能骗你**。

---

## 2. 核心问题：怎么判断一次检索是"对"的

需要**真值**。M1 讲过：三个序列逐帧对齐，**图像编号就是位置标签**。

```
query:   night_right/Image050.jpg   → 位置标签 50
检索到:  day_left/Image048.jpg      → 位置标签 48
                                      |50 − 48| = 2 ≤ 3   ✓ 正确
```

```python
def image_index_from_path(path: str) -> int:
    match = re.search(r"Image(\d+)\.jpg$", path)
    if match is None:
        raise ValueError(f"Could not parse image index from path: {path}")
    return int(match.group(1))


def is_correct_match(query_path, database_path, tolerance) -> bool:
    return abs(image_index_from_path(query_path)
               - image_index_from_path(database_path)) <= tolerance
```

### 2.1 正则里的三个符号

| 符号 | 含义 |
|---|---|
| `(\d+)` | 括号是**捕获组**，`\d+` 是"一个或多个数字"。`group(1)` 取括号里捕到的内容 |
| `\.` | 转义的点。正则里 `.` 表示"任意字符"，匹配真正的点必须转义 |
| `$` | 锚定**字符串末尾**，保证匹配的是文件名结尾，而非路径中间碰巧叫 `Image123.jpg` 的目录 |

### 2.2 `raise` 而不是返回 `None`

解析失败直接抛异常。这是对的——**文件名格式不对意味着整个评测的真值是错的，必须立刻炸掉，而不是悄悄返回默认值继续跑。**

> 这个原则叫 **fail fast**。M3 里 `ImageFolderDataset` 发现空目录抛异常是同一原则。被问"你怎么处理异常"时，能说出"在真值这种一错全错的地方我选择立刻失败而不是容错"，比说"我加了 try-except"好得多。

### 2.3 一个工程小瑕疵

`triplet_dataset.py` 里有一个**几乎一模一样**的 `image_index_from_path`（区别只是接收 `Path`、用 `path.name`）。重复代码，应抽到公共模块。

小事，但被问"这份代码哪里可以改"时是个具体答案。

---

## 3. Recall@K 的实现

```python
for k in recall_ks:                              # recall_ks = [1, 5, 10]
    num_success = 0
    for query_idx, query_path in enumerate(query_paths):
        retrieved_indices = top_indices[query_idx, :k]          # ← 取前 k 个

        hit = any(
            is_correct_match(query_path, database_paths[db_idx.item()], tolerance)
            for db_idx in retrieved_indices
        )
        if hit:
            num_success += 1

    metrics[f"recall@{k}"] = num_success / len(query_paths)
```

### 3.1 `top_indices[query_idx, :k]` 背后的正确性依据 ⭐

> `torch.topk` 返回的结果是**按分数降序排好的**。所以 top-10 结果的**前 1 个**就等于 top-1 结果。

正因如此，代码只需**调用一次** `retrieve_top_k(k=10)`，靠切片就能同时得到 recall@1/@5/@10，不必为每个 K 重跑检索。漂亮的省法。

### 3.2 `any(...)` + 生成器

这就是 Recall@K 的定义："Top-K 里**至少有一个**正确"。

`any()` 配生成器有**短路求值**特性：找到第一个 `True` 立刻停止。如果 top-1 就命中，剩下 9 个连解析都不会解析。

### 3.3 `.item()`

`db_idx` 是 0 维 tensor，`.item()` 转成 Python `int` 才能索引 list。

> 注意：这里加 `.item()` 是**类型需要**，不是 A2 讲的"切断计算图"——这个 tensor 本来就没有梯度。

---

## 4. Precision@K 的实现

```python
total_precision = 0.0
for query_idx, query_path in enumerate(query_paths):
    retrieved_indices = top_indices[query_idx, :precision_k]     # precision_k = 5

    num_correct = sum(
        is_correct_match(query_path, database_paths[db_idx.item()], tolerance)
        for db_idx in retrieved_indices
    )
    total_precision += num_correct / precision_k

metrics[f"precision@{precision_k}"] = total_precision / len(query_paths)
```

和 Recall 只差两处：

| | Recall@K | Precision@K |
|---|---|---|
| 聚合方式 | `any(...)` —— **有没有** | `sum(...)` —— **有几个** |
| 单 query 得分 | 0 或 1（二值） | 0 ~ 1 的小数 |

**`sum()` 对布尔求和**：Python 里 `True == 1`、`False == 0`，`sum` 一串布尔就是"有几个 True"。惯用写法。

**两次除法含义不同**：

```
num_correct / precision_k        ← 这一个 query 的 precision
total_precision / len(queries)   ← 对所有 query 取平均
```

这叫 **macro average（宏平均）**：先算个体指标再对个体平均，每个 query 权重相同。

（micro average 是把所有正确数和检索总数分别加总再相除。此处两者相同，因为每个 query 的 K 一样。）

---

## 5. 怎么切 train/test split

```python
def select_query_indices(query_paths, split_name, min_index=None, max_index=None):
    selected_indices = []
    for index, path in enumerate(query_paths):
        if f"/{split_name}/" not in path:                         # ① 按目录名筛
            continue
        image_index = image_index_from_path(path)
        if min_index is not None and image_index < min_index:     # ② 按帧号筛
            continue
        if max_index is not None and image_index > max_index:
            continue
        selected_indices.append(index)
    return selected_indices
```

**① 按 split 名筛**：用 `"/night_right/"` 子串匹配路径。这就是 M3 里"一次性把两个 split 抽进同一个特征文件"能成立的原因——混着存，靠路径字符串区分。

**② 按帧号区间筛**：`--min-index 70 --max-index 99` 切出 held-out 测试段。M0 那组严格对比就是这么跑的。

> ⚠️ **可移植性问题**：`f"/{split_name}/"` 硬编码 `/` 作路径分隔符。**Windows 上是 `\`，匹配会全部失败**（然后被 `main` 里的 `raise ValueError` 兜住，至少不静默出错）。正确写法是用 `Path(path).parts` 判断。
>
> "你的代码在别的平台能跑吗"是真实存在的面试问题。

---

## 6. `main` 怎么串起来

```python
_, top_indices = retrieve_top_k(query_features, database_features, k=args.top_k)
#  ↑ 对【全部】query 一次性检索

for split_name in split_names:
    split_indices = select_query_indices(query_paths, split_name, args.min_index, args.max_index)
    if len(split_indices) == 0:
        raise ValueError(...)                        # fail fast

    split_query_paths = [query_paths[i] for i in split_indices]
    split_top_indices = top_indices[split_indices]   # ← fancy indexing

    metrics = evaluate_retrieval(
        query_paths=split_query_paths, database_paths=database_paths,
        top_indices=split_top_indices,
        recall_ks=[1, 5, 10], precision_k=5, tolerance=args.tolerance,
    )
```

**关键设计：先对全部 query 检索一次，再按 split 切片。**

为什么对？M4 讲过检索就是一次矩阵乘法，**算 200 个 query 和 100 个 query 成本差别可忽略**。一次算完再切比切完分别算简单，也保证了两个 split 用的是完全相同的 database。

**`top_indices[split_indices]`** 是 **fancy indexing（花式索引）**：用下标 list 索引 tensor，按给定顺序挑出那些行。

---

## 7. ⚠️ 一个静默失效（实测确认）

注意 `main` 里这两行的关系：

```python
_, top_indices = retrieve_top_k(..., k=args.top_k)    # 列数由命令行决定
...
recall_ks=[1, 5, 10],                                 # 但这里是【硬编码】的
```

传 `--top-k 5` 时 `top_indices` 形状是 `[N, 5]`，只有 5 列。然后执行 `top_indices[query_idx, :10]`：

> **Python / PyTorch 切片超出范围时不报错，静默返回它有的那部分。**

所以 `[:10]` 实际只拿到 5 个元素，算出来的"recall@10"其实是 recall@5，**但输出仍然打印 `recall@10:`**。

### 7.1 实测

```
--top-k 10 :  recall@1: 0.5100   recall@5: 0.7400   recall@10: 0.8700   precision@5: 0.3920   ← 正确
--top-k 5  :  recall@1: 0.5100   recall@5: 0.7400   recall@10: 0.7400   precision@5: 0.3920   ← recall@10 是假的
--top-k 1  :  recall@1: 0.5100   recall@5: 0.5100   recall@10: 0.5100   precision@5: 0.1020   ← 三个都是假的
```

`precision@5` 更隐蔽：`--top-k 1` 时分子只统计 1 个检索结果，分母仍除以 `precision_k=5`，结果直接变成原值的 1/5（0.51/5 = 0.102，完全吻合）。

### 7.2 为什么值得单独讲

**"静默失效"的教科书案例**：不崩、不警告、正常输出、数字看起来合理——但它是错的。这类 bug 比崩溃危险得多。

**怎么修**（面试可直接给方案）：

```python
# 方案一：断言，fail fast
assert args.top_k >= max(recall_ks), f"--top-k 必须 ≥ {max(recall_ks)}"

# 方案二：自动取需要的 K
top_k = max(max(recall_ks), precision_k, args.top_k)

# 方案三：把 recall_ks / precision_k 也做成命令行参数，并做一致性校验
```

> 📌 **这是"你觉得自己代码哪里有问题"的最佳答案之一**——同时展示三件事：真的读过自己的代码、理解 Python 切片的静默行为、能给出修法。
>
> **重要**：项目中所有报告的数字都是用 `--top-k 10` 跑的，**所以结论全部有效**。这个 bug 只在用小 top-k 时触发。这点也要说清楚。

---

## 8. 怎么"读"结果数字

### 8.1 Baseline

| Query split | R@1 | R@5 | R@10 | P@5 |
|---|---:|---:|---:|---:|
| `day_right` | 0.960 | 1.000 | 1.000 | 0.802 |
| `night_right` | 0.510 | 0.740 | 0.870 | 0.392 |

**第一行**：R@1=0.96 → 96% 的 query 最像的那张就是对的。R@5=1.00 → **每一个** query 的正确答案都在前 5 名。P@5=0.802 → 前 5 名里平均 4 个是对的（0.802×5≈4.01）。

> M1 算过：±3 下每个 query 约有 7 张正确的 database 图，K=5 < 7，所以 P@5 天花板是 1.0。**0.802 是真实排序质量，没被上限卡住。**

**第二行**：R@1 只有 0.51，R@10 有 0.87，**从 1 到 10 涨了 36 个点**。

> 这个形状本身就是诊断结论：**模型不是找不到，而是排不准。** 呼应 M2 的判别间隙 0.081——分数差被压缩，排序乱，但正确答案还在候选池里。

### 8.2 微调后（严格 held-out 070-099）

| | R@1 | R@5 | R@10 | P@5 |
|---|---:|---:|---:|---:|
| Baseline | 0.500 | 0.767 | 0.900 | 0.433 |
| Triplet | 0.567 | **1.000** | **1.000** | 0.493 |

R@5 / R@10 打满 → 这 30 个 held-out query，**没有一个的正确答案掉出前 5 名**。但 R@1 只涨 6.7 点。

> 完整故事是：**"把正确答案捞进候选池"泛化了，"把它排到第一"没泛化。**
>
> 而 M1 讲过 VPR 在 SLAM 里的职责恰好是前者。所以这个结果是**可用的**。

---

## 9. 这套评测还缺什么

### 9.1 ⚠️ 真实 SLAM 必须做、这里没有的：时序排除

真实回环检测里不能拿当前帧和**刚刚经过的关键帧**比——那当然很像，但那不是"回环"，只是"我还站在原地"。

标准做法是排除最近 N 秒 / 最近 M 个关键帧：

```python
if abs(query_timestamp - db_timestamp) < 30:   # 秒
    skip
```

**为什么本项目不需要**：database 和 query 是**两次独立采集**（`day_left` vs `night_right`），天然没有时序重叠。

> 📌 很好的主动补充点，证明你知道"数据集设定"和"真实部署"的差距。可以说：
> 「我这个设定里 database 和 query 是两次独立采集，所以不存在时序泄漏。但真实 SLAM 里必须排除最近一段时间的关键帧，否则检索到的全是刚刚经过的地方，那不是回环。」

### 9.2 其他缺口（M1 已展开）

- **没有拒绝机制**：永远返回 Top-K，不设相似度阈值 → 算不了 Recall@100%Precision，也无法判断"这是个新地方"
- **闭集设定**：每个 query 必有正确答案，没有 distractor
- **100 个样本**：R@1 一个点 = 一张图，小数点后没意义

---

## 10. 面试怎么答

### Q1：你的评测是怎么实现的？

> 数据集三个序列逐帧对齐，所以图像编号就是位置标签，检索结果的帧号和 query 帧号差在 ±3 内就判正确。Recall@K 用 `any()` 判断 Top-K 里至少有一个正确，Precision@K 用 `sum()` 数有几个正确再除以 K，然后对所有 query 做宏平均。
>
> 实现上有个小设计：因为 `topk` 返回的结果是有序的，我只调用一次检索拿 Top-10，然后靠切片同时得到 recall@1、@5、@10，不用为每个 K 重跑。切 split 是靠路径里的目录名加帧号区间筛，这样 held-out 评测不需要重新抽特征。

### Q2 ⭐：你的评测代码有什么问题吗？

> 有一个静默失效。`main` 里检索的 K 是命令行参数，但 `recall_ks=[1,5,10]` 是硬编码的。如果传 `--top-k 5`，`top_indices` 只有 5 列，而代码里 `[:10]` 这个切片在 Python 和 PyTorch 里超出范围不会报错，会静默返回它有的那 5 个。结果就是算出来的"recall@10"其实是 recall@5，但输出仍然打印 recall@10。`precision@5` 更隐蔽——分子只统计了实际检索到的个数，分母还是固定除以 5。
>
> 我实测确认过：`--top-k 1` 时三个 recall 全部变成 0.51，precision@5 变成 0.102，正好是 0.51 除以 5。
>
> 修法很简单，加个 `assert args.top_k >= max(recall_ks)` 让它 fail fast，或者直接让检索的 K 自动取需要的最大值。另外我确认过项目里所有报告的数字都是 `--top-k 10` 跑的，所以结论是有效的，这个 bug 只在小 top-k 时触发。

### Q3：为什么 `image_index_from_path` 解析失败要抛异常而不是跳过？

> 因为这是真值解析。文件名格式不对意味着整个评测的 ground truth 是错的，这种情况必须立刻失败，而不是悄悄返回默认值继续跑出一堆看似正常的数字。fail fast 在真值和数据加载这类"一错全错"的地方是正确选择。

### Q4：R@1 只有 51% 但 R@10 有 87%，这说明什么？

> 说明模型不是找不到，而是排不准——正确答案往往在候选池里但没排到第一。这和我在特征空间里量化的结果一致：夜间的判别间隙只有 0.081，是白天的一半，正确答案和干扰项的分数差被压缩到了噪声量级，所以排序乱但候选还在。
>
> 这个形状对 SLAM 其实是可接受的，因为 VPR 的职责就是给候选，最终由几何验证拍板。

### Q5 ⭐：你的评测和真实 SLAM 里的回环检测评测有什么差别？

> 最大的差别是**时序排除**。真实系统里不能拿当前帧去和刚刚经过的关键帧比，那些当然长得像，但那不是回环，只是还站在原地，所以必须排除最近 N 秒或最近 M 个关键帧。我这里 database 和 query 是两次独立采集，天然没有时序重叠，所以不需要做这件事——但如果换成单序列自匹配的设定，不做时序排除指标会虚高得离谱。
>
> 另外我的设定是闭集，每个 query 必有正确答案，没有 distractor；也没有拒绝机制，永远返回 Top-K，所以算不了 SLAM 里更关心的 Recall@100%Precision。

### 分方向追问

| 方向 | 追问 | 落点 |
|---|---|---|
| **ML 工程** | 评测代码怎么保证自己是对的？ | 断言 + fail fast + §7 那个 bug |
| **ML 工程** | macro 和 micro average 的区别？ | §4 |
| **CV 检索** | 为什么一次检索能算出所有 K 的 recall？ | §3.1，topk 有序 |
| **SLAM** | 单序列自匹配时怎么评测？ | §9.1 时序排除 |
| **DL 基础** | `any()` 短路求值是什么？ | §3.2 |

---

## 11. 本模块自测

1. 为什么说评测代码是唯一不能出错的模块？
2. 正则 `r"Image(\d+)\.jpg$"` 里三个符号各是什么作用？
3. 为什么解析失败要 `raise` 而不是返回默认值？
4. Recall 用 `any()`、Precision 用 `sum()`，分别对应指标定义的哪一句？
5. 为什么只调用一次 `retrieve_top_k` 就能同时算出 @1/@5/@10？
6. 什么是 macro average？
7. **`--top-k 5` 时 recall@10 会输出什么？为什么不报错？怎么修？**
8. R@1 低但 R@10 高，说明模型的什么问题？和 M2 的哪个指标呼应？
9. 真实 SLAM 评测里必须做而这里没做的是什么？为什么这里不需要？
