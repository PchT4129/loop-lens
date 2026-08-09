# 附录 E · PyTorch 基础构件

> 小白向。从零讲 Tensor / Dataset / DataLoader / nn.Module 以及怎么搭网络。
> 所有例子对应本项目的真实代码。

---

## 1. 全局图：PyTorch 只有四个基础零件

```
① Tensor      数据的载体（多维数组，能上 GPU，能自动求导）
      ↓
② Dataset     定义"一个样本长什么样"      —— 你写
   DataLoader 定义"怎么批量取"           —— 现成的
      ↓
③ nn.Module   模型                       —— 你写
      ↓
④ Loss + Optimizer  怎么衡量、怎么更新    —— 现成的
```

**你真正要写的只有 ② 和 ③。**

---

## 2. Tensor：学会读 shape 就够了

Tensor 就是**多维数组**，和 numpy 的 array 几乎一样，多了两个能力：能放 GPU、能自动求导（A2）。

**最重要的技能不是记 API，是"看 shape 讲故事"。**

```
[16,     3,      224,    224]
 ↑       ↑        ↑       ↑
 batch   通道     高      宽
```

PyTorch 图像约定是 **NCHW**（TensorFlow 是 NHWC，转换时容易错）。

### 2.1 本项目用到的形状操作

| 操作 | 作用 | 在哪 |
|---|---|---|
| `torch.flatten(x, 1)` | 从第 1 维起拍平：`[16,512,1,1]` → `[16,512]` | `models.py` |
| `torch.cat([a,b,c], dim=0)` | 沿第 0 维拼接：三个 `[16,…]` → `[48,…]` | `train_triplet.py` BN 修复 |
| `.chunk(3, dim=0)` | 沿第 0 维切三份（cat 的逆操作） | 同上 |
| `a @ b.T` | 矩阵乘：`[100,512] @ [512,100]` → `[100,100]` | `retrieve.py` |

> 💡 **调试第一法则**：报错时先 `print(x.shape)`。绝大多数 PyTorch 错误都是 shape 对不上。

---

## 3. Dataset：定义"一个样本长什么样"

### 3.1 最小 Dataset 只要三个方法

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self):
        self.data = [1, 2, 3, 4, 5]      # 准备工作

    def __len__(self):                    # "一共多少个样本？"
        return len(self.data)

    def __getitem__(self, index):         # "第 index 个样本是什么？"
        return self.data[index]
```

**为什么必须是这三个？** 因为 DataLoader 只做两件事：先问 `__len__` 有多少个，然后不停喊 `__getitem__(i)` 收集结果。**它对你的数据一无所知，全靠这两个方法。**

（Python 里这叫**协议**：不需要继承特殊接口，实现了这两个方法就"是"一个可索引对象。）

### 3.2 本项目的 `ImageFolderDataset`

```python
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = self._find_images(Path(root_dir))   # ← 只记路径，不读图
        if len(self.image_paths) == 0:
            raise ValueError(...)                              # fail fast

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")   # ← 用到时才读
        if self.transform is not None:
            image = self.transform(image)
        return image, str(image_path)
```

**`__init__` 里只记路径不读图**，叫**惰性加载**。10 万张图全读进内存会直接爆。

### 3.3 transform 为什么要作为参数传进来

```python
transforms.Compose([
    transforms.Resize((224, 224)),   # PIL → 缩放
    transforms.ToTensor(),           # PIL → Tensor，0~255 变 0~1
    transforms.Normalize(mean, std), # 减均值除标准差
])
```

因为**训练和推理需要不同处理**——这正是 v2 做的：

```python
get_train_transform()    # 带 ColorJitter 随机增强 → 训练用
get_default_transform()  # 确定性，无随机     → 抽特征用
```

> ⚠️ 抽特征**绝对不能**用带随机增强的 transform，否则同一张图每次特征都不同，检索完全失效。

---

## 4. DataLoader：定义"怎么批量取"

```python
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
for batch in loader:
    ...
```

| 参数 | 作用 | 怎么选 |
|---|---|---|
| `batch_size` | 一批几个样本 | 16/32/64，见 A1 的权衡 |
| `shuffle` | 是否打乱 | **训练 True，推理 False** |
| `num_workers` | 几个进程并行读 | CPU 核数附近 |
| `drop_last` | 最后不满一批是否丢弃 | 用 BN 且最后一批只剩 1 个样本会报错，此时设 True |

### 4.1 collate：把单样本堆成一批

Dataset 返回**单个**样本 `[3,224,224]`，但循环里拿到 `[16,3,224,224]`。中间这步叫 **collate**，由 `default_collate` 完成：

- `Tensor` → `torch.stack`，多一个 batch 维
- `str` → **保持 `list[str]`，不转换**
- `dict` → 对每个 key 分别 collate（`TripletPlaceDataset` 返回 dict，所以能拿 `batch["anchor"]`）

> 这就是为什么代码写 `all_paths.extend(paths)` 而非 `append`——`paths` 已经是长度 16 的列表。

---

## 5. `nn.Module` ⭐ 本章重点

### 5.1 最小 Module

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()              # ← 必须调，且在最前
        self.fc = nn.Linear(10, 5)      # 定义"有哪些零件"

    def forward(self, x):               # 定义"数据怎么流过"
        return self.fc(x)
```

> **`__init__` 说"我有哪些零件"，`forward` 说"数据怎么依次流过这些零件"。**

### 5.2 为什么不能用普通 Python 类 —— 反面实验 ⭐

```python
class Bad(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(10,10), nn.Linear(10,10)]      # 普通 list

class Good(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10,10), nn.Linear(10,10)])
```

实测：

```
普通 list 装子模块  -> model.parameters() 找到 0 个张量
nn.ModuleList 装    -> model.parameters() 找到 4 个张量

Bad  .to(cuda) 后，list 里的层在 cuda 上吗: False
Good .to(cuda) 后，ModuleList 里的层在 cuda 上吗: True
```

> ⚠️ **用普通 list 装子模块，优化器根本看不到这些参数，训练时它们永远不会被更新——而且不报任何错。** 新手最容易踩、最难发现的坑之一。

### 5.3 nn.Module 免费给的四件事

| 能力 | 说明 |
|---|---|
| **① 参数自动注册** | `model.parameters()` 递归找出所有参数交给优化器 |
| **② 递归操作** | `.to('cuda')` / `.eval()` / `.state_dict()` 自动递归到所有子模块 |
| **③ 状态切换** | `train()` / `eval()` 一次性切换所有 BN 和 Dropout（M3） |
| **④ 保存加载** | `state_dict()` 导出所有参数 |

**怎么做到的？** `nn.Module` 重写了 `__setattr__`：每次 `self.xxx = 某物` 时检查它是不是 `nn.Module` / `nn.Parameter`，是就登记进内部字典。**普通 list 不是 Module，躲过了检查——这就是 Bad 失败的原因。**

对应容器：子模块用 `nn.ModuleList` / `nn.ModuleDict`，裸张量用 `nn.ParameterList`。

### 5.4 为什么写 `model(x)` 而不是 `model.forward(x)`

实测：

```
model(x)         -> hook 被触发了
model.forward(x) -> （没触发）
```

`model(x)` 走 `nn.Module.__call__`，它在调用 `forward` **前后**还做了别的事（hook、profiling）。

> **规矩：永远写 `model(x)`。** 一旦用到 hook（特征可视化、梯度分析）直接调 forward 就会出问题。

---

## 6. 怎么搭网络：三种搭法

### 6.1 `nn.Sequential` —— 纯串行时最省事

```python
self.features = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.BatchNorm2d(16),
    nn.ReLU(),
    nn.MaxPool2d(2),
)
```

自带 forward，按顺序传递。**但只对纯串行网络成立**（M2 §2.4）。

### 6.2 自定义 `forward` —— 有分支/skip 时必须用

```python
def forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.relu(out)
    out = self.conv2(out)
    out = out + identity        # ← 残差连接，Sequential 表达不了
    return self.relu(out)
```

### 6.3 `nn.ModuleList` —— 层数不固定时

```python
self.blocks = nn.ModuleList([Block() for _ in range(n)])

def forward(self, x):
    for block in self.blocks:
        x = block(x)
    return x
```

**`ModuleList` 没有自带 forward**，必须自己写循环。它唯一的作用是让参数被正确注册。

---

## 7. 完整例子：TinyCNN 的形状流动

```python
class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
```

实测形状流动：

```
输入                : (4, 3, 64, 64)
after Conv2d(3->16)   : (4, 16, 64, 64)   ← 通道 3→16，尺寸不变（padding 补偿）
after BatchNorm2d     : (4, 16, 64, 64)   ← 归一化不改形状
after ReLU            : (4, 16, 64, 64)   ← 激活不改形状
after MaxPool2d       : (4, 16, 32, 32)   ← 尺寸减半
after Conv2d(16->32)  : (4, 32, 32, 32)
after MaxPool2d       : (4, 32, 16, 16)
after AdaptiveAvgPool : (4, 32,  1,  1)
after flatten         : (4, 32)
after Linear(32->10)  : (4, 10)
```

**规律**：
- `Conv2d` 改**通道数**，空间尺寸由 stride/padding 决定
- `BatchNorm` / `ReLU` **完全不改形状**
- `MaxPool2d(2)` 空间尺寸减半
- `Linear` 只作用最后一维，之前必须 `flatten`

### 7.1 参数量（实测）

```
features.0.weight   (16, 3, 3, 3)      432     ← Conv: 16×3×3×3
features.0.bias     (16,)               16
features.1.weight   (16,)               16     ← BatchNorm 的 γ
features.1.bias     (16,)               16     ← BatchNorm 的 β
classifier.weight   (10, 32)           320     ← Linear: 输出×输入
总计                                  5514
```

- **卷积参数量 = 输出通道 × 输入通道 × k高 × k宽**（+ 输出通道个 bias）
- **BatchNorm 每层只有 2×通道数** 个参数，极轻量
- BN 的 `running_mean/var` 是 **buffer 不是 parameter**，不参与梯度更新（A2）

### 7.2 卷积输出尺寸公式（实测验证）

```
输出 = (输入 + 2×padding − kernel) ÷ stride + 1
```

| kernel | stride | padding | 输入 32 → 输出 | 用途 |
|---:|---:|---:|---:|---|
| 3 | 1 | 1 | **32**（不变） | 标准配置 |
| 3 | 2 | 1 | **16**（减半） | 下采样 |
| 7 | 2 | 3 | **16** | ResNet 第一层 |
| 1 | 1 | 0 | **32** | 1×1 卷积，只改通道 |

> 💡 记住 `k=3, s=1, p=1` 尺寸不变。规律：`padding = (kernel−1)/2` 时尺寸不变。

---

## 8. 完整训练骨架（50 行）

```python
# ② Dataset
class FakeData(Dataset):
    def __init__(self, n=256):
        self.x, self.y = torch.randn(n,3,64,64), torch.randint(0,10,(n,))
    def __len__(self):  return len(self.x)
    def __getitem__(self, i):  return self.x[i], self.y[i]

loader = DataLoader(FakeData(), batch_size=16, shuffle=True)

# ③ 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyCNN(10).to(device)                          # ← 别忘 .to(device)

# ④ 损失与优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(3):
    model.train()                                        # ← 切训练模式
    total = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)   # ← 数据也要 .to
        outputs = model(images)                          # 前向（走 __call__）
        loss = criterion(outputs, labels)

        optimizer.zero_grad()                            # 清梯度
        loss.backward()                                  # 反向
        optimizer.step()                                 # 更新

        total += loss.item() * images.size(0)            # ← 见 §9
    print(f"epoch {epoch}: loss={total/len(loader.dataset):.4f}")
```

**这就是所有 PyTorch 训练代码的骨架**，`train_triplet.py` 只是换了损失和数据形态。

---

## 9. 拆解 `total += loss.item() * images.size(0)` ⭐

这一行塞了两件毫不相干的事。

```python
total_loss += loss.item() * anchor.size(0)
#             └──┬───┘      └──────┬──────┘
#          切断计算图        还原成"总和"
```

### 9.1 为什么乘 `size(0)`

`anchor.size(0)` = 第 0 维大小 = **这一批有几个样本**。不写死 16 是因为**最后一批可能不满**（v2 训练是 60 个样本、batch=16 → `16,16,16,12`）。

**根源：`criterion` 返回的是"批内平均值"，不是"总和"。** 实测：

```
reduction='none' 返回形状: (8,)      ← 每个样本一个 loss
reduction='mean' 返回: 0.088240      ← 默认行为
手动对 none 取均值:    0.088240      ← 一致
```

**而平均值不能直接再平均。** 考试打比方：

> 一班 **50 人**平均 **80** 分，二班 **10 人**平均 **90** 分。
> 全年级平均是 (80+90)÷2 = 85 吗？**不是**，是 (50×80+10×90)÷60 = **81.67**。

实测同一现象：

```
四批：[16, 16, 16, 12] 个样本，每批平均 loss [0.50, 0.30, 0.20, 0.90]

写法A  直接平均每批均值      = 0.4750   ← 12个样本的批被高估
写法B  乘回batch size再总除  = 0.4467   ← 正确
验证：60个样本摊平求均值     = 0.4467   ← 与B一致 ✓
```

所以：**先还原成总和，最后统一除一次**，每个样本权重才相同。

> 💡 **"平均值不能直接再平均，得先还原成总和。"** 这个坑在统计准确率、F1 等各种指标时都会遇到。

### 9.2 为什么必须 `.item()`

`loss` 不是数字，是 **tensor**，通过 `grad_fn` **引用着整张计算图**（A2）。

累加 tensor = 每次迭代的图都释放不掉。实测（A2 §6）：

```
循环次数 │ total += loss (错) │ total += loss.item() (对)
    1    │    515 MB         │    515 MB
    4    │   2016 MB         │    515 MB
   16    │   8019 MB         │    515 MB
```

**线性增长直到 OOM。**

> 💡 判断标准：**这个东西还需要参与反向传播吗？** 不需要就 `.item()` 或 `.detach()`。

（副作用：`.item()` 会 GPU→CPU 强制同步。极致优化时可用 `total += loss.detach()` 在 GPU 上累加，最后再 `.item()`。本项目规模不用管。）

---

## 10. 新手最常踩的六个坑

| 坑 | 症状 | 修法 |
|---|---|---|
| 忘记 `super().__init__()` | `cannot assign module before Module.__init__() call` | 加上，放最前面 |
| **用普通 list 装子模块** | **不报错**，参数不更新 | 用 `nn.ModuleList` |
| 忘记 `.to(device)` | `Expected all tensors to be on the same device` | 模型和数据都要 |
| **忘记 `model.eval()`** | **不报错**，推理结果不稳定 | `eval()` + `no_grad()`（M3） |
| **忘记 `zero_grad()`** | **不报错**，训练不收敛 | 三部曲顺序（A1） |
| shape 不匹配 | `mat1 and mat2 shapes cannot be multiplied` | `print(x.shape)` 逐层查 |

> ⚠️ 上表**有三个是"不报错"的**——这才最危险。PyTorch 很多错误是静默的（和 M5 那个评测静默失效同类）。

**怎么读 shape 报错**：

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x2048 and 512x10)
                                                          ^^^^^^^     ^^^^^^
                                                        你给的      Linear期待的
```

你的 `Linear` 定义成了 `nn.Linear(512, 10)`，实际喂进来 2048 维。要么改 Linear 输入维度，要么检查前面 flatten 是否算错。

---

## 11. 本模块自测

1. PyTorch 的四个基础零件是什么？你真正要写的是哪两个？
2. map-style Dataset 必须实现哪两个方法？为什么是这两个？
3. 为什么 `__init__` 里只记路径不读图？
4. 为什么训练和抽特征要用不同的 transform？
5. **用普通 list 装子模块会怎样？为什么不报错？**
6. `nn.Module` 免费给了哪四件事？它靠什么机制实现参数注册？
7. 为什么写 `model(x)` 而不是 `model.forward(x)`？
8. 三种搭网络的方式各适用什么场景？
9. 卷积输出尺寸公式是什么？`k=3,s=1,p=1` 的输出尺寸？
10. **`total += loss.item() * images.size(0)` 里，两个部分各解决什么问题？**
11. 六个常见坑里，哪三个是"不报错"的？
