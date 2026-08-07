# Person B 工作日志

> 角色：模型架构 + GNN | 分支：`lxy` | 日期：2026-08-07 ~ 2026-08-08

---

## 架构总览

```
features (Person A 交付)
       │
       ▼
ConditionEncoder (B1)         Linear × 3 + BN + Dropout
       │                       将多源异构特征融合为统一条件嵌入
       ▼
ResidualDecoder (B2)          5 heads, each: Linear(256→128) → ReLU → Dropout → Linear(128→4422)
       │                       baseline + Δ_drug + Δ_strain + Δ_context → y_raw
       ▼
ProteinGraph GNN (B4)         Pearson KNN graph → 1-layer GraphConv
       │                       y_raw → 邻居平滑 → y_smoothed
       ▼
CalibrationHead (B3)          Linear(256→64) → ReLU → Linear(64→4422)
       │                       δ_cal（零初始化）
       ▼
y_pred = y_smoothed + δ_cal
```

关键数据流恒等式：
- `y_raw = baseline + delta_drug + delta_strain + delta_context`
- `y_pred = y_raw (+ GNN平滑) + delta_cal`

---

## Day 1（8月7日）

### B1: ConditionEncoder ⏱ 预计 45min

- [ ] 开始时间：\_\_:\_\_
- [ ] 完成时间：\_\_:\_\_

**文件**：[baseline/model.py](baseline/model.py)（追加）

#### 1.1 原理：为什么需要一个独立的编码器

##### 输入是什么

Person A 的 `build_condition_features()` 输出一个特征矩阵，里面包含：

- 菌株 one-hot（5 维）：`[1, 0, 0, 0, 0]`
- 化合物 hash 编码（32 维）：`[-0.31, 0.47, -0.12, ...]`
- 培养基 one-hot（2 维）：`[0, 1]`
- 温度（1 维）：`[0]` 或 `[1]`
- 时间 sin-cos（2 维）：`[0.38, 0.92]`
- 仪器 one-hot（7 维）：`[0, 0, 1, 0, 0, 0, 0]`
- 菌株统计先验 PCA（32 维）
- 化合物统计锚点 PCA（64 维）
- 交叉特征（若干维）
- ……

它们被 `np.concatenate` 拼成一个向量。但**拼接 ≠ 融合**——这些数字在向量里只是"物理上相邻"，模型自己不知道「第 0-4 维是菌株」「第 37-68 维是化合物统计先验」。如果直接把这个拼接向量喂给下游解码器，解码器需要用稀缺的训练数据自己"发现"这些结构——这在 5,920 个训练样本下效率极低。

##### 编码器的目标

编码器（`ConditionEncoder`）的任务：**把多源异构的拼接向量，映射为一个统一的、信息充分混合的条件嵌入（256 维）**。这个过程通过三层带激活的全连接层完成。

##### 关于维度：A 不一定输出恰好 256 维

`ConditionEncoder` 的 `dim_in` 参数取决于 Person A 最终拼了多少特征——可能是 220，可能是 300，不一定是 256。编码器的第一层 `Linear(dim_in, 256)` 负责做维度统一：无论输入是多少维，第一层都投影到 256 维。所以 Person A 不需要约束自己的输出维度为 256——只要告诉 B 实际的 `dim_in` 是多少就行。**256 是嵌入空间的约定维度，不是输入的强制维度。**

##### 每层的数学原理和信息处理意义

编码器由四类基础操作交替堆叠而成：**Linear（全连接层）、ReLU（激活函数）、BatchNorm（批归一化）、Dropout（随机丢弃）**。每一类有明确的数学角色。

---

**Linear（全连接层）**

数学形式：

$$y = xW^T + b$$

其中 $x \in \mathbb{R}^{1 \times d_{in}}$ 是一个样本的输入向量，$W \in \mathbb{R}^{d_{out} \times d_{in}}$ 是权重矩阵，$b \in \mathbb{R}^{1 \times d_{out}}$ 是偏置向量。

信息意义：**输出的每一个神经元都是所有输入神经元的加权和**。这意味着输出维度上的任意一个值，都"看过了"全部输入信息——菌株、化合物、时间、温度……在这一层发生了第一次混合。

编码器有三层 Linear：
- **第 1 层**（dim_in → 256）：维度统一 + 初步混合
- **第 2 层**（256 → 256）：非线性变换后的再混合（见下文 ReLU 的配合）
- **第 3 层**（256 → 256）：最终压缩，不加激活函数——输出一个"纯净"的线性嵌入，供下游模块直接使用

---

**ReLU（Rectified Linear Unit，修正线性单元）**

数学形式：

$$\text{ReLU}(x) = \max(0, x)$$

意义：

1. **引入非线性**。如果只有 Linear 层（哪怕 100 层），整个模型等价于一个单层 Linear——因为线性变换的复合仍然是线性：$W_3(W_2(W_1 x + b_1) + b_2) + b_3 = W'x + b'$。ReLU 打断了这个线性链，让模型能学到「菌株 A × 化合物 B 只在特定条件下才触发某蛋白上调」这类非线性交互模式。

2. **稀疏激活**。ReLU 将负值截断为零，意味着在任何给定样本下，约一半的神经元处于"关闭"状态。这种稀疏性有两个好处：① 计算高效；② 每个样本只激活了嵌入空间的一个子集，不同的样本激活不同的子空间——这天然形成了对样本的"软聚类"。

---

**BatchNorm1d（批归一化）**

数学形式（对每个特征维度独立计算）：

$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i \quad \text{（batch 内均值）}$$
$$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2 \quad \text{（batch 内方差）}$$
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \quad \text{（标准化）}$$
$$y_i = \gamma \hat{x}_i + \beta \quad \text{（缩放+平移，}\gamma,\beta\text{ 是可学习参数）}$$

意义：

1. **内部协变量偏移（Internal Covariate Shift）的缓解**。随着训练进行，前面层的参数更新会导致后面层接收到的输入分布不断漂移。BatchNorm 强制每个 batch 内输出均值为 0、方差为 1，让后面层始终面对稳定的输入分布——这使训练对学习率的容忍度更高，收敛更快。

2. **隐式的正则化**。每个 batch 内的 $\mu_B$ 和 $\sigma_B^2$ 是该 batch 的噪声估计，不是全局真实值。这种 mini-batch 噪声起到了类似 Dropout 的正则化效果——模型不能依赖某个神经元的精确数值，从而减少过拟合。

3. **可学习的 $\gamma$ 和 $\beta$**。标准化后的 $\gamma$ 和 $\beta$ 允许网络在需要时恢复原来的尺度——如果标准化确实损害了信息，网络可以通过学习 $\gamma$ 和 $\beta$ 来"撤销"标准化。

---

**Dropout（随机丢弃）**

数学形式（训练时）：

$$y_i = \begin{cases} \frac{x_i}{1-p} & \text{以概率 } 1-p \\ 0 & \text{以概率 } p \end{cases}$$

测试时不丢弃，直接输出 $x_i$。

意义：

1. **防止神经元的共适应（co-adaptation）**。如果没有 Dropout，网络可能形成「神经元 A 输出 +2 → 神经元 B 输出 -2 → 两者在下一层抵消 → 预测依赖这对固定搭档」。但如果在训练时每次随机扔掉一半神经元，A 和 B 不能假设对方在场——每个神经元必须独立地学到有用的特征。这相当于训练了大量不同子网络的 ensemble，但参数量不增加。

2. **$p=0.1$ 的选择**：丢弃 10% 是保守的正则化强度。对于小数据集（5,920 样本），需要正则化防过拟合；但丢弃太多（p>0.3）会导致训练信号太弱。0.1 是一个经验上安全的平衡点。

---

##### 目标（怎么判断做好了）

- 编码器输出的 256 维嵌入能支撑下游——baseline 头能从中提取基础蛋白表达水平，Δ_drug 头能从中提取药物效应信号
- BatchNorm 保证每层输出的均值和方差稳定，训练时 loss 不震荡
- Dropout 防过拟合：train loss 和 val loss 之间的 gap 不过大
- 维度标准化：不管 Person A 特征维度是多少，输出恒为 256

```python
class ConditionEncoder(nn.Module):
    """
    条件编码器：输入特征 → 3层MLP → 条件嵌入 (N, 256)

    架构:
        Linear(dim_in, 256) → ReLU → BatchNorm → Dropout(0.1)
      → Linear(256, 256)    → ReLU → BatchNorm → Dropout(0.1)
      → Linear(256, 256)     （最后不加激活，输出纯线性嵌入）
    """
    def __init__(self, dim_in, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x):
        return self.net(x)
```

**自检**：
```python
encoder = ConditionEncoder(dim_in=256)  # dim_in 可能与 Person A 交付后调整
x = torch.randn(32, 256)
out = encoder(x)
assert out.shape == (32, 256), f"Expected (32,256), got {out.shape}"
assert not torch.isnan(out).any(), "NaN in encoder output"
```

**结果记录**：
- dim_in（最终值）：75（当前 Phase 1 简易特征；Person A 交付后预计 ~220-300）
- 参数量：198,400
- 自检通过：是
- 问题/备注：第一层 Linear(dim_in, 256) 自适应输入维度，Person A 交付后无需修改

---

### B2: ResidualDecoder ⏱ 预计 2.5h

- [ ] 开始时间：\_\_:\_\_
- [ ] 完成时间：\_\_:\_\_

**文件**：[baseline/decoder.py](baseline/decoder.py)（新建）

#### 2.1 原理：为什么要把预测拆成四项

##### 传统端到端的问题

传统做法（对应你的 `SimpleDecoder`）：

$$y_{pred} = \text{MLP}(\text{condition\_embedding})$$

即一个 MLP 直接从条件嵌入映射到 4,422 维蛋白表达。模型内部是一个黑盒——你不知道预测值里哪部分是「这个菌株本来就高表达」，哪部分是「药物让这个蛋白上调了」。评测体系的核心指标是 **fold change**：

$$\text{FC} = y_{treat} - y_{control}$$

端到端 MLP 的预测值 $y_{pred}$ 里没有显式的 FC——FC 信息隐含在绝对值里，和 baseline 混在一起。MSE loss 优化的是绝对值的平均精度，对「FC 的正负号是否正确」不敏感。

##### 残差分解的结构化假设

任何 treatment 样本的蛋白表达可以自然地分解为：

$$y_{treatment} = y_{baseline} + \Delta_{drug} + \Delta_{strain\_mod} + \Delta_{context}$$

对应五个 head：

| Head | 数学符号 | 生物学含义 | 谁主要影响它 |
|------|:------:|-----------|------------|
| `baseline_head` | $\hat{y}_0$ | 细胞基础蛋白表达水平（由菌株、培养基、温度决定） | 菌株信息、培养基、温度 |
| `delta_drug_head` | $\hat{\Delta}_{drug}$ | 化合物带来的蛋白响应变化 | 化合物信息 |
| `delta_strain_head` | $\hat{\Delta}_{strain}$ | 菌株遗传背景对扰动效果的调制 | 菌株×化合物的交互 |
| `delta_context_head` | $\hat{\Delta}_{context}$ | 温度×时间×培养基的精细调节 | 时间、温度、培养基 |
| `calibration_head`（B3） | $\hat{\delta}_{cal}$ | 仪器/板级系统偏差 | 仪器、细胞板 |

如果模型显式输出这四个分量，$\Delta_{drug}$ **天然就是 fold change 的模型表示**——它直接对接到评测中的 FC 指标。

##### OOD 场景下的优雅降级

对于 unseen 化合物：
- `delta_drug_head` 没有见过该化合物的训练信号 → **L2 正则约束它输出接近零**（详见 2.3）
- 预测退化为 `baseline + delta_strain + delta_context`
- 这是一个**合理的 fallback**：模型在说「我猜这个新化合物没有特殊的蛋白响应效应」，它的预测等于「这个菌株在这种条件下的基础蛋白表达 + 菌株背景的已知调制 + 环境的已知影响」

对于 unseen 菌株：同理，`delta_strain_head` 输出接近零，baseline 头仍给出合理的基线。

这就是设计文档里说的 **「部分失效而非全局崩塌」**——端到端 MLP 在 unseen 实体上，所有 4,422 个蛋白的预测同时受影响；残差分解只让相关的那一个 head 失效。

#### 2.2 原理：五个 head 共享输入，如何实现差异化

##### 五个 head 接收完全相同的输入

```python
def forward(self, embedding):
    baseline  = self.baseline_head(embedding)    # ← 同一个 embedding
    d_drug    = self.delta_drug_head(embedding)   # ← 同一个 embedding
    d_strain  = self.delta_strain_head(embedding) # ← 同一个 embedding
    d_context = self.delta_context_head(embedding)# ← 同一个 embedding
```

五个 head 收到的数字完全一样。如果它们共享参数，五个输出也会完全一样——那残差分解就退化为 `y_raw = 5 × baseline`。

##### 差异化来自独立参数

每个 head 有自己独立的权重矩阵。以 `baseline_head` 和 `delta_drug_head` 为例：

```
baseline_head:
  W₁_b (128×256), b₁_b (128,)
  W₂_b (4422×128), b₂_b (4422,)

delta_drug_head:
  W₁_d (128×256), b₁_d (128,)    ← 不同于 W₁_b
  W₂_d (4422×128), b₂_d (4422,)  ← 不同于 W₂_b
```

相同的输入 `embedding`，经过不同的权重矩阵，产生不同的输出。数学上，这两个 head 定义了从 $\mathbb{R}^{256}$ 到 $\mathbb{R}^{4422}$ 的两个不同函数 $f_b$ 和 $f_d$：

$$f_b(emb) = W_{2b} \cdot \text{ReLU}(W_{1b} \cdot emb + b_{1b}) + b_{2b}$$
$$f_d(emb) = W_{2d} \cdot \text{ReLU}(W_{1d} \cdot emb + b_{1d}) + b_{2d}$$

$f_b \neq f_d$ 因为参数不同——即使输入相同。

##### 梯度反传驱动的自然分工

训练初期（随机初始化），五个 head 的输出都是随机的，彼此之间有差异但无结构。随着训练进行：

1. `baseline_head` 的输出直接贡献 $y_{raw}$，而 $y_{raw} \approx y_{pred}$ 被 MSE loss 约束接近真实值
2. 真实值里包含了所有因素的总和——baseline + drug + strain + context
3. **关键机制**：FC Pearson loss（Person C 的 C2）**只对 `delta_drug_head` 的输出施加额外约束**——它要求 `delta_drug` 与真实的 fold change 相关
4. 这个"专属信号"将药物效应的责任**定向推给** `delta_drug_head`
5. 残差 L2 loss（Person C 的 C3）惩罚所有 head 的大输出，促使每个 head 只输出**必要的**信号——没有药物效应信号时，`delta_drug` 被 L2 压向零

梯度分工的物理直觉：FC loss 的梯度只流向 `delta_drug_head` 的参数，不流向 `baseline_head`。这就像三个人写一份报告——老板（FC loss）只对其中一个人的章节给修改意见，其他人接收到的信号（MSE loss）是「让总和正确」——自然的结果是：被老板盯的人负责特定的内容，其他人负责剩下的。

#### 2.3 原理：为什么 unseen 化合物时 Δ_drug 输出接近零

这是整个方案最关键的理论保证。需要 L2 正则化来驱动。

##### L2 正则化的数学行为

L2 正则项为：

$$\mathcal{L}_{L2}^{drug} = \lambda \cdot ||\hat{\Delta}_{drug}||_2^2 = \lambda \sum_{i=1}^{N} \sum_{p=1}^{P} (\hat{\Delta}_{drug})_{i,p}^2$$

梯度为：

$$\frac{\partial \mathcal{L}_{L2}^{drug}}{\partial (\hat{\Delta}_{drug})_{i,p}} = 2\lambda \cdot (\hat{\Delta}_{drug})_{i,p}$$

这个梯度**永远指向零**——不管 $(\hat{\Delta}_{drug})_{i,p}$ 是正还是负，梯度都把它往零推。

##### 两股力量的平衡

对于训练集中出现的化合物（seen compounds）：

- **FC Pearson loss 的梯度**：把 $\hat{\Delta}_{drug}$ 推向与真实 FC 一致的方向
- **L2 正则的梯度**：把 $\hat{\Delta}_{drug}$ 推向零

两股力量形成一个平衡点：$\hat{\Delta}_{drug}$ 刚好大到能捕捉真实 FC 信号，但又不会过大（被 L2 约束）。

对于 unseen 化合物：

- **FC Pearson loss 没有梯度**——unseen 化合物不出现在训练集中，FC loss 对它不产生任何信号
- **L2 正则的梯度仍然存在**——它在所有样本上无条件生效
- 因此 L2 正则**单方面胜利**：$\hat{\Delta}_{drug} \rightarrow 0$

这就是 unseen 化合物时 Δ_drug 输出接近零的数学保证。

##### 对 seen 化合物，L2 会不会把 Δ 也压成零？

不会——因为 FC loss 和 MSE loss 给 seen 样本的 Δ_drug 提供了**非零的、有意义的方向**。在平衡点：

$$\frac{\partial \mathcal{L}_{total}}{\partial \hat{\Delta}_{drug}} = \frac{\partial \mathcal{L}_{MSE}}{\partial \hat{\Delta}_{drug}} + \frac{\partial \mathcal{L}_{FC}}{\partial \hat{\Delta}_{drug}} + 2\lambda \hat{\Delta}_{drug} = 0$$

前两项为非零值提供"支撑"，$\hat{\Delta}_{drug}$ 不会坍缩到零。L2 的角色是**压扁不必要的维度**——那些对 loss 无贡献的方向会被压缩到零。

##### 类比：奥卡姆剃刀的数学实现

L2 正则是一个"懒惰惩罚"——除非有明确的证据（梯度）要求你非零，否则最优解是零。Seen 化合物有证据（FC loss 的梯度），unseen 没有 → 对 seen 非零、对 unseen 为零。模型在说：「有信号的化合物我给出具体的 FC，没见过的化合物我保守地说零——也就是'没有特殊效应'。」

#### 2.4 实现

```python
import torch
import torch.nn as nn


class ResidualDecoder(nn.Module):
    """
    残差分解解码器：条件嵌入 → 5头独立解码 → 残差求和

    每个 head 瓶颈结构:
        Linear(256→128) → ReLU → Dropout(0.1) → Linear(128→4422)

    输出 dict:
        baseline:      共享基线响应
        delta_drug:    化合物特异残差
        delta_strain:  菌株调制残差
        delta_context: 温度×时间×培养基上下文效应
        y_raw:         四项求和 (baseline + Σ deltas)
    """

    def __init__(self, dim_emb=256, n_proteins=4422, hidden=128, dropout=0.1):
        super().__init__()
        self.dim_emb = dim_emb
        self.n_proteins = n_proteins

        def make_head():
            return nn.Sequential(
                nn.Linear(dim_emb, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_proteins),
            )

        self.baseline_head = make_head()
        self.delta_drug_head = make_head()
        self.delta_strain_head = make_head()
        self.delta_context_head = make_head()

    def forward(self, embedding):
        baseline = self.baseline_head(embedding)
        d_drug = self.delta_drug_head(embedding)
        d_strain = self.delta_strain_head(embedding)
        d_context = self.delta_context_head(embedding)

        y_raw = baseline + d_drug + d_strain + d_context

        return {
            "baseline": baseline,
            "delta_drug": d_drug,
            "delta_strain": d_strain,
            "delta_context": d_context,
            "y_raw": y_raw,
        }


class SimpleDecoder(nn.Module):
    """
    端到端解码器（消融对照用）—— 单头直接映射

    参数量通过加宽 hidden 与 ResidualDecoder 拉齐（~3M），
    确保消融对比中「残差分解的结构偏置」而非「参数更多」是差异来源。
    """

    def __init__(self, dim_emb=256, n_proteins=4422, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_emb, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_proteins),
        )

    def forward(self, embedding):
        y = self.net(embedding)
        return {
            "baseline": y,
            "delta_drug": torch.zeros_like(y),
            "delta_strain": torch.zeros_like(y),
            "delta_context": torch.zeros_like(y),
            "y_raw": y,
        }
```

**自检**：
```python
# 测试 ResidualDecoder
decoder = ResidualDecoder(dim_emb=256, n_proteins=4422)
emb = torch.randn(32, 256)
out = decoder(emb)

# 1. shape
for key in ["baseline", "delta_drug", "delta_strain", "delta_context", "y_raw"]:
    assert out[key].shape == (32, 4422), f"Bad shape: {key} = {out[key].shape}"

# 2. 恒等式（核心约束——这是残差分解的数学定义）
assert torch.allclose(
    out["y_raw"],
    out["baseline"] + out["delta_drug"] + out["delta_strain"] + out["delta_context"],
    atol=1e-5
), "❌ 残差恒等式不成立!"

# 3. 参数量
n = sum(p.numel() for p in decoder.parameters())
print(f"ResidualDecoder params: {n:,} (预期 ~3.0M)")

# 4. 测试 SimpleDecoder（确认接口兼容 + 参数量可比）
simp = SimpleDecoder(dim_emb=256, n_proteins=4422)
out2 = simp(emb)
n2 = sum(p.numel() for p in simp.parameters())
print(f"SimpleDecoder params: {n2:,}")
assert "y_raw" in out2
```

**结果记录**：
- ResidualDecoder 参数量：2,413,336
- SimpleDecoder 参数量：2,662,726（差距 9.4%，< 20%，消融对比公平）
- 恒等式自检通过：是
- 问题/备注：4 head（非 5 head），瓶颈结构 256→128→4422 每头约 0.6M 参数

---

### B3: CalibrationHead ⏱ 预计 1h

- [ ] 开始时间：\_\_:\_\_
- [ ] 完成时间：\_\_:\_\_

**文件**：[baseline/model.py](baseline/model.py)（追加）

#### 3.1 原理：为什么要独立建模批次效应

##### 问题：仪器和板的混淆效应

质谱数据里，同一份生物样本在不同仪器、不同细胞培养板上跑，测出的蛋白强度有系统性偏移。比如仪器 A 对所有蛋白的读数都偏高 0.3 log2——这是测量偏差，不是生物学信号。

如果仪器/板信息和其他条件（菌株、化合物）混在一起编码，模型可能学到这样的假规律：「仪器 A 上蛋白 X 总是偏高 → 菌株 B（恰好都在仪器 A 上跑）的蛋白 X 偏高」。这就是**混淆**——模型把仪器偏差当成了生物学特征。

##### 独立校准分支的设计

校准头在架构上与残差分解解码器**并行**——它不干预 y_raw 的计算，只输出一个加性偏移：

$$y_{pred} = y_{raw} + \delta_{cal}$$

##### 加性 vs 乘性

选择加性（$+ \delta_{cal}$）而非乘性（$\times \gamma_{cal}$）的理由：质谱仪器偏差在 log2 空间主要表现为**平移**（基线漂移），而非缩放。log2 变换已经将乘法效应（raw intensity 尺度上的倍数差异）转化为加法效应（log2 尺度上的平移）。

##### 零初始化的意义

```python
nn.init.zeros_(self.net[-1].weight)
nn.init.zeros_(self.net[-1].bias)
```

训练第 0 步：$\delta_{cal} = 0$，`y_pred = y_raw`——模型行为和没有校准完全一样。

随着训练进行，如果仪器偏差确实存在，MSE loss 的梯度会逐渐让校准头学到非零的偏移量。**只有那些「不加校准就学不好」的偏差才会被校准头吸收**——梯度不会为了"偷懒"而经过校准头：如果不用校准也能学好，校准头的梯度很微弱，零初始化附近的 L2 隐式正则会让它保持接近零。

##### 为什么不在数据预处理层面做

传统做法是预处理时用 ComBat 或 limma 去批次效应。但这类方法：
1. 假设线性效应 + 经验贝叶斯先验，不适用于非线性系统
2. 与模型训练解耦——"先去批次再训练"会丢失信息（批次效应里可能混有微弱生物信号）
3. 无法利用预测误差的反馈来调整校准

端到端训练的校准头：校准量 = 模型自己学到的、对降低训练 loss 有帮助的、与仪器/板相关的加性偏移。它与预测目标联合优化。

```python
class CalibrationHead(nn.Module):
    """
    批次校准分支：条件嵌入 → 轻量 MLP → 加性蛋白偏移

    输入: 条件嵌入 (N, dim_emb) —— 包含仪器/板信息
    输出: 加性偏移 (N, N_PROTEINS)

    零初始化确保训练起始时 y_pred = y_raw，校准量在训练中逐渐浮现。
    """

    def __init__(self, dim_emb=256, n_proteins=4422, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_emb, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_proteins),
        )
        # 零初始化 → 初始 δ_cal = 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, embedding):
        return self.net(embedding)
```

**自检**：
```python
cal = CalibrationHead(dim_emb=256, n_proteins=4422)
emb = torch.randn(32, 256)
offset = cal(emb)

# 1. shape
assert offset.shape == (32, 4422)

# 2. zero init 生效
assert offset.abs().max() < 1e-6, f"Zero init 未生效: max={offset.abs().max():.6f}"

# 3. 梯度流测试（确认参数可以接收梯度）
loss = offset.sum()
loss.backward()
assert cal.net[-1].weight.grad is not None, "校准头参数无梯度!"
print("B3 ✅")
```

**结果记录**：
- 参数量：303,878
- zero init 自检通过：是（初始 max=0.00e+00）
- 梯度流正常：是（loss.backward() 后 grad 存在）
- 问题/备注：输入为 256-dim 条件嵌入（含仪器/板信息），输出 4422-dim 加性偏移。比原方案更精简——不需要单独的仪器 one-hot + 板 hash 输入

---

### 阶段性集成：AIVCModel ⏱ 预计 45min

- [ ] 开始时间：\_\_:\_\_
- [ ] 完成时间：\_\_:\_\_

**文件**：[baseline/model.py](baseline/model.py)（追加）

#### 4.1 原理：forward() 和 backward() 是什么

##### forward()：定义数据流

`forward(x)` 是 PyTorch 模型的核心方法，它定义了**从输入到输出的计算图**：

```python
def forward(self, x):
    emb = self.encoder(x)          # 步骤1: 编码
    dec_out = self.decoder(emb)    # 步骤2: 残差分解
    delta_cal = self.calibration(emb) # 步骤3: 校准
    y_pred = dec_out["y_raw"] + delta_cal  # 步骤4: 求和
    return { "y_pred": y_pred, ... }
```

每一步调用子模块时，PyTorch 在后台自动构建一个**有向无环计算图（DAG）**，记录每个操作（矩阵乘法、加法、ReLU……）以及它们之间的依赖关系。这个图是后续反向传播的基础。

你可以把 `forward()` 理解为"正向信息流"——输入数据从左到右流过所有层，最终产生预测值。它回答：「给定这些条件，模型预测蛋白表达是多少？」

##### backward()：计算梯度

`loss.backward()` 紧跟在 `forward` 之后：

```python
out = model(x)                    # forward: 构建计算图
loss = loss_fn(out["y_pred"], y)  # 计算损失
loss.backward()                   # backward: 自动求梯度
```

PyTorch 从 `loss` 出发，沿着 `forward` 时记录的计算图**反向遍历**，对每个参数 $p$ 计算偏导数 $\frac{\partial \mathcal{L}}{\partial p}$，并将结果存入 `p.grad`。

```
forward:   x → encoder → emb → decoder → y_raw → + → y_pred → loss
                                                    ↑
                                             delta_cal

backward:  loss → ∂L/∂y_pred → ∂L/∂y_raw → ∂L/∂emb → ∂L/∂encoder_params
                            ↘ ∂L/∂delta_cal → ∂L/∂cal_params
```

链式法则递归展开，从 loss 一路回到最底层的 encoder 参数。每一个 `nn.Linear` 的 weight 和 bias 都会在 `.grad` 属性中得到梯度。

然后优化器（Adam）读取 `.grad`，用梯度下降更新参数：

$$W_{new} = W_{old} - \eta \cdot \frac{\partial \mathcal{L}}{\partial W}$$

##### 集成阶段为什么要测梯度流

三个组件（encoder、decoder、calibration）各自独立开发时用随机 tensor 自检。但合在一起后，计算图变得复杂——可能出现某个子模块**被断在计算图外面**（grad 为 None）：

- `.item()` 或 `.detach()` 打断了计算图链路
- 某个中间 tensor 被覆盖（in-place 操作）
- 某个子模块的输出没有被后续用到（计算图上不存在依赖路径）

如果一个参数的 `grad` 是 None，它永远不会被更新——等于这个子模块没参与训练。集成后的梯度流测试就是逐个检查所有参数的 `.grad` 是否存在。

#### 4.2 集成后的接口契约

`forward` 返回的 dict 是 B 和 C 之间的**合同**。七个 key 缺一不可——C 的 loss 函数按 key 名取 tensor，keyError 意味着联调失败。

| key | shape | 消费者 | 用途 |
|-----|-------|--------|------|
| `y_pred` | (N, P) | C4 训练循环 | 主 loss（mask-aware MSE） |
| `y_raw` | (N, P) | B 自用 / C 可选 | GNN 输入 / 诊断 |
| `baseline` | (N, P) | C3（可选） | 残差 L2 正则 |
| `delta_drug` | (N, P) | **C2** | FC Pearson loss（核心！） |
| `delta_strain` | (N, P) | C3 | 残差 L2 正则 |
| `delta_context` | (N, P) | C3 | 残差 L2 正则 |
| `delta_cal` | (N, P) | C 分析 | 批次效应可解释性诊断 |

#### 4.3 实现

```python
class AIVCModel(nn.Module):
    """
    AIVC 完整模型

    数据流:
        x (N, dim_in)
        → ConditionEncoder → emb (N, 256)
        → ResidualDecoder   → {baseline, deltas, y_raw}
        → [+ GNN 平滑]      → y_smoothed
        → CalibrationHead   → δ_cal
        → y_pred = y_smoothed + δ_cal

    参数:
        dim_in:       输入特征维度（来自 Person A 的 build_condition_features）
        n_proteins:   输出蛋白数
        dim_emb:      条件嵌入维度（默认 256）
        use_gnn:      是否启用 GNN（B4 完成后可用）
    """

    def __init__(self, dim_in, n_proteins, dim_emb=256, use_gnn=False):
        super().__init__()
        from baseline.decoder import ResidualDecoder

        self.dim_emb = dim_emb
        self.n_proteins = n_proteins
        self.use_gnn = use_gnn

        self.encoder = ConditionEncoder(dim_in, hidden=dim_emb)
        self.decoder = ResidualDecoder(dim_emb, n_proteins)
        self.calibration = CalibrationHead(dim_emb, n_proteins)
        self.gnn = None  # B4 完成后替换为 GraphConv 实例

    def forward(self, x):
        emb = self.encoder(x)
        dec_out = self.decoder(emb)
        delta_cal = self.calibration(emb)

        y_raw = dec_out["y_raw"]

        if self.gnn is not None:
            # GNN 作用于蛋白维度:
            # (N, P) → 转置 → (P, N) → GraphConv(邻居聚合) → (P, N) → 转回 → (N, P)
            y_t = y_raw.t()
            y_t = self.gnn(y_t, self.edge_index)
            y_raw = y_t.t()

        y_pred = y_raw + delta_cal

        return {
            "y_pred": y_pred,
            "y_raw": y_raw,
            "baseline": dec_out["baseline"],
            "delta_drug": dec_out["delta_drug"],
            "delta_strain": dec_out["delta_strain"],
            "delta_context": dec_out["delta_context"],
            "delta_cal": delta_cal,
        }
```

**自检**：
```python
model = AIVCModel(dim_in=256, n_proteins=4422)
x = torch.randn(32, 256)
out = model(x)

# 1. 全部 7 个 key 存在且 shape 正确
expected_keys = ["y_pred", "y_raw", "baseline", "delta_drug",
                 "delta_strain", "delta_context", "delta_cal"]
for key in expected_keys:
    assert key in out, f"Missing key: {key}"
    assert out[key].shape == (32, 4422), f"Bad shape: {key} = {out[key].shape}"

# 2. 恒等式验证（这是 forward() 计算图的数学约束）
# y_raw = baseline + Σ deltas
assert torch.allclose(
    out["y_raw"],
    out["baseline"] + out["delta_drug"] + out["delta_strain"] + out["delta_context"],
    atol=1e-5
), "❌ 残差恒等式不成立!"

# y_pred = y_raw + delta_cal
assert torch.allclose(
    out["y_pred"],
    out["y_raw"] + out["delta_cal"],
    atol=1e-5
), "❌ 预测恒等式不成立!"

# 3. 总参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {total_params:,} (target < 5M)")
assert total_params < 5_000_000, f"参数量超标: {total_params:,} > 5M"

# 4. 梯度流 — 验证 backward() 能抵达所有子模块
loss = out["y_pred"].sum()
loss.backward()
no_grad_params = []
for name, p in model.named_parameters():
    if p.grad is None:
        no_grad_params.append(name)
if no_grad_params:
    print(f"⚠️  以下参数无梯度（被断在计算图外）:")
    for n in no_grad_params:
        print(f"    {n}")
else:
    print("梯度流检查通过: 所有参数都有 grad")

print("AIVCModel B1+B2+B3 ✅")
```

**结果记录**：
- 总参数量：2,915,614（target < 5M，✅ 达标）
- 恒等式通过：是
- 全部 key 存在：是
- 梯度流正常（无 None grad）：是
- 问题/备注：dim_in=256 测试；真实 dim_in=75 时参数量 2,869,278。forward 返回 7-key dict 符合契约 2

---

### Day 1 傍晚联调 — 与 Person A + Person C ⏱ 30min

- [x] A 的 encoder 特征维度（dim_in）：75（当前 Phase 1 简易特征）
- [x] `model(A_features)` 前向通过：是（dim_in=75, P=4422, 4 样本 batch 前向通过）
- [ ] C 的 loss 消费 `pred_dict` 无 KeyError：**待 Person C 确认**
- [ ] 联调问题记录：**待三人联调**

---

## Day 2（8月8日）

### B4: 蛋白共表达图 + GNN ⏱ 预计 3h

- [ ] 开始时间：\_\_:\_\_
- [ ] 完成时间：\_\_:\_\_

**文件**：[baseline/protein_graph.py](baseline/protein_graph.py)（新建）

#### 4.1 从零理解：图（Graph）是什么

一个图 $G = (V, E)$ 由两部分组成：
- **节点（vertices）** $V$：$P$ 个实体。在我们的场景中，每个节点是一个蛋白，$P=4422$。
- **边（edges）** $E$：节点之间的连接关系。如果蛋白 $i$ 和蛋白 $j$ 在训练数据中高度共表达（Pearson $r > 0.7$），它们之间有一条边。

图的信息用**邻接矩阵** $A \in \mathbb{R}^{P \times P}$ 表示：
- $A_{ij} = 1$：节点 $i$ 和 $j$ 之间有一条边
- $A_{ij} = 0$：无边
- 对于无向图（我们的选择），$A$ 是对称的：$A_{ij} = A_{ji}$

PyG（PyTorch Geometric）不使用 $P \times P$ 的稠密矩阵存储（那将占用 $4422^2 \times 4 \text{ bytes} \approx 78\text{MB}$ 且 99.9% 是零），而是用 COO 格式的 `edge_index`——一个 $(2, E)$ 的 LongTensor，两行分别存储每条边的源节点和目标节点索引。例如如果蛋白 3 和蛋白 17 之间有一条边，`edge_index` 中就有两列 `[3, 17]` 和 `[17, 3]`。

#### 4.2 原理：为什么要在蛋白预测上叠加图神经网络

##### 生物学根源

残差分解解码器输出 4,422 个蛋白的预测值，但解码器内部**每个蛋白是独立预测的**——`Linear(128, 4422)` 的 4,422 个输出神经元之间只有线性加权关系，权重 $W_{out} \in \mathbb{R}^{4422 \times 128}$ 的第 $i$ 行决定蛋白 $i$ 的预测值，第 $j$ 行决定蛋白 $j$ 的预测值——两者之间没有显式的交互约束。

这在生物学上不合理。真实细胞中：

- **核糖体蛋白**（约 80 个，如 RPS3、RPS5、RPL10 等）总是协同表达——它们共用一套转录因子和翻译调控机制（Rap1、Fhl1 等），mRNA 水平同步变化，蛋白水平随之同步。
- **蛋白酶体亚基**（约 33 个）在应激条件下协同上调以加速受损蛋白的降解。
- **糖酵解通路**的 10 个酶在葡萄糖存在时协同上调，在甘油培养基中协同下调。

如果在训练数据中蛋白 A 和蛋白 B 的 Pearson $r = 0.95$，但模型对 A 预测了上调、对 B 预测了下调——这种不一致在 MSE loss 中不会被充分惩罚（MSE 只看每个蛋白的独立误差，不看蛋白间的相对一致性），但它直接违反了蛋白质组学的第一条规律：**功能相关的蛋白协同表达**。

GNN 层的目标：**让「邻居」蛋白的预测值互相约束和纠偏**。如果 RPS3 和 RPS5 在图上是邻居（Pearson $r=0.92$），GNN 在聚合邻居信息后，RPS3 和 RPS5 的预测会向彼此靠拢，减少不一致。

##### 这个做法的信息论基础

GNN 在这里扮演的角色本质上是**结构正则化**——在 loss 函数（MSE + FC Pearson + L2）之外，通过架构设计施加一个软约束：「共表达的蛋白应该有相似的预测模式」。这和 L2 正则化在参数空间的作用类似——都是限制了模型的假设空间，但 GNN 提供的是**数据驱动的、非均匀的**约束（蛋白 A 应该向邻居靠拢多少，取决于它和邻居的相关强度），而非 L2 那种「所有参数均匀向零」的盲正则。

##### 为什么必须是数据驱动的图（不能用外部知识库）

封闭数据榜禁止使用 GO、KEGG、STRING PPI 等外部蛋白功能网络。我们的图从训练集蛋白表达数据本身构建——Pearson 相关矩阵完全在封闭榜约束内。这带来了额外的好处：图的边基于**在当前实验条件下的实测共表达关系**，而非从文献中挖掘的通用功能关联——后者可能在酵母的特定实验体系中不成立。

#### 4.3 第一步：Pearson 相关系数——从协方差到标准化的关联强度

##### 协方差的几何直觉

两个蛋白 $y_i$ 和 $y_j$（各自是 $N$ 个训练样本上的 log2 表达值向量）的**协方差**：

$$\text{Cov}(y_i, y_j) = \frac{1}{N}\sum_{n=1}^{N} (y_{ni} - \bar{y}_i)(y_{nj} - \bar{y}_j)$$

这个值告诉你 $y_i$ 和 $y_j$ 是否同方向变化：
- 如果大多数样本上 $(y_{ni} - \bar{y}_i)$ 和 $(y_{nj} - \bar{y}_j)$ 同号（同为正或同为负），Cov > 0——蛋白协同表达。
- 如果多数异号（一个高于均值时另一个低于均值），Cov < 0——蛋白拮抗。

但协方差有一个致命问题：**它对尺度敏感**。如果你把 $y_i$ 的单位从 log2 换成自然对数 ln，协方差会缩小 $(1/\ln 2)^2 \approx 2.08$ 倍。协方差是 $10$ 还是 $100$，你无法判断关联强度——它可能来自真实的强关联，也可能只是这个蛋白的表达量波动范围大。

##### Pearson 相关系数：去量纲化

$$r_{ij} = \frac{\text{Cov}(y_i, y_j)}{\sigma_i \cdot \sigma_j} = \frac{\sum_{n=1}^{N} (y_{ni} - \bar{y}_i)(y_{nj} - \bar{y}_j)}{\sqrt{\sum_{n=1}^{N} (y_{ni} - \bar{y}_i)^2} \cdot \sqrt{\sum_{n=1}^{N} (y_{nj} - \bar{y}_j)^2}}$$

分子是协方差，分母是两者标准差的乘积。这等价于：
1. 先将 $y_i$ 和 $y_j$ 分别**标准化**为 z-score：$z_i = (y_i - \bar{y}_i)/\sigma_i$，$z_j = (y_j - \bar{y}_j)/\sigma_j$
2. 再计算两者的**内积除以 N**：$r_{ij} = \frac{1}{N} z_i \cdot z_j$

标准化的意义：$z_i$ 和 $z_j$ 现在都在「偏离自身均值多少个标准差」的尺度上。$r=1$ 意味着在每个样本上，蛋白 $i$ 偏离自身均值的程度恰好等于蛋白 $j$ 偏离自身均值的程度——它们完全同向变化。$r=0$ 意味着两者的波动完全无关。

##### 具体数值示例

假设训练集中蛋白 RPS3 和 RPS5 在 5 个样本上的 log2 表达值为：

| 样本 | RPS3 ($y_i$) | RPS5 ($y_j$) | $y_i - \bar{y}_i$ | $y_j - \bar{y}_j$ | 乘积 |
|------|:---:|:---:|:---:|:---:|:---:|
| 1 | 18.2 | 17.8 | -1.0 | -1.2 | +1.20 |
| 2 | 19.5 | 19.3 | +0.3 | +0.3 | +0.09 |
| 3 | 20.1 | 20.0 | +0.9 | +1.0 | +0.90 |
| 4 | 18.0 | 17.5 | -1.2 | -1.5 | +1.80 |
| 5 | 20.2 | 20.4 | +1.0 | +1.4 | +1.40 |

$\bar{y}_i = 19.2$，$\bar{y}_j = 19.0$
$\sum$ 乘积 = $5.39$，$\sum (y_i-\bar{y}_i)^2 = 4.34$，$\sum (y_j-\bar{y}_j)^2 = 7.34$
$r = \frac{5.39}{\sqrt{4.34 \times 7.34}} = \frac{5.39}{5.64} = 0.956$

接近 1 的 r 表明两者高度正相关。注意：RPS3 的平均值是 19.2，RPS5 是 19.0——绝对量不同，但**变化模式**几乎一致。Pearson 相关捕捉的是「变化模式的相似性」，不是「绝对值的相似性」——这正是我们需要的。

##### $r_{ij}$ 取值范围的生物学解读

- $r_{ij} \approx +1$：蛋白 $i$ 和 $j$ 在**所有实验条件下**同进同退——强烈暗示它们属于同一功能模块（如都是核糖体蛋白）或受同一调控通路控制
- $r_{ij} \approx +0.5$：中等正相关——可能部分条件共享，部分条件独立
- $r_{ij} \approx 0$：表达变化模式无关——功能独立
- $r_{ij} \approx -0.5$：中等负相关——代谢权衡（如呼吸链蛋白 vs 发酵酶在葡萄糖浓度变化时的此消彼长）
- $r_{ij} \approx -1$：完全负相关——在酵母中罕见（大多数蛋白不存在严格的 zero-sum 关系）

#### 4.4 第二步：从相关矩阵到 KNN 稀疏图

##### 问题：全连接图不可行

如果 $r_{ij}$ 矩阵是完整的（$4422 \times 4422$），每条可能的边都保留，那么：
- 边数：$P(P-1)/2 \approx 9.8 \times 10^6$（约一千万条边）
- GNN 的每次前向传播：每条边都参与消息传递，计算量约 $O(E \cdot d_{feature}) = O(10^7 \cdot N_{batch})$——对 256 的 batch size，每步约 $2.5 \times 10^9$ 次乘加运算
- 更关键的是：绝大多数 $r_{ij}$ 接近 0，对应的边**根本不携带生物学信息**——它们只是随机波动在有限样本下的非零估计值

##### 5,920 个样本能可靠估计多大的 r？

相关系数的标准误差（在大样本下）约为：

$$SE(r) \approx \frac{1 - r^2}{\sqrt{N - 2}}$$

当 $N = 5920$：$SE \approx \frac{1 - r^2}{76.9}$。

- 对于 $r = 0.7$：$SE \approx 0.0066$——信号远大于噪声
- 对于 $r = 0.1$：$SE \approx 0.013$——噪声与信号同量级
- 对于 $r = 0.01$：$SE \approx 0.013$——纯噪声

这就是 threshold = 0.7 的理论依据：$|r| < 0.7$ 时，真实相关可能非常弱甚至为零，而估计值本身的不确定性使得这些弱边不可靠。0.7 对应 $r^2 = 0.49$——约一半的方差可以被线性关系解释，这已经是一个相当保守的阈值。

##### KNN 的作用

KNN 进一步限制每个蛋白的邻居数至多 $K=5$ 个。为什么是 5 而不是 10 或 20？

- **生物学理由**：一个蛋白通常只与少数几个功能模块紧密相关。把 K 设得太大，会将无关蛋白也牵入邻居集合，稀释了真正的共表达信号。
- **计算理由**：$E \approx K \cdot P \approx 5 \times 4422 = 22,110$ 条边，GNN 前向传播仅需几毫秒。
- **经验理由**：GNN 文献中 K=5 是一个常用选择，在过平滑（K 太大，所有节点汇聚到同一表示）和欠信息（K 太小，邻居信号不足以纠偏）之间取得平衡。

##### 图构建的完整算法

以蛋白 $i$ 为例：
1. 从 Pearson 矩阵 $R$ 中取出第 $i$ 行——蛋白 $i$ 与所有其他 4,421 个蛋白的相关系数
2. 将 $R_{ii}$ 设为 $-\infty$（排除自环——蛋白不与自身建立边）
3. 用 `np.argsort` 找出相关度最高的 5 个邻居：`top_k = argsort(R[i])[-5:]`
4. 对每个候选邻居 $j$：如果 $|R_{ij}| \geq 0.7$，建立无向边 `[i, j]` 和 `[j, i]`
5. 如果 5 个候选邻居中没有任何一个满足 $|r| \geq 0.7$，蛋白 $i$ 成为**孤立节点**（无邻居）——它与其他蛋白都不高度共表达

**孤立节点怎么办？** 在 GNN 的消息传递中，孤立节点的更新公式退化为 $h_i^{(new)} = W \cdot h_i^{(old)}$——只经过自身权重变换，无邻居聚合。在残差连接下，$h_i^{(new)} = h_i^{(old)} + W \cdot h_i^{(old)}$——仅做轻度的自身微调。这是合理的：如果一个蛋白与任何其他蛋白都不高度共表达，那么没有邻居信息可以利用——GNN 不应该强行改变它的预测。

#### 4.5 第三步：GraphConv 的消息传递——信息如何沿边流动

##### 图卷积的一阶近似

`torch_geometric.nn.GraphConv`（无 bias 版本，不带残差连接时的原始形式）对节点 $i$ 的更新为：

$$h_i^{(new)} = W \cdot h_i^{(old)} + W \cdot \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{d_i \cdot d_j}} \cdot h_j^{(old)}$$

其中：
- $h_i^{(old)} \in \mathbb{R}^{B}$：蛋白 $i$ 的特征向量——在 batch 中 $B$ 个样本上的预测 log2 表达值。这是残差解码器输出的 `y_raw` 的第 $i$ 列。
- $\mathcal{N}(i)$：蛋白 $i$ 的邻居集合（最多 5 个）
- $d_i = |\mathcal{N}(i)|$：节点 $i$ 的度数
- $W \in \mathbb{R}^{B \times B}$：可学习的权重矩阵——**所有节点共享同一个 $W$**（这是 GNN 参数高效的关键）
- $h_i^{(new)} \in \mathbb{R}^{B}$：更新后的蛋白特征

##### 公式逐项拆解

**项 1 —— 自环项：**$W \cdot h_i^{(old)}$

蛋白 $i$ 保留自己对自身表达模式的"判断"，但经过 $W$ 的线性变换。$W$ 是一个 $B \times B$ 矩阵——在 batch size=256 时，$W$ 有 $256^2 = 65,536$ 个参数。它学习的是：蛋白的表达模式应该如何被重新加权才能更好地与邻居信息融合。

**项 2 —— 邻居聚合项：**$W \cdot \sum_{j \in \mathcal{N}(i)} \frac{1}{\sqrt{d_i \cdot d_j}} \cdot h_j^{(old)}$

这是 GNN 的核心。对蛋白 $i$ 的每个邻居 $j$：
- 取邻居的特征 $h_j^{(old)}$（邻居在整个 batch 上的预测值）
- 乘以归一化因子 $\frac{1}{\sqrt{d_i \cdot d_j}}$
- 对全部邻居求和
- 左乘相同的 $W$（与自环项共享）

**项 3 —— 对称归一化因子：**$\frac{1}{\sqrt{d_i \cdot d_j}}$

为什么需要归一化？考虑两种情况：
- 蛋白 A 有 5 个邻居（$d_A=5$），蛋白 B 有 5 个邻居（$d_B=5$）→ 归一化因子 = $1/\sqrt{25} = 1/5$
- 蛋白 C 有 5 个邻居（$d_C=5$），蛋白 Hub 有 50 个邻居（$d_{Hub}=50$）→ 归一化因子 = $1/\sqrt{250} \approx 1/15.8$

第二种情况下归一化因子更小——因为 Hub 蛋白被很多蛋白"仰仗"（以它为邻居），如果不用更小的权重稀释 Hub 的贡献，Hub 的信息会在图中被过度放大，主导所有邻居的更新。对称归一化确保了信息在图中的传播是**平衡的**：高度数节点的每条边携带更少的信息量。

**没有 bias 的原因**：GraphConv 的 bias 会给所有节点的所有特征维度加一个全局偏置——在蛋白预测任务中，这相当于给所有蛋白统一加/减一个常数。这没有生物学意义（不同蛋白的表达水平差异很大，不存在统一的偏移），且会与残差分解中的 baseline head 和校准头产生冗余。

##### 具体消息传递的数值示例

假设 batch size=4，蛋白 0（称为 RPS3）有 2 个邻居：蛋白 17（RPS5，$d_{17}=3$）和蛋白 88（RPL10，$d_{88}=4$）。蛋白 0 自己有 $d_0=2$ 个邻居。

在某个 batch 中：
```
h_0  = [18.2, 19.5, 17.8, 20.1]  ← RPS3 的解码器初始预测
h_17 = [17.8, 19.3, 17.2, 19.8]  ← RPS5 的解码器初始预测  
h_88 = [18.5, 19.8, 18.0, 20.5]  ← RPL10 的解码器初始预测
```

邻居聚合（先不看 W，假设 W 为单位矩阵以理解几何直觉）：

$$\text{aggregated} = \frac{h_{17}}{\sqrt{d_0 \cdot d_{17}}} + \frac{h_{88}}{\sqrt{d_0 \cdot d_{88}}} = \frac{h_{17}}{\sqrt{6}} + \frac{h_{88}}{\sqrt{8}} = 0.408 \cdot h_{17} + 0.354 \cdot h_{88}$$

$$= [7.27, 7.88, 7.02, 8.08] + [6.54, 7.01, 6.37, 7.26] = [13.81, 14.89, 13.39, 15.34]$$

$$h_0^{(new)} = h_0 + \sum \text{neighbors} = [32.01, 34.39, 31.19, 35.44]$$

等等——这个数值太大了。我们没有除以 batch 维度，而且直接加会导致数值爆炸。这就是为什么需要 $W$ 矩阵——它不仅是可学习的变换，也是**缩放器**：初始零初始化让输出接近零，避免数值溢出；训练中 $W$ 学到合适的缩放比例。

##### 为什么用 1 层而非多层 GNN

多层 GNN 会导致**过平滑（over-smoothing）**——经过 2-3 层后，所有蛋白节点的表示会趋同。数学上，每层 GNN 等价于邻接矩阵的谱滤波的一次迭代；多层就是多次滤波——高频（局部差异）被反复衰减，最终只剩直流分量（全局均值）。

物理直觉：1 层 GNN 让蛋白直接聚合其 K 个邻居的信息（1-hop）。2 层会让蛋白聚合邻居的邻居的信息（2-hop）——RPS3 不仅受 RPS5 影响，还受 RPS5 的邻居（可能包括与 RPS3 无关的蛋白）影响。层数越多，"六度分隔"效应越强，最终所有蛋白都在聚合全局均值——失去了个性化的预测能力。**1 层是蛋白共表达图的合理选择**：蛋白只与直接共表达的伙伴交换信息。

#### 4.6 设计决策：残差连接 + 零初始化

##### 为什么零初始化不能单独使用（重要修正）

如果零初始化 GraphConv 且不用残差连接：
```python
y_t = self.gnn(y_t, self.edge_index)  # W=0 → 输出全是0
y_raw = y_t.t()                        # 摧毁了所有解码器输出!
```

训练第 0 步，解码器的所有努力被 GNN 一键清零——loss 爆炸，训练无法恢复。**这是此前版本的设计缺陷。**

##### 修正方案：残差连接

```python
y_t = y_raw.t()
delta_gnn = self.gnn(y_t, self.edge_index)  # W=0 → delta=0
y_t = y_t + delta_gnn                        # = y_t（恒等映射）
y_raw = y_t.t()
```

这是 ResNet（He et al., 2016）的核心思想：让网络学习**增量修正**而非完整变换。$F(x) = H(x) - x$ 而非 $F(x) = H(x)$。当 $F(x)$ 的初始化输出为零时，模块初始为恒等映射——上层训练不受新模块影响，GNN 的贡献在训练中逐渐"生长"出来。

这与我们架构中的三处设计形成统一：
- 残差分解解码器：预测 = 多个残差头的求和，各头 L2 正则趋向零
- 校准头：零初始化 → 初始 $\delta_{cal} = 0$
- GNN 层：零初始化 + 残差连接 → 初始等效于无 GNN

所有"可选增强组件"遵循同一设计哲学：**从零出发，增量学习**。这使得模型可以分层训练——先跑 Encoder+Decoder（校准和 GNN 还不存在），确认收敛后加入校准（从零开始增量学习），确认收敛后加入 GNN（从零开始增量学习）。

##### 残差连接的梯度流优势

残差连接创造了一条"短路"梯度路径：

$$\frac{\partial \mathcal{L}}{\partial y_{raw}} = \frac{\partial \mathcal{L}}{\partial y_{pred}} \cdot \frac{\partial y_{pred}}{\partial y_{raw}} + \frac{\partial \mathcal{L}}{\partial y_{pred}} \cdot \frac{\partial y_{pred}}{\partial \text{gnn}} \cdot \frac{\partial \text{gnn}}{\partial y_{raw}}$$

第一项直接回传（跳过 GNN），第二项经过 GNN。即使 GNN 的梯度很小（训练初期），第一项确保了 loss 信号仍能有效回传到解码器和编码器——GNN 不会成为梯度瓶颈。

#### 4.7 为什么图结构固定（不参与梯度更新）

三个独立理由：

1. **数据纪律（最关键的）**：图结构从训练集 Pearson 相关矩阵构建——这是从**标签数据**（训练集蛋白表达值）计算出的统计量。如果训练过程中图结构可更新，等价于用标签信息在调整模型架构——这类似于在训练集上做特征选择再在测试集上评估一样，构成信息泄露的灰色地带。固定图结构避免了这一问题。

2. **计算不可行性**：$P(P-1)/2 \approx 9.8 \times 10^6$ 对蛋白的 Pearson 相关计算，每个蛋白对需要遍历 $N$ 个样本并计算内积——每 epoch 重算一次需要约 10-30 分钟。固定图把这一步骤变成一次性的预处理。

3. **生物学合理性**：蛋白共表达关系主要由基因组编码的蛋白功能和调控关系决定——在稳态条件下（不同实验条件 = 不同稳态），共表达结构是相对稳定的。训练过程中，模型学到的是**如何利用这些已知关系来改进预测**，而不是**发现新的关系**（后者需要更多的数据和不同的实验设计）。

#### 4.8 性能优化说明

`build_protein_graph` 中的嵌套循环是 $O(P^2)$：

```python
for i in range(P):        # 4,422 次
    for j in range(i+1, P):  # 平均 2,210 次 = 约 9.8M 次迭代
        r = np.corrcoef(y[valid, i], y[valid, j])[0, 1]
```

每次迭代调用一次 `np.corrcoef`（Python 函数调用开销 + NumPy 内部计算）。预估耗时 15-30 分钟。

**向量化加速方案**（如果太慢再采用）：

```python
# 步骤 1: 对每列做 z-score 标准化（mask-aware）
y_centered = y - np.nanmean(y, axis=0)  # (N, P)
# 用 mask 覆盖 NA 为 0
y_centered = np.nan_to_num(y_centered, nan=0.0)
# 计算每列的范数（仅有效样本上的标准差×sqrt(N)）
norms = np.sqrt(np.nansum(y_centered**2, axis=0))  # (P,)
norms[norms == 0] = 1.0
y_normed = y_centered / norms  # (N, P)
# 步骤 2: 相关矩阵 = 内积 (N, P)^T × (N, P) / N
corr_matrix = (y_normed.T @ y_normed) / y_normed.shape[0]  # (P, P)
# 步骤 3: 裁剪到 [-1, 1]（处理浮点精度）
corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
```

向量化版本约 10 秒完成。但精度略低（未逐个蛋白对地处理 mask 差异——两个蛋白的公共有效样本数可能不同于其他对）。**建议先跑循环版本确保逻辑正确，用结果作为参考标准；后续训练时切换到向量化版本以节省时间。**

#### 4.9 实现

```python
"""
蛋白共表达图构建模块

从训练集蛋白表达数据中计算 Pearson 相关矩阵 → KNN 图，
用作 GNN 的固定图结构（不参与梯度更新）。

数据纪律: 所有统计量仅从训练集计算。
"""
import numpy as np
import torch


def build_protein_graph(y_train_log2, mask_train, k=5, threshold=0.7):
    """
    从训练集蛋白表达数据构建 KNN 图

    参数:
        y_train_log2: (N_train, P) log2 蛋白表达 DataFrame
        mask_train:   (N_train, P) 观测 mask DataFrame
        k:            K 近邻数
        threshold:    |Pearson r| 低于此值的边被裁减

    返回:
        edge_index: torch.LongTensor (2, E)  PyG 格式边列表
    """
    P = y_train_log2.shape[1]
    y = y_train_log2.values.astype(np.float64)  # (N, P)
    m = mask_train.values.astype(bool)

    # --- 计算蛋白-蛋白 Pearson 相关矩阵 ---
    corr_matrix = np.zeros((P, P), dtype=np.float32)
    pair_count = 0

    for i in range(P):
        for j in range(i + 1, P):
            valid = m[:, i] & m[:, j]
            if valid.sum() < 10:        # 少于 10 个公共观测 → 不可靠
                continue
            r = np.corrcoef(y[valid, i], y[valid, j])[0, 1]
            if not np.isnan(r):
                corr_matrix[i, j] = r
                corr_matrix[j, i] = r
                pair_count += 1

    print(f"蛋白相关矩阵: {P}×{P}, {pair_count} 对有效计算 "
          f"({pair_count / (P*(P-1)/2) * 100:.1f}% 覆盖)")

    # --- KNN + threshold 构建边 ---
    edges = []
    for i in range(P):
        scores = corr_matrix[i].copy()
        scores[i] = -np.inf           # 排除自环
        top_k_indices = np.argsort(scores)[-k:]
        for j in top_k_indices:
            r = corr_matrix[i, j]
            if abs(r) >= threshold:
                edges.append([i, j])   # 无向边：只加一次

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # --- 图统计 ---
    degrees = np.bincount(edge_index[0].numpy(), minlength=P)
    isolated = (degrees == 0).sum()
    print(f"图统计: {P} 节点, {edge_index.shape[1]} 边, "
          f"平均度 {edge_index.shape[1]/P:.1f}, "
          f"{isolated} 孤立节点 ({isolated/P*100:.1f}%)")

    return edge_index
```

#### 4.10 GNN 层集成到 AIVCModel

在 `model.py` 的 `AIVCModel.__init__` 中添加：

```python
if use_gnn:
    from torch_geometric.nn import GraphConv
    self.gnn = GraphConv(n_proteins, n_proteins, bias=False)
    # 零初始化 + 残差连接 → 初始等效于无 GNN
    nn.init.zeros_(self.gnn.lin.weight)
```

在 `AIVCModel.forward` 中：
```python
if self.gnn is not None:
    y_t = y_raw.t()                              # (N,P) → (P,N)
    delta_gnn = self.gnn(y_t, self.edge_index)   # GNN 增量修正
    y_t = y_t + delta_gnn                        # 残差连接
    y_raw = y_t.t()                              # (P,N) → (N,P)
```

#### 4.11 自检

```python
from baseline.data import load_raw_data, preprocess
from baseline.protein_graph import build_protein_graph

meta, prot = load_raw_data()
y_log2, mask_matrix, meta, _, train_mask = preprocess(meta, prot)

edge_index = build_protein_graph(
    y_log2.loc[train_mask],
    mask_matrix.loc[train_mask],
    k=5,
    threshold=0.7
)

# 结构检查
assert edge_index.shape[0] == 2, f"edge_index 第一维应为 2，实际 {edge_index.shape[0]}"
assert edge_index.max() < len(y_log2.columns), "边索引超出蛋白范围!"
assert edge_index.min() >= 0

# 覆盖检查
n_proteins = len(y_log2.columns)
nodes_in_graph = len(set(edge_index[0].tolist()) | set(edge_index[1].tolist()))
print(f"图中覆盖蛋白: {nodes_in_graph}/{n_proteins} ({nodes_in_graph/n_proteins*100:.1f}%)")

# GNN 层前向测试（随机输入，无残差连接时输出应为 0）
from torch_geometric.nn import GraphConv
gnn = GraphConv(n_proteins, 4, bias=False)
nn.init.zeros_(gnn.lin.weight)
x = torch.randn(n_proteins, 4)
out = gnn(x, edge_index)
assert out.abs().max() < 1e-6, "零初始化 GNN 输出应为零"
print("GNN 零初始化验证通过")
```

**结果记录**：
- 蛋白数 P：4422
- 边数 E：19924
- 平均度：4.5
- 孤立节点数：0（0.0%，✅ 全部蛋白都有至少 1 个共表达邻居）
- 图中覆盖蛋白比例：100%（4422/4422）
- build_protein_graph 运行耗时：~10-15 秒（向量化实现，非逐对循环）
- 问题/备注：① 向量化 mask-aware Pearson（y_centered.T @ y_centered 矩阵运算）替代逐对循环，约 10s 完成。② GNN 层使用自定义 ProteinGraphSmooth（α 可学习平滑）替代 PyG GraphConv——PyG 的 GraphConv 要求 in_channels=out_channels，但此处节点特征维度 = batch_size，与 n_proteins 不匹配。③ 残差连接（y_smoothed = y_t + α*(neighbor_mean - y_t)）保证初始 α=0.1 时接近恒等映射，训练中 α 从 0.10 → 0.14

---

### B5: GNN 集成 + 全模型测试 ⏱ 预计 1.5h

- [ ] 开始时间：\_\_:\_\_
- [ ] 完成时间：\_\_:\_\_

**文件**：[baseline/model.py](baseline/model.py)

#### 5.1 原理：转置 (N, P) → (P, N) 背后的张量语义

##### 为什么需要转置——两种视角的切换

残差解码器输出的 `y_raw` 形状是 `(N, P)`——这是**样本视角**：
- 第 $n$ 行 = 第 $n$ 个实验条件下，4,422 个蛋白的预测 log2 表达值
- 这是"自然的"形状：模型接收 N 个样本，为每个样本输出 P 个预测

但 GNN 操作的是**蛋白视角**——图上的 4,422 个节点，每个节点的"特征"是该蛋白在不同样本上的行为模式。所以需要 `(P, N)`：
- 第 $i$ 行 = 蛋白 $i$ 在 N 个样本上的预测 log2 表达值（这个 256 维向量是蛋白 $i$ 在当前 batch 中的"表达模式快照"）

`y_raw.t()` 实质上是**视角切换**——从"每个样本的蛋白谱"变成"每个蛋白的样本谱"。

##### PyTorch 中 transpose 的语义

`.t()` 对于 2D tensor 等价于 `.transpose(0, 1)`——交换第 0 维和第 1 维。关键特性：
- **内存共享**：`.t()` 返回的是原 tensor 的一个**视图（view）**，不复制数据。两个 tensor 指向同一块物理内存，只是步长（stride）不同。
- **梯度传播**：`.t()` 是 PyTorch autograd 中的一个可微操作——梯度会正确地通过转置回传：$\frac{\partial L}{\partial X} = (\frac{\partial L}{\partial X^T})^T$
- **contiguous 问题**：`.t()` 后的 tensor 通常不是 contiguous 的（内存布局不连续）。某些操作（如 `.view()`）要求 contiguous tensor。如果后续需要 reshape，用 `.t().contiguous()`。

在 GNN 场景中：
```python
y_t = y_raw.t()                        # view，共享内存
delta_gnn = self.gnn(y_t, edge_index)  # 在 (P,N) 上做消息传递
y_t = y_t + delta_gnn                  # 原地加法（新 tensor）
y_raw = y_t.t()                        # view，转回 (N,P)
```

梯度流：`loss → ∂L/∂y_raw → ∂L/∂y_t（通过 .t() 的 VJP）→ ∂L/∂(GNN 参数)（通过 GraphConv 的 VJP）→ ∂L/∂y_t（残差连接的分支）→ ...`

##### 为什么确保 edge_index 在正确设备上

```python
model.edge_index = edge_index.to(device)
```

这是一个常见的坑。`edge_index` 是 LongTensor，如果在 CPU 上而模型在 GPU 上，GNN 前向会报错 `Expected all tensors to be on the same device`。在训练循环中，确保 `edge_index` 随模型一起 `.to(device)`。

##### GNN 层插入位置的完整理由链

```
encoder → embedding → decoder → y_raw → [GNN] → y_smoothed → + δ_cal → y_pred
```

GNN 平滑的是**生物学共表达一致性**。$\delta_{cal}$ 是仪器/板偏差的**技术校正**——如果 GNN 在校准之后，仪器偏差会被当作生物学信号在蛋白之间"传播"（仪器 A 的偏差会污染仪器 A 上所有样本的所有蛋白的预测），导致校准失效。

反之，GNN 在校准之前：先让生物学信号在蛋白邻居间平滑（"核糖体蛋白应该同涨同跌"），再加仪器校正（"但仪器 A 读数偏高 0.3"）。两个校正互不干扰。

#### 5.2 全模型前向的逐步骤追踪（Day 2 里程碑）

以一个具体的 batch（N=4, P=4422）为例，追踪每一步的 tensor shape 和语义：

```python
model = AIVCModel(dim_in=256, n_proteins=4422, use_gnn=True)
model.edge_index = edge_index.to(device)
x = torch.randn(4, 256, device=device)

# Step 1: Encoder
emb = model.encoder(x)                    # (4, 256)  条件嵌入

# Step 2: Decoder
dec_out = model.decoder(emb)              # dict, 每个value (4, 4422)
# dec_out["baseline"]   = 菌株基础的蛋白表达
# dec_out["delta_drug"] = 药物的蛋白响应（FC的模型表示）
# dec_out["y_raw"]      = 四项求和

# Step 3: GNN（内部）
y_t = dec_out["y_raw"].t()               # (4, 4422) → (4422, 4)
# y_t[i, :] = 蛋白i在4个样本上的初始预测值

delta_gnn = model.gnn(y_t, model.edge_index)  # (4422, 4)  
# 每个蛋白聚合并了K个共表达邻居的信息后得到的增量修正
# W=0 → delta_gnn ≈ 0 → 残差连接后 ≈ y_t 不变

y_t = y_t + delta_gnn                     # (4422, 4)  修正后的蛋白表达模式
y_smoothed = y_t.t()                      # (4, 4422)  回到样本视角

# Step 4: Calibration
delta_cal = model.calibration(emb)        # (4, 4422)  仪器/板偏移

# Step 5: Final
y_pred = y_smoothed + delta_cal           # (4, 4422)  最终预测

# 交付 dict
out = model(x)  # 包含全部7个key
```

关键验证点：
```python
# 1. 恒等式仍成立
assert torch.allclose(out["y_pred"], out["y_raw"] + out["delta_cal"], atol=1e-5)

# 2. GNN 未启用时 vs 启用时（W=0），y_pred 应该完全一致
model_no_gnn = AIVCModel(dim_in=256, n_proteins=4422, use_gnn=False)
out_no = model_no_gnn(x)
assert torch.allclose(out["y_pred"], out_no["y_pred"], atol=1e-5), \
    "零初始化 GNN + 残差连接应与无 GNN 输出一致"
print("B5 GNN 集成 ✅  残差连接保证初始恒等映射")
```

**与 Person C 的接口确认**：
- [ ] C 的 `correlation_consistency_loss` 拿到了 `edge_index`：是 / 否
- [ ] C 的训练循环将 `edge_index` 传入模型的 device：是 / 否
- [ ] C 确认 `y_pred` shape = (N, 4422) 与现有 loss 函数兼容：是 / 否

**结果记录**：
- GNN 前向通过：是（dim_in=75, P=4422 真实数据 2 epoch 训练正常）
- 零初始化恒等映射验证通过：N/A（自定义 ProteinGraphSmooth 用 α=0.1 初始而非零初始化 W；若需严格恒等映射可设 α_init=0）
- 预测恒等式（含 GNN+残差）通过：是（y_pred = GNN(y_raw) + δ_cal）
- 参数量（含 GNN）：2,869,279（dim_in=75, P=4422），GNN 仅 1 个可学习参数 α
- 问题/备注：① ProteinGraphSmooth 仅含 1 个标量参数 α（可学习平滑强度），训练 2 epoch 后 α 从 0.10 → 0.14。② edge_index 通过 model.set_edge_index() 注入。③ 与 Person C 的接口确认项待联调（correlation_consistency_loss 需 edge_index）

---

### B6: 架构消融实验 ⏱ 预计 2h

- [ ] 开始时间：\_\_:\_\_
- [ ] 完成时间：\_\_:\_\_

**文件**：[experiments/ablation_arch.py](experiments/ablation_arch.py)（新建）

#### 6.1 原理：消融实验的认识论基础

##### 从相关到因果——科学方法在架构验证中的应用

当你看到一个模型的 val_both Per-Protein R² = 0.520，你不能说"残差分解贡献了 0.520"——这个数字是模型中**所有组件**（encoder 结构、decoder 形式、校准头、训练超参、数据质量……）共同作用的结果。就像你不能说"这杯咖啡好喝是因为加了糖"——它可能是咖啡豆好、水温合适、牛奶新鲜、以及糖的共同效果。

消融实验 = **控制变量法**。只改变一个因素，保持其他一切不变，观察性能变化。变化量 = 该因素对最终效果的净贡献。

##### 为什么只看指标不够——辛普森悖论的风险

考虑一种可能的（但误导性的）情况：

| 场景 | ResidualDecoder 的 PP R² | SimpleDecoder 的 PP R² |
|------|:---:|:---:|
| val_strain_only（seen 菌株） | 0.65 | 0.68 |
| val_chem_only（unseen 化合物） | 0.52 | 0.45 |
| val_both（双 unseen） | 0.42 | 0.30 |
| **平均** | **0.53** | **0.48** |

如果你只看平均，残差分解提升了 0.05。但拆开看：在 seen 菌株场景下残差分解反而**更差**（-0.03），但在 OOD 场景下**大幅更好**（+0.07 和 +0.12）。这说明残差分解的归纳偏置在 OOD 泛化上发挥了关键作用，而在 seen 场景下端到端 MLP 的优势（更大的假设空间、更少的约束）表现出来。

这个故事**只有分场景报告才能被看到**——只看全局平均，你会错过最关键的发现。这就是为什么消融必须分 val_chem_only / val_strain_only / val_both / val_time 四个场景独立报告。

##### 公平对比的四条纪律

1. **相同参数量**：两个被对比的模型必须拥有大致相等的可训练参数。如果 A 有 3M 参数而 B 有 1.5M，B 可能输在容量不够，而非架构不好。$SimpleDecoder(hidden=512)$ 通过加宽隐藏层拉齐参数——这是一个被广泛接受的公平对比方法（参见 ResNet 原始论文中 plain vs residual 的对比：两者参数量相同，仅差残差连接）。

2. **相同随机种子**：不同 seed 的训练结果可能波动 0.02-0.05 Per-Protein R²（尤其在只有 5,920 训练样本的情况下）。所有消融实验必须用同一 seed 初始化模型参数和 batch 顺序。如果要在论文中报告，建议跑 3 个 seed 取均值 ± 标准差。

3. **相同训练超参**：学习率、batch size、epoch 数、早停标准必须完全一致。消融的不是训练策略，是架构设计。

4. **相同数据流水线**：预处理参数、蛋白过滤阈值、mask 构建方式必须一致。数据差异不是消融的目标。

#### 6.2 四组消融的详细逻辑

##### 消融 1：`full_residual`（基准线）

```
ConditionEncoder + ResidualDecoder + CalibrationHead + 无 GNN
```

这是所有创新组件的完整组合。它建立了消融的**参照系**——其他三个实验都与它做减法或加法对比。

##### 消融 2：`no_residual` — 量化残差分解的归纳偏置贡献

**改变**：`ResidualDecoder` → `SimpleDecoder(hidden=512)`

**科学问题**：将蛋白表达显式分解为 baseline + deltas 的归纳偏置，是否比让 MLP 自由学习从条件嵌入到蛋白表达的黑盒映射更好？

**两个假设的竞争**：
- $H_0$（无差异）：残差分解的结构化约束对性能无帮助，端到端 MLP 的更大自由度能学到等效甚至更好的映射
- $H_1$（残差分解更优）：在 OOD 场景下（val_chem_only, val_both），残差分解「部分失效」的 OOD 行为比端到端 MLP「全局崩塌」更鲁棒

**可证伪条件**：如果 `no_residual ≥ full_residual` 在 val_both 上 → 残差分解无效 → 回退到端到端 MLP 解码器（方案文档 3.1.6 节回退路径）。

**特别注意**：SimpleDecoder 构建时 `hidden=512` 是要确保总参数量与 ResidualDecoder（5 heads × 128）可比。残差分解 5 头共享 256→128 瓶颈层，端到端 MLP 自己 256→512→512→4422——约 2.9M vs 约 3.0M，差异 < 5%。

##### 消融 3：`no_calibration` — 量化批次校准的贡献

**改变**：`CalibrationHead` → `lambda emb: torch.zeros(...)`

**科学问题**：仪器/板级系统偏差是否真实存在于数据中，且被独立的可训练校准头捕获？还是校准头在训练中学到的只是噪声（过拟合到训练集的特定板分布）？

**关键指标**：除了全局 Per-Protein R²，还要**按 instrument 分组计算 R²**：
```python
for instrument_id in range(7):
    mask_inst = meta["instrument"] == instrument_id
    r2_inst = evaluate_global_r2(y_true[mask_inst], y_pred[mask_inst], mask[mask_inst])
```
如果校准有效，`full_residual` 的跨仪器 R² 方差应**显著小于** `no_calibration` 的跨仪器 R² 方差——因为校准头抹平了仪器间的系统性偏差。

**可证伪条件**：如果 `no_calibration` 与 `full_residual` 在所有指标上无差异（且按 instrument 分组 R² 方差相同）→ 批次效应在数据中不显著 → 可移除校准分支以简化模型。

##### 消融 4：`with_gnn` — 量化蛋白图 GNN 的增量贡献

**改变**：在 `full_residual` 基础上增加 GNN 层（含残差连接）。

**科学问题**：从训练数据中学到的蛋白共表达图结构，是否能提供超出残差分解的额外预测信号？GNN 的贡献是独立的还是与残差分解冗余？

**关键指标**：分蛋白 R² 中位数（消除孤立蛋白的离群预测）和蛋白-蛋白预测相关矩阵与训练集真实相关矩阵的 Frobenius 范数差。

**可证伪条件**：如果 `with_gnn` 与 `full_residual` 无差异 → 图信息已被残差分解隐式捕获 → 跳过 GNN（方案文档 3.1.6 节回退路径 "GNN 层无提升则跳过"）。

**注意**：GNN 的参数量非常小（GraphConv 仅一个 $N_{batch} \times N_{batch}$ 的权重矩阵，无 bias），增量贡献如果存在，完全来自**图结构信息**而非参数容量增加。

##### 消融矩阵

| 实验名 | 解码器 | 校准 | GNN | 验证的科学问题 |
|--------|:------:|:----:|:---:|--------------|
| `full_residual` | ResidualDecoder | ✅ | ❌ | **基线**——所有创新组件的基准性能 |
| `no_residual` | SimpleDecoder | ✅ | ❌ | 残差分解的归纳偏置：结构化分解 vs 黑盒映射 |
| `no_calibration` | ResidualDecoder | ❌ | ❌ | 批次校准：仪器/板偏差是否真实存在并可被学习 |
| `with_gnn` | ResidualDecoder | ✅ | ✅ | 蛋白图 GNN：共表达结构是否提供独立于残差分解的增量信号 |

#### 6.3 结果解读的统计注意事项

单次训练的 Per-Protein R² 差异 < 0.02 时，**不能轻易下结论说 A 比 B 好**。理由是：

- 随机种子波动 ±0.01-0.03
- Batch 采样的随机性 ±0.005-0.01
- 5,920 个训练样本对 4,422 维输出——模型处于欠定状态，收敛路径有随机性

**建议的判断标准**：
- 差异 ≥ 0.05：可以确信 A 比 B 好
- 差异 0.02-0.05：有趋势但需多 seed 验证
- 差异 < 0.02：实质无差异，选更简单的方案（奥卡姆剃刀）

#### 6.4 结果表格的完整呈现

除了跑出指标，你需要填这张表：

| 实验名 | 参数量 | val_strain | val_chem | val_both | val_time | test_strain | test_chem | test_both | test_time |
|--------|:------:|:---------:|:--------:|:--------:|:--------:|:----------:|:---------:|:---------:|:---------:|
| full_residual | | | | | | | | | |
| no_residual | | | | | | | | | |
| Δ (残差分解贡献) | — | | | | | | | | |
| no_calibration | | | | | | | | | |
| Δ (校准贡献) | — | | | | | | | | |
| with_gnn | | | | | | | | | |
| Δ (GNN 贡献) | — | | | | | | | | |

每个单元格是 Per-Protein R² 中位数。

#### 6.5 消融结论的写作模板

训练完成后，用这个模板写结论（填入实际数值）：

```
架构消融实验结论：

1. 残差分解解码器 vs 端到端 MLP：
   val_both 上残差分解提升 Per-Protein R² 中位数 __X__，
   val_chem_only 上提升 __Y__。
   结论：[残差分解是核心创新，贡献显著 / 提升不显著，建议回退]

2. 批次校准分支：
   跨 instrument 的 R² 方差从 __A__ 降至 __B__（降幅 __C__%）。
   Global R² 提升 __D__。
   结论：[批次效应确实存在，校准有效 / 批次效应可忽略，可移除]

3. 蛋白共表达图 GNN：
   Per-Protein R² 中位数提升 __E__。
   预测蛋白相关矩阵与真实相关矩阵的 F-范数差缩小 __F__。
   结论：[GNN 有正向贡献，保留 / 贡献不显著，跳过]
```

#### 6.6 实现

```python
"""
架构消融实验

消融逻辑：每次只改变一个变量，保持参数量可比，
从而将性能差异归因于结构偏置而非模型容量。
"""
import torch
import numpy as np
from baseline.model import AIVCModel
from baseline.decoder import SimpleDecoder


def build_model(config, dim_in, n_proteins):
    """
    按配置构建模型

    config:
        {"decoder": "residual"|"simple", "calibration": True|False, "gnn": True|False}

    返回:
        model: nn.Module，forward(x) → pred_dict
        n_params: int，可训练参数总数
    """
    model = AIVCModel(dim_in, n_proteins, use_gnn=config.get("gnn", False))

    if config["decoder"] == "simple":
        model.decoder = SimpleDecoder(
            dim_emb=model.dim_emb,
            n_proteins=n_proteins,
            hidden=512,
        )

    if not config["calibration"]:
        def zero_cal(emb):
            return torch.zeros(emb.shape[0], n_proteins, device=emb.device)
        model.calibration = zero_cal

    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params


ABLATIONS = [
    {"name": "full_residual",   "decoder": "residual", "calibration": True,  "gnn": False},
    {"name": "no_residual",     "decoder": "simple",   "calibration": True,  "gnn": False},
    {"name": "no_calibration",  "decoder": "residual", "calibration": False, "gnn": False},
    {"name": "with_gnn",        "decoder": "residual", "calibration": True,  "gnn": True},
]


def run_ablation(ablation_configs, train_data, val_data, device, seed=42):
    """
    跑一组消融实验，每个实验相同 seed + 相同超参。

    返回:
        results: list[dict]，每个 dict 包含 name, n_params,
                 val_strain_pp_r2, val_chem_pp_r2, val_both_pp_r2, val_time_pp_r2,
                 test_strain_pp_r2, test_chem_pp_r2, test_both_pp_r2, test_time_pp_r2
    """
    results = []
    for cfg in ablation_configs:
        name = cfg["name"]
        print(f"\n{'='*60}")
        print(f"消融: {name}")
        print(f"  解码器: {'残差分解' if cfg['decoder']=='residual' else '端到端MLP'}")
        print(f"  校准:   {'✅' if cfg['calibration'] else '❌'}")
        print(f"  GNN:    {'✅' if cfg['gnn'] else '❌'}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        model, n_params = build_model(cfg, ...)
        print(f"  参数量: {n_params:,}")

        # ... 训练 + 评估（使用与 Person C 统一的训练入口）...
        result = {"name": name, "n_params": n_params, ...}
        results.append(result)

    return results
```

#### 自检
- [ ] 四个配置均能成功 `build_model()` 不报错
- [ ] 参数量最大差值 < 20%（报告各配置参数量）
- [ ] `no_residual` 的 SimpleDecoder 输出 dict 兼容现有接口（含 `delta_drug` 等 key）
- [ ] `no_calibration` 的 zero_cal 输出 shape = (N, n_proteins)
- [ ] 每个配置能跑通 1 epoch 训练（loss 下降，不报错）

**结果记录**：

> ⚠️ **训练尚未运行**。下表为自检阶段数据（模型构建 + 1 epoch 冒烟测试），完整消融需执行：
> ```bash
> python experiments/ablation_arch.py --run --epochs 100
> ```

| 实验名 | 参数量 (dim_in=75) | 构建自检 | 1-epoch 冒烟 | 备注 |
|--------|:------:|:---:|:---:|------|
| **full_residual** | 2,869,278 | ✅ | ✅ loss 245→15 | 基准 |
| no_residual | 3,118,668 | ✅ | 待跑 | SimpleDecoder hidden=512 |
| no_calibration | 2,565,400 | ✅ | 待跑 | _ZeroCalibration(n_proteins) |
| with_gnn | 2,869,279 | ✅ | ✅ loss 245→15, α 0.10→0.14 | GNN 平滑生效 |

- 参数量差异：17.5%（< 20%，公平对比 ✅）
- 全部 4 配置 build_model() 不报错 ✅
- SimpleDecoder 输出 dict 兼容现有接口 ✅
- _ZeroCalibration 输出 shape = (N, 4422) ✅
- full_residual + with_gnn 各跑通 2 epoch 训练 ✅

**消融结论**：
- 残差分解 vs 端到端的 Per-Protein R² 差值（val_both）：**待完整训练后填入** → 结论：________
- 校准分支的 Per-Protein R² 差值（val_both）：**待完整训练后填入** → 结论：________
- GNN 的 Per-Protein R² 差值（val_both）：**待完整训练后填入** → 结论：________
- 是否触发回退（是 / 否 / 哪个组件）：**待完整训练后判断**

---

## 完成检查清单

- [x] B1: `ConditionEncoder` 实现 + 自检
- [x] B2: `ResidualDecoder` + `SimpleDecoder` 实现 + 自检
- [x] B3: `CalibrationHead` 实现 + 自检
- [x] `AIVCModel` 集成 + 全接口自检（恒等式 + 梯度流）
- [x] B4: `build_protein_graph()` + GNN 层 + 自检
- [x] B5: GNN 集成到 `AIVCModel` + 端到端前向测试
- [x] B6: 四组消融实验代码完成 + 自检（完整训练待跑）
- [ ] 与 Person A 联调：特征维度确认 + 前向通过（dim_in=75 已确认，待 A 交付新特征后更新）
- [ ] 与 Person C 联调：pred_dict 接口确认 + loss 消费正常
- [x] 所有自检代码保留在文件中

---

## 问题记录

| 时间 | 问题描述 | 解决方案 | 状态 |
|------|---------|---------|:----:|
| 08-07 | PyG `GraphConv` 通道维度不匹配：`GraphConv(P, P)` 期望输入 (P, P) 但 transpose 后输入为 (P, batch_size) | 自定义 `ProteinGraphSmooth` 层（可学习 α 平滑），纯 PyTorch 实现，无外部依赖 | ✅ |
| 08-07 | `_ZeroCalibration` 返回 (N, 1) 而非 (N, n_proteins)，与 y_raw 加法时触发广播 | 添加 `n_proteins` 参数，返回 `torch.zeros(N, n_proteins)` | ✅ |
| 08-07 | 消融参数量差异在 P=100 测试时达 42.8% | 将 sanity check 改用 P=4422 真实蛋白数，差异降至 17.5% | ✅ |
| 08-07 | PyG 2.8 `GraphConv` 无 `.lin` 属性（改为 `lin_rel` + `lin_root`） | 放弃 PyG GraphConv，用自定义 `ProteinGraphSmooth` | ✅ |
| 08-07 | GBK 编码不支持 emoji（✅ ❌ ⚠️） | 全部替换为 ASCII 标记（[OK] [FAIL] [WARN]） | ✅ |
| 08-07 | B6 冒烟测试：full_residual + with_gnn 各 2 epoch 训练通过，loss 正常下降 | — | ✅ |
| 08-08 | B6 完整消融训练待跑（需 ~30-40 min CPU） | `python experiments/ablation_arch.py --run --epochs 100` | ⏳ |

---

*日志开始：2026-08-07 | 分支：lxy*
