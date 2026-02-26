# CT_cnn_256_0.07: 基于学习梯度下降的单角度CT图像重建

## 目录

- [项目概述](#项目概述)
- [数学原理](#数学原理)
- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [核心模块详解](#核心模块详解)
- [模型架构](#模型架构)
- [数据流水线](#数据流水线)
- [训练流程](#训练流程)
- [评估与测试](#评估与测试)
- [关键参数说明](#关键参数说明)
- [使用指南](#使用指南)
- [实验结果](#实验结果)
- [辅助工具脚本](#辅助工具脚本)
- [项目命名含义](#项目命名含义)

---

## 项目概述

本项目实现了一种**学习梯度下降 (Learned Gradient Descent, LGD)** 方法，用于从**单角度 Radon 变换**的傅里叶域测量数据中重建二维 CT 图像。

**核心思想**：给定单一投影方向（由方向向量 **beta** 参数化），系统通过以下步骤恢复二维图像：

1. 将前向问题建模为傅里叶域中的对角运算：`F = Phi * FFT(d)`
2. 通过 **Tikhonov 正则化反演** 计算初始估计（利用频率域对角结构的闭式解）
3. 通过 **学习梯度下降网络** (LGD) 对初始估计进行迭代优化，每次迭代中使用 CNN 代替传统的固定梯度更新

该方法将经典的傅里叶域物理模型与可训练的 CNN 更新算子相结合，采用展开优化 (unrolled optimization) 架构，在保持物理一致性的同时通过数据驱动的方式显著提升重建质量。

---

## 数学原理

### 1. 函数表示

图像 $f$ 用像素基函数（一阶基数 B-样条的张量积）展开：

$$f(x,y) = \sum_{k} c_k \cdot \varphi(x - k_x, y - k_y)$$

其中 $\varphi(x,y) = B_1(x) \cdot B_1(y)$，$B_1$ 为 $[0,1]$ 上的特征函数（一阶基数 B-样条）。

### 2. 傅里叶域前向模型

对于方向 $\alpha = \beta / \|\beta\|$，Radon 变换在频率 $\xi_j = 2\pi j / (N\|\beta\|)$ 处为：

$$F(\xi_j) = \Phi(\xi_j) \cdot \sum_{n} d_n \cdot \exp\left(-i \cdot n \cdot \frac{2\pi j}{N}\right)$$

矩阵形式：$\mathbf{F} = \Phi \cdot \text{FFT}(\mathbf{d})$，其中 $\Phi$ 为对角矩阵，由 B-样条傅里叶变换值构成。

### 3. B-样条傅里叶变换

$$\hat{B}_1(\xi) = e^{-i\xi/2} \cdot \text{sinc}\left(\frac{\xi}{2\pi}\right)$$

二维情况下：$\hat{\varphi}(\xi) = \hat{B}_1(\xi \cdot \alpha_x) \cdot \hat{B}_1(\xi \cdot \alpha_y)$

### 4. Tikhonov 正则化初始估计

利用对角结构的闭式解：

$$\hat{d} = \text{IFFT}\left(\frac{N \cdot \Phi^H \cdot \mathbf{F}_{obs}}{N \cdot |\Phi|^2 + \lambda}\right)$$

### 5. 噪声模型

观测数据：$F_{\text{observed}} = F_{\text{clean}} + \eta$

加性噪声：$\eta \sim \mathcal{CN}(0, \sigma^2)$，即复正态分布，实部和虚部独立同分布：

$$\text{Re}(\eta), \text{Im}(\eta) \sim \mathcal{N}\left(0, \frac{\sigma^2}{2}\right)$$

### 6. 学习梯度下降

将经典梯度更新 $d_{k+1} = d_k - \eta \cdot \nabla$ 替换为 CNN 预测的自适应更新，CNN 根据当前状态、梯度信息和累积记忆进行调整。

---

## 项目结构

```
CT_cnn_256_0.07/
├── models/                          # 核心源代码目录
│   ├── config.py                    # 全局配置参数
│   ├── model.py                     # 神经网络架构定义
│   ├── radon_transform.py           # 前向/伴随算子、数据生成、Tikhonov 求解器
│   ├── box_spline.py                # 基数 B-样条基函数
│   ├── image_generator.py           # 系数到图像的映射
│   ├── Data_Generator.py            # 离线数据集生成脚本
│   ├── train_offline.py             # 主训练脚本（离线模式，梯度累积）
│   ├── train_continue.py            # 续训脚本（从检查点继续训练）
│   ├── train.py                     # 早期版本训练脚本
│   ├── test.py                      # 在 Shepp-Logan 体模上评估
│   ├── tikhonov_eval.py             # 独立 Tikhonov 重建演示
│   ├── tikhonov_find.py             # 两阶段 lambda 搜索
│   ├── compare_best_vs_config_lambda.py  # lambda 选择对比
│   ├── theory_verification.py       # Float32 精度验证
│   ├── condition number.py          # 条件数分析
│   ├── theoretical_ct_best_model.pth     # 最佳模型权重
│   ├── theoretical_ct_best_model_0.07.pth # 达到 ~0.07 RES 的模型
│   ├── theoretical_ct_model.pth          # 最新模型权重
│   ├── checkpoints/                 # 迭代检查点 (iter 0, 500, 1000, ...)
│   └── Documentation_tex/           # 数学推导文档
│
├── data/                            # 数据集目录（由 Data_Generator.py 生成）
│   ├── train_dataset.pt             # 训练集（20,000 随机椭圆体模）
│   └── val_dataset.pt               # 验证集（2,000 Shepp-Logan 体模）
│
├── logs/                            # 训练日志
│   ├── training.log                 # 初始训练日志
│   ├── training_continue.log        # 续训日志
│   ├── training_progress.png        # 损失/RES 曲线图
│   └── continue_training_plot.png   # 续训曲线图
│
├── results/                         # 当前评估结果
│   └── shepp_logan_last.png         # 最新评估结果图
│
├── results1/                        # 历史结果存档
│
├── models_change/                   # 设计笔记与算法描述
│
├── mistake/                         # 调试笔记与 Bug 修复记录
│
├── test_/                           # 中间算子验证测试
│
├── test.py                          # 根目录下测试脚本
├── 噪声定义.md                       # 噪声定义数学文档
└── README.md                        # 本文件
```

---

## 环境依赖

### Python 版本
- Python >= 3.8

### 核心依赖库

| 库          | 用途                                      |
|------------|-------------------------------------------|
| PyTorch    | 深度学习框架（torch, torch.nn, torch.optim, torch.fft, torch.amp） |
| NumPy      | 数值计算、体模生成                          |
| Matplotlib | 结果可视化（使用 Agg 后端）                  |
| tqdm       | 数据生成进度条                              |

### 硬件要求
- 推荐使用 CUDA 兼容的 NVIDIA GPU（训练时使用 `cuda:0`）
- 无 GPU 时自动回退到 CPU（速度会显著降低）

### 安装

```bash
pip install torch torchvision numpy matplotlib tqdm
```

---

## 核心模块详解

### `models/config.py` — 全局配置

集中定义所有项目参数，包括：
- 图像尺寸 (`IMAGE_SIZE = 256`)
- 模型超参数（beta 向量、正则化类型、迭代次数、记忆单元数）
- 噪声配置（模式、强度、Tikhonov lambda）
- 训练配置（批大小、验证间隔、早停耐心值、梯度裁剪）
- 路径配置（模型保存、数据目录、日志目录）

### `models/model.py` — 神经网络架构

包含6个核心组件：

| 类名 | 功能 |
|------|------|
| `CoefficientMapping` | 二维系数网格 (H,W) 与一维向量之间的映射（行优先展平） |
| `RadonFourierOperator2D` | 基于物理的前向模型 `A(d) = Phi * FFT(permute(d))` 及其伴随算子 |
| `TheoreticalGradientDescent` | 经典梯度下降（含 Tikhonov / Dirichlet / TV 正则化选项） |
| `LearnedGradientDescent` | 学习梯度下降网络（展开 15 次迭代，每次使用 CNN 更新） |
| `TheoreticalCTNet` | 顶层封装（组合 LGD + 系数映射 + 优化指标计算） |

### `models/radon_transform.py` — 物理引擎与数据生成

| 类名 | 功能 |
|------|------|
| `NumpyPhantomGenerator` | 生成随机椭圆体模（泊松分布，约100个椭圆） |
| `SheppLoganGenerator` | 标准10椭圆 Shepp-Logan 体模 |
| `FourierOperatorCalculator` | 预计算 Phi 对角线和展平排序 |
| `TheoreticalDataGenerator` | 完整数据生成流水线（体模 -> 前向模型 -> 加噪 -> Tikhonov 初估计） |
| `direct_tikhonov_solve` | Tikhonov 闭式解 |

### `models/box_spline.py` — B-样条基函数

实现一阶基数 B-样条 $B_1$ 及其傅里叶变换 $\hat{B}_1$，二维基函数为张量积 $\varphi(x,y) = B_1(x) \cdot B_1(y)$。

### `models/image_generator.py` — 图像合成

`DifferentiableImageGenerator`：将系数网格转换为图像。当网格尺寸与图像尺寸一致（256=256）时，为恒等操作。

---

## 模型架构

### 整体架构

```
TheoreticalCTNet
└── LearnedGradientDescent (展开 n_iter=15 次)
    ├── RadonFourierOperator2D (物理算子 A 和 A^H)
    ├── TheoreticalGradientDescent (计算梯度)
    └── CNN 更新网络 (每次迭代共享权重)
```

### CNN 更新网络结构

采用**膨胀卷积 (Dilated Convolution)** 设计，逐步扩大感受野以捕获多尺度特征：

```
输入: 11 通道 = [当前系数(1) + 数据保真梯度(1) + 正则化梯度(1) + 记忆(8)]
  │
  ├── InstanceNorm2d(11, affine=True)
  ├── Conv2d(11 → 64, 3x3, pad=1)     + InstanceNorm2d(64) + ReLU    [感受野: 3x3]
  ├── Conv2d(64 → 64, 3x3, dilation=2) + InstanceNorm2d(64) + ReLU    [感受野: 7x7]
  ├── Conv2d(64 → 64, 3x3, dilation=4) + InstanceNorm2d(64) + ReLU    [感受野: 15x15]
  ├── Conv2d(64 → 64, 3x3, dilation=8) + InstanceNorm2d(64) + ReLU    [感受野: 31x31]
  └── Conv2d(64 → 9, 3x3, pad=1)
  │
输出: 9 通道 = [更新量(1) + 新记忆(8)]
```

**设计原理**：
- 膨胀卷积用于处理单角度 CT 重建中产生的大尺度条纹伪影
- 递增的膨胀率 (1, 2, 4, 8) 在不增加参数量的情况下捕获从局部到全局的多尺度特征
- InstanceNorm 用于归一化，适合图像重建任务中的样本间差异
- 记忆通道 (8个) 在迭代间传递信息，类似 RNN 的隐藏状态

### 单次迭代流程

```
对于第 k 次迭代 (k = 1, ..., 15):
  1. data_grad = 2 * A^H(A * coeff_k - F_obs)        # 数据保真梯度
  2. reg_grad = lambda_reg * TV_gradient(coeff_k)      # 全变分正则化梯度
  3. cnn_input = concat[coeff_k, data_grad, reg_grad, memory_k]  # 11 通道输入
  4. cnn_output = CNN(cnn_input)                        # 9 通道输出
  5. coeff_{k+1} = coeff_k - step_size * cnn_output[:, 0:1]      # 更新系数
  6. memory_{k+1} = ReLU(cnn_output[:, 1:])             # 更新记忆
```

### 可学习参数

| 参数 | 初始值 | 说明 |
|------|--------|------|
| CNN 权重 | Kaiming 初始化 | 5层卷积网络的权重和偏置 |
| `reg_lambda` | 0.01 | 正则化系数 |
| `step_size` | 0.01 | 梯度更新步长 |
| `loss_params[0]` | 1.0 | 系数域损失权重 |
| `loss_params[1]` | 0.1 | 图像域损失权重 |

---

## 数据流水线

### 数据生成流程

```
体模生成（随机椭圆 或 Shepp-Logan）
    │
    ▼
coeff_true: 256x256 float32 系数矩阵
    │
    ▼
前向算子: F_clean = Phi * FFT(permute(coeff_true))
    │   - permute: 按 beta·k 点积排序重排像素
    │   - FFT: 长度 N=65536 的一维 FFT
    │   - Phi: 由 B1_hat(xi·alpha_x) * B1_hat(xi·alpha_y) 构成的对角矩阵
    │
    ▼
加噪: F_observed = F_clean + delta * N(0, I)    (加性高斯噪声, delta=0.1)
    │
    ▼
Tikhonov 初估计: coeff_init = IFFT(N * Phi^H * F_obs / (N * |Phi|^2 + lambda))
    │
    ▼
保存为 .pt 文件: {coeff_true, F_observed(复数), coeff_initial}
```

### 数据集规模

| 数据集 | 样本数 | 体模类型 | 用途 |
|--------|--------|----------|------|
| `train_dataset.pt` | 20,000 | 随机椭圆 | 训练 |
| `val_dataset.pt` | 2,000 | Shepp-Logan | 验证 |

### 关键维度

| 张量 | 形状 | 类型 |
|------|------|------|
| `coeff_true` | (B, 1, 256, 256) | float32 |
| `F_observed` | (B, 65536) | complex64 |
| `coeff_initial` | (B, 1, 256, 256) | float32 |
| CNN 输入 | (B, 11, 256, 256) | float32 |
| CNN 输出 | (B, 9, 256, 256) | float32 |

---

## 训练流程

### 第一阶段：初始训练 (`train_offline.py`)

| 配置项 | 值 |
|--------|---|
| 目标迭代次数 | 2,500 (优化器更新次数) |
| 物理批大小 | 5 (n_data) |
| 梯度累积步数 | 4 |
| 有效批大小 | 20 |
| 损失函数 | L1(pred, true) + 0.001 * TV(pred) |
| 优化器 | AdamW (lr=1e-3, weight_decay=1e-4) |
| 学习率调度 | LambdaLR 逆时间衰减: lr = lr_0 / (1 + step/500) |
| 梯度裁剪 | 最大范数 1.0 |
| 验证频率 | 每 10 次累积更新 |
| 早停耐心值 | 500 次迭代 |
| 最佳模型选择 | 基于验证损失 (L1 + TV) |

### 第二阶段：续训 (`train_continue.py`)

| 配置项 | 值 |
|--------|---|
| 起始迭代 | 从第一阶段最佳检查点加载 |
| 额外迭代 | 2,500 (目标总迭代 3,500) |
| 梯度累积步数 | 3 |
| 有效批大小 | 15 |
| 损失函数 | L1 + 0.1 * TV (TV 权重提升 100 倍) |
| 其他配置 | 与第一阶段相同，从检查点恢复优化器状态 |

### 训练收敛情况

根据训练日志：
- **第一阶段**：Val RES 从 ~0.36 下降至 ~0.08（2,500 次迭代）
- **第二阶段**：Val RES 稳定在 0.073~0.085 之间，最佳约 0.074

---

## 评估与测试

### 评估指标

**RES (相对误差 / Relative Error)**：

$$\text{RES} = \sqrt{\frac{\sum |d_{\text{pred}} - d_{\text{true}}|^2}{\sum |d_{\text{true}}|^2}}$$

同时计算：
- **Tikhonov 初估计的 RES**（基线，无 CNN 优化）
- **CNN 精化后的 RES**（经过 15 次 LGD 迭代）

### 训练过程中跟踪的指标

| 指标 | 说明 |
|------|------|
| Training Loss | L1 + TV 损失 |
| Validation Loss | 验证集上的 L1 + TV 损失 |
| Train RES | 训练集上的相对误差 |
| Val RES | 验证集上的相对误差 |
| Learning Rate | 当前学习率 |

### 测试脚本 (`models/test.py`)

加载最佳模型，在 Shepp-Logan 体模测试样本上：
1. 生成含噪观测数据
2. 计算 Tikhonov 初始重建
3. 通过 CNN 精化重建
4. 对比展示：真实图像 | Tikhonov 初估计 | CNN 重建
5. 报告各样本的 RES 和均值 RES

---

## 关键参数说明

| 参数 | 值 | 含义 |
|------|---|------|
| `IMAGE_SIZE` | 256 | 图像维度: 256x256 像素，N=65,536 总系数 |
| `beta_vector` | (1, 256) | Radon 投影方向向量，alpha = beta/\|\|beta\|\| 决定单一投影角度 |
| `noise_mode` | "additive" | 噪声模式：加性高斯噪声 |
| `noise_level` (delta) | 0.1 | 傅里叶域测量数据上的加性高斯噪声标准差 |
| `lambda_reg` | 0.01 | Tikhonov 正则化参数（用于初始估计） |
| `n_iter` | 15 | LGD 网络中展开的梯度下降迭代次数 |
| `n_memory_units` | 8 | CNN 在迭代间传递的记忆通道数 |
| `regularizer_type` | "tv" | 正则化类型：全变分 (Total Variation) |
| `n_train` | 3,000 | 目标训练迭代次数 |
| `n_data` | 5 | 物理批大小 |
| `learning_rate` | 1e-3 | 初始学习率 |
| `gradient_clip_value` | 1.0 | 梯度裁剪最大范数 |
| `early_stopping_patience` | 500 | 早停耐心值（验证损失无改善的迭代次数） |
| `image_loss_weight` | 0.1 | 图像域损失权重 |

### 噪声模式支持

| 模式 | 公式 | 说明 |
|------|------|------|
| `additive` | $F_{obs} = F_{clean} + \delta \cdot \mathcal{N}(0, I)$ | 加性高斯噪声（默认） |
| `multiplicative` | $F_{obs} = F_{clean} + \delta \cdot F_{clean} \cdot U[-1,1]$ | 乘性噪声 |
| `snr` | 根据目标 SNR(dB) 计算 $\sigma$ | 旧版本 SNR 模式 |

---

## 使用指南

### 1. 生成数据集

```bash
cd models
python Data_Generator.py
```

生成的文件：
- `data/train_dataset.pt`：20,000 个训练样本（随机椭圆体模）
- `data/val_dataset.pt`：2,000 个验证样本（Shepp-Logan 体模）

### 2. 训练模型

**初始训练：**
```bash
python models/train_offline.py
```

训练产出：
- `models/theoretical_ct_best_model.pth`：验证损失最低的模型
- `models/theoretical_ct_model.pth`：最新模型
- `models/checkpoints/`：每 500/1000 迭代的检查点
- `logs/training.log`：训练日志
- `logs/training_progress.png`：损失曲线

**续训（可选）：**
```bash
python models/train_continue.py
```

从最佳检查点继续训练，进一步优化模型。

### 3. 评估模型

```bash
python models/test.py
```

- 在 Shepp-Logan 体模上评估
- 输出各样本的 RES（Tikhonov 初估计 vs CNN 重建）
- 保存对比图到 `results/` 目录

### 4. 辅助分析

```bash
# 寻找最优 Tikhonov 参数 lambda
python models/tikhonov_find.py

# Tikhonov 重建演示
python models/tikhonov_eval.py

# 对比不同 lambda 选择
python models/compare_best_vs_config_lambda.py

# 验证 Float32 精度限制
python models/theory_verification.py

# 条件数分析
python models/"condition number.py"
```

---

## 实验结果

### 重建效果示例

评估在 Shepp-Logan 体模上进行，噪声配置为加性高斯噪声 (delta=0.1)，Tikhonov 参数 lambda=0.01：

| 方法 | RES (相对误差) |
|------|----------------|
| Tikhonov 初估计 | ~0.96 |
| CNN 重建 (LGD) | ~0.29 |

结果图展示 (保存于 `results/shepp_logan_last.png`)：
- **左**：真实 Shepp-Logan 图像
- **中**：Tikhonov 初估计（明显的条纹伪影和模糊）
- **右**：CNN 重建（伪影显著减少，结构细节更清晰）

### 训练曲线

- 初始训练曲线：`logs/training_progress.png`
- 续训曲线：`logs/continue_training_plot.png`

---

## 辅助工具脚本

| 脚本 | 用途 |
|------|------|
| `models/tikhonov_eval.py` | 扫描不同 lambda 值，展示纯 Tikhonov 重建结果 |
| `models/tikhonov_find.py` | 两阶段搜索（粗搜 + 细搜）寻找最优 Tikhonov 正则化参数 |
| `models/compare_best_vs_config_lambda.py` | 可视化对比 lambda=1e-5、配置 lambda、最优 lambda 的重建效果 |
| `models/theory_verification.py` | 验证 Float32 精度限制对 Tikhonov 恢复的影响 |
| `models/condition number.py` | 分析前向算子的条件数 |

---

## 项目命名含义

**CT_cnn_256_0.07**：
- **CT**：计算断层扫描 (Computed Tomography) 图像重建
- **cnn**：使用卷积神经网络作为学习梯度下降的更新算子
- **256**：图像分辨率为 256x256 像素
- **0.07**：模型在验证集上达到的目标/最佳相对误差 (RES) 约为 0.07

---

## 文档资料

项目包含丰富的数学推导和设计文档（中文）：

- `models/Documentation_tex/`：核心数学公式推导
- `models_change/`：算法理论说明、框架设计笔记
- `mistake/`：调试记录和 Bug 修复过程
- `噪声定义.md`：噪声模型的数学定义
