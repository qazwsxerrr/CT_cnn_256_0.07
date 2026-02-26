如果 $\mathbf{d}$ 被显式地视为一个二维矩阵（对应于 `full.md` 中定义在二维网格 $E^+$ 上的系数 $c_{\boldsymbol{k}}$），处理数据保真项梯度和正则化项梯度的核心原则是：**物理计算在 1D（为了匹配 FFT 矩阵定义），特征提取在 2D（为了利用空间相关性），两者通过 Reshape 操作无缝衔接。**

在 `full.md` 的公式 (3.3) 中，1D 向量 $d_n$ 与 2D 网格点 $\boldsymbol{k} \in E^+$ 是一一对应的。因此，这种维度转换是有严格数学依据的。

以下是详细的处理步骤和代码实现方式。

------



### 1. 数据保真项梯度 (Data Fidelity Gradient)



数据保真项通常定义为 $f_{data}(\mathbf{d}) = \frac{1}{2} \| \mathcal{A}(\mathbf{d}) - \mathbf{y} \|_2^2$。

其中 $\mathcal{A}$ 是物理前向算子（Radon + Fourier）。由于 $\mathcal{A}$ 的核心是 FFT 矩阵 $\mathbf{G}$ 和对角矩阵 $\mathbf{\Phi}$，它们数学上作用于向量。

**处理流程：**

1. **输入 (Input)**: 当前估计值是一个二维矩阵 $\mathbf{D}^{(k)}$，尺寸为 $(H, W)$。
2. **拉直 (Flatten)**: 将 $\mathbf{D}^{(k)}$ 变换为 1D 向量 $\mathbf{d}$，尺寸为 $N = H \times W$。这是为了匹配公式 $\mathbf{F} = \mathbf{\Phi} \mathbf{G} \mathbf{d}$。
3. **物理算子 (Physics Op)**:
   - 前向：计算 $\mathbf{A}\mathbf{d}$（FFT $\rightarrow$ 乘 $\mathbf{\Phi}$）。
   - 残差：计算 $\mathbf{r} = \mathbf{A}\mathbf{d} - \mathbf{y}_{obs}$。
   - 伴随：计算 $\mathbf{g}_{1D} = \mathbf{A}^H \mathbf{r}$（乘 $\mathbf{\Phi}^*$ $\rightarrow$ IFFT $\rightarrow$ 取实部）。
4. **重塑 (Reshape)**: 将得到的 1D 梯度向量 $\mathbf{g}_{1D}$ 重新变回 2D 矩阵 $(H, W)$。
5. **输出**: 得到 2D 形式的数据梯度 $\nabla_{data}$，可以直接作为 CNN 的一个图像通道输入。



### 2. 正则化项梯度 (Regularization Gradient)



在深度学习优化（LGD）框架中，我们通常不需要显式计算复杂的正则化梯度（如 TV 范数的次梯度），而是将“当前估计值”本身作为正则化梯度的**代理信息**输入网络，让 CNN 学习具体的正则化方向。

**处理流程：**

- **假设**: 正则项是简单的 Tikhonov 正则 $\frac{\lambda}{2}\|\mathbf{D}\|_F^2$。
- **计算**: 梯度就是 $\lambda \mathbf{D}$。
- **2D 处理**: 因为 $\mathbf{D}$ 已经是 2D 矩阵，这步不需要任何变换，直接使用 $\mathbf{D}$ 本身（或乘以权重 $\lambda$）作为特征图。

------



### 3. 详细代码实现 (封装为自动处理维度的算子)



我们可以编写一个 `RadonFourierOperator2D` 类，它对外接收 2D 矩阵，内部自动处理 Flatten 和 Reshape，从而对用户屏蔽 1D 物理细节。

Python



```
import torch
import torch.nn as nn
import numpy as np

class RadonFourierOperator2D(nn.Module):
    def __init__(self, height, width, m=2):
        """
        height, width: d 矩阵的 2D 尺寸 (对应 E+ 网格大小)
        m: B-spline 阶数
        """
        super().__init__()
        self.height = height
        self.width = width
        self.N = height * width  # 总采样点数
        
        # === 预计算物理参数 (与之前相同，保持 1D 定义) ===
        # 依据 full.md 公式 (3.1) 和 (3.4)
        k = torch.fft.fftfreq(self.N, d=1.0/self.N)
        w = 2 * np.pi * k
        
        w_safe = w.clone()
        w_safe[w_safe == 0] = 1e-9
        magnitude = (torch.sin(w_safe / 2) / (w_safe / 2)) ** m
        magnitude[w == 0] = 1.0
        phase = torch.exp(-1j * m * w / 2)
        phi_diag = magnitude * phase
        
        self.register_buffer('phi_diag', phi_diag)

    def forward(self, x_2d):
        """
        计算前向投影 A(x)
        输入 x_2d: (Batch, 1, Height, Width) -> 2D 矩阵
        输出 y:    (Batch, 1, N) -> 频域观测数据 (通常保持 1D 存储)
        """
        batch_size = x_2d.shape[0]
        
        # 1. 【拉直】 Flatten: (B, 1, H, W) -> (B, 1, N)
        x_1d = x_2d.view(batch_size, 1, self.N)
        
        # 2. 物理计算 (1D FFT)
        x_freq = torch.fft.fft(x_1d, dim=-1)
        y = self.phi_diag * x_freq
        
        return y

    def adjoint(self, residual_1d):
        """
        计算数据梯度 A^H(r)
        输入 residual_1d: (Batch, 1, N) -> 频域残差
        输出 grad_2d:     (Batch, 1, Height, Width) -> 2D 梯度图
        """
        batch_size = residual_1d.shape[0]
        
        # 1. 物理计算 (1D IFFT)
        x_freq = torch.conj(self.phi_diag) * residual_1d
        x_complex = torch.fft.ifft(x_freq, dim=-1)
        grad_1d = x_complex.real * self.N  # 取实部
        
        # 2. 【重塑】 Reshape: (B, 1, N) -> (B, 1, H, W)
        grad_2d = grad_1d.view(batch_size, 1, self.height, self.width)
        
        return grad_2d

# ==========================================
# 在 LGD 迭代中的使用方式
# ==========================================
class LearnedGD_2D_Matrix_Mode(nn.Module):
    def __init__(self, height, width, n_iter=10):
        super().__init__()
        self.n_iter = n_iter
        self.op = RadonFourierOperator2D(height, width)
        # CNN 输入通道为 3: x(1) + grad_data(1) + grad_reg(1)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )
        self.reg_lambda = nn.Parameter(torch.tensor(0.01))

    def forward(self, y_obs):
        # y_obs 是频域数据 (Batch, 1, N)
        batch_size = y_obs.shape[0]
        height, width = self.op.height, self.op.width
        
        # 初始化 d 为 2D 零矩阵 (Batch, 1, H, W)
        d = torch.zeros(batch_size, 1, height, width, device=y_obs.device)
        
        for i in range(self.n_iter):
            # --- 1. 计算数据保真梯度 (自动处理 2D<->1D) ---
            y_pred = self.op(d)             # d(2D) -> y_pred(1D)
            residual = y_pred - y_obs       # 1D 减法
            grad_data = self.op.adjoint(residual) # residual(1D) -> grad(2D)
            
            # --- 2. 计算正则化梯度 (直接在 2D 上操作) ---
            # LGD 中直接使用 d 本身作为正则项特征
            grad_reg = self.reg_lambda * d  # 保持 2D
            
            # --- 3. CNN 更新 (全 2D 卷积) ---
            # 拼接: (B, 3, H, W)
            cnn_input = torch.cat([d, grad_data, grad_reg], dim=1)
            update = self.cnn(cnn_input)
            
            d = d - update
            
        return d
```



### 总结



当 $d$ 是二维矩阵时：

1. **数据梯度 $\nabla_{data}$**：必须经过 **Reshape (1D $\to$ 2D)**。这是因为物理模型 $\mathbf{\Phi}$ 定义在频率域（1D序列），而我们需要在空间域（2D网格）更新参数。
2. **正则梯度 $\nabla_{reg}$**：直接使用 **2D 矩阵本身**。
3. **网络输入**：将上述两者在 Channel 维度拼接，形成 $(Batch, C, H, W)$ 张量，直接利用 PyTorch 强大的 `Conv2d` 处理。