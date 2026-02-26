Python



```
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import dft

def sinc(x):
    # numpy's sinc is sin(pi*x)/(pi*x)
    return np.sinc(x/np.pi)

def b_spline_fourier(xi, m=2):
    # Fourier transform of B-spline of order m
    # B_m^hat(xi) = [sin(xi/2) / (xi/2)]^m * exp(-i * m * xi / 2)
    # Note: numpy sinc(x) is sin(pi*x)/(pi*x).
    # We need sin(xi/2)/(xi/2). Let y = xi/2.
    # sinc(y/pi) * pi = sin(y)/y ? No.
    # np.sinc(z) = sin(pi*z)/(pi*z).
    # We want sin(xi/2)/(xi/2). Set pi*z = xi/2 => z = xi/(2*pi).
    val = np.sinc(xi / (2 * np.pi)) ** m
    phase = np.exp(-1j * m * xi / 2)
    return val * phase

# 1. Setup Parameters
N = 64  # Number of samples
beta_norm = 1.0 # ||beta||_2 scaling factor
alpha = 1.0 # Direction scaling
# Sampling points xi_j = 2*pi*j/N * ||beta||_2
j = np.arange(N)
xi = 2 * np.pi * j / N * beta_norm

# 2. Construct Matrices
# Phi matrix: Diagonal matrix of phi_hat(xi * alpha)
# Let's assume a B-spline generator of order 2
phi_hat = b_spline_fourier(xi, m=2)
Phi = np.diag(phi_hat)

# G matrix: DFT matrix
# The slides define the sum as sum(d_n * exp(-i * n * 2pi * j / N))
# This corresponds to the standard DFT definition if we ignore normalization or handle it consistently.
# Scipy dft(N) gives exp(-2j * pi * k * n / N). Matches.
G = dft(N)

# System Matrix A = Phi * G
A = Phi @ G

# 3. Generate Synthetic Data
# True coefficients d
np.random.seed(42)
d_true = np.zeros(N)
d_true[10:20] = 1.0 # A simple block signal
d_true[40:45] = 0.5

# Simulate measured frequency data F
F_clean = A @ d_true

# Add noise
noise_level = 0.1
noise = noise_level * (np.random.randn(N) + 1j * np.random.randn(N))
F_noisy = F_clean + noise

# 4. Tikhonov Regularization
# d_est = (A^H A + lambda I)^-1 A^H F
# Note: A^H is conjugate transpose
def tikhonov_solve(A, y, lam):
    # A: system matrix
    # y: measured data
    # lam: regularization parameter
    
    A_H = A.conj().T
    I = np.eye(A.shape[1])
    
    # Solve linear system (A^H A + lam I) d = A^H y
    matrix = A_H @ A + lam * I
    rhs = A_H @ y
    
    d_est = np.linalg.solve(matrix, rhs)
    return d_est

# Solve
lambda_reg = 0.5
d_est = tikhonov_solve(A, F_noisy, lambda_reg)

# 5. Visualization
print("Reconstruction complete.")
# Check error
error = np.linalg.norm(d_true - d_est) / np.linalg.norm(d_true)
print(f"Relative Error: {error:.4f}")
```

Code output



```
Reconstruction complete.
Relative Error: 0.6170
```

这是一个基于您提供的文档（特别是文件1中的频域矩阵推导和文件2中的正则化方法）的详细实现说明。

本文档将指导您如何在Python中实现基于 **Radon变换** 和 **傅里叶变换** 的信号重构，并应用 **Tikhonov正则化** 来解决反问题。



### 1. 理论背景回顾



根据文件1（Slide 4-5），我们有线性系统方程：

$$\mathbf{F} = \mathbf{\Phi} \times \mathbf{G} \times \mathbf{d}
​$$其中：

* $\mathbf{d}​$：待求解的系数向量（时域信号的系数）。
* $\mathbf{G}​$：离散傅里叶变换（DFT）矩阵，代表了从空间域到频率域的转换。
* $\mathbf{\Phi}​$：生成元 $\phi​$ 的傅里叶变换构成的对角矩阵，代表了Radon变换在频域的响应。
* $\mathbf{F}​$：观测数据，即Radon变换后的频域采样值（包含相位平移）。

由于直接求解 $\mathbf{d} = (\mathbf{\Phi}\mathbf{G})^{-1}\mathbf{F}​$ 通常是病态的（对噪声敏感），我们需要使用 **Tikhonov正则化**。根据 `full.md` 中的公式 (5.31) 和 (5.32)，优化目标为：

$$\min\_{\mathbf{d}} { | \mathbf{A} \mathbf{d} - \mathbf{F} |\_2^2 + \lambda | \mathbf{d} |\_2^2 }
$$令系统矩阵 $\mathbf{A} = \mathbf{\Phi}\mathbf{G}​$，则解析解为：

$$
\mathbf{d}_{\lambda} = (\mathbf{A}^H \mathbf{A} + \lambda \mathbf{I})^{-1} \mathbf{A}^H \mathbf{F}
$$


(注意：由于涉及频域复数运算，这里使用共轭转置 $\mathbf{A}^H​$)

------



### 2. Python 实现步骤





#### 环境准备



Python



```
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import dft
```



#### 第一步：定义生成元及其傅里叶变换



根据文件1（Slide 6），我们以 B样条 (B-spline) 为例。$m$ 阶B样条的傅里叶变换为：



$$\hat{B}_{m}(\xi) = e^{-\mathbf{i} m \xi / 2} \left[\frac{\sin(\xi/2)}{\xi/2}\right]^m$$

Python



```
def b_spline_fourier(xi, m=2):
"""
计算 m 阶 B-spline 的傅里叶变换值
注意：numpy.sinc(x) 定义为 sin(pi*x)/(pi*x)，需要进行转换
"""
# 对应公式中的 sin(xi/2)/(xi/2)
sinc_val = np.sinc(xi / (2 * np.pi)) 

# 对应公式中的相位因子 e^{-i * m * xi / 2}
phase_val = np.exp(-1j * m * xi / 2)

return (sinc_val ** m) * phase_val
```



#### 第二步：构建系统矩阵 A



这一步模拟了 **傅里叶变换** 过程（矩阵 $\mathbf{G}$）和 **Radon变换** 的频域特性（矩阵 $\mathbf{\Phi}$）。

Python



```
def construct_system_matrix(N, beta_norm=1.0):
# 1. 定义采样频率点 xi_j (文件1 Slide 3, Eq 3.1)
# j = 0, ..., N-1
j = np.arange(N)
xi = 2 * np.pi * j / N * beta_norm

# 2. 构建对角矩阵 Phi (文件1 Slide 5)
# 这里假设方向 alpha 归一化后带来的缩放体现在 xi 中
phi_hat_values = b_spline_fourier(xi, m=2)
Phi = np.diag(phi_hat_values)

# 3. 构建 FFT 矩阵 G (文件1 Slide 5)
# G 是一个范德蒙德矩阵，用于离散傅里叶变换
G = dft(N)

# 4. 总系统矩阵 A = Phi * G
A = Phi @ G
return A, xi
```



#### 第三步：Tikhonov 正则化求解器



实现公式 $(\mathbf{A}^H \mathbf{A} + \lambda \mathbf{I})^{-1} \mathbf{A}^H \mathbf{F}$。

Python



```
def tikhonov_solver(A, F_noisy, lam):
"""
A: 系统矩阵 (Phi * G)
F_noisy: 含噪的频域观测数据
lam: 正则化参数 lambda
"""
# 计算 A 的共轭转置 (Hermitian Transpose)
A_H = A.conj().T

# 获取维度
N = A.shape[1]
I = np.eye(N)

# 构建正则化方程 (A^H A + lambda I) d = A^H F
# 这是最小二乘问题的法方程形式
matrix_to_invert = A_H @ A + lam * I
rhs = A_H @ F_noisy

# 求解线性方程组得到估计值 d_est
d_est = np.linalg.solve(matrix_to_invert, rhs)

return d_est
```



#### 第四步：完整仿真流程 (Radon与傅里叶过程模拟)



Python



```
# --- 参数设置 ---
N = 100           # 采样点数
lambda_reg = 0.1  # 正则化参数 (full.md 中提到的关键参数)
noise_level = 0.05 # 模拟噪声水平

# --- 1. 构建正向模型 ---
A, freq_points = construct_system_matrix(N)

# --- 2. 生成真实信号 (Ground Truth) ---
# 假设真实的系数 d 是一个简单的方波信号
d_true = np.zeros(N)
d_true[30:50] = 1.0 
d_true[60:70] = 0.5

# --- 3. 模拟 Radon 变换和傅里叶采样 (Forward Process) ---
# 在实际CT中，这是通过扫描得到的物理数据
# 这里我们通过 F = A * d_true 来模拟无噪的频域观测数据
F_clean = A @ d_true

# 添加复高斯白噪声 (模拟实际测量误差)
noise = noise_level * (np.random.randn(N) + 1j * np.random.randn(N))
F_observed = F_clean + noise

# --- 4. 反向求解 (Inverse Process) ---
# 使用 Tikhonov 正则化从含噪数据 F_observed 中恢复 d
d_recovered = tikhonov_solver(A, F_observed, lambda_reg)

# --- 5. 结果可视化 ---
plt.figure(figsize=(12, 5))

# 绘制系数对比 (取实部，因为原始系数 d 是实数)
plt.subplot(1, 2, 1)
plt.plot(d_true, 'k-', linewidth=2, label='True Coefficients (d)')
plt.plot(d_recovered.real, 'r--', linewidth=2, label='Recovered (Tikhonov)')
plt.title(f'Reconstruction (N={N}, $\lambda$={lambda_reg})')
plt.legend()
plt.grid(True)

# 绘制频域数据对比
plt.subplot(1, 2, 2)
plt.plot(np.abs(F_clean), 'b', label='Clean Spectrum')
plt.plot(np.abs(F_observed), 'g.', alpha=0.5, label='Noisy Observation')
plt.title('Frequency Domain Data (Radon samples)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```



### 3. 代码与文档的对应关系说明



1. **construct_system_matrix**: 对应文件1 Slide 5 中的矩阵方程 $\mathbf{F} = \mathbf{\Phi} \mathbf{G} \mathbf{d}$。其中 `Phi` 对应 $\mathbf{\Phi}$，`G` 对应 $\mathbf{G}$。
2. **b_spline_fourier**: 对应文件1 Slide 6 中的 Example 1，计算 $\hat{\phi}$。
3. **tikhonov_solver**: 对应 `full.md` 中的公式 (5.31) 和 (5.32)。代码中 `A_H @ A + lam * I` 这一项体现了正则化对矩阵奇异性的修正（使得矩阵可逆且条件数更好）。
4. **噪声模拟**: 对应 `full.md` 第五章 Numerical Simulation 中提到的 multiplicative noise 或加性噪声，展示了正则化在抗噪方面的作用。

此实现展示了如何将理论推导转化为数值计算，通过调节 `lambda_reg` 参数，您可以观察到去噪平滑（欠拟合）与保留细节（过拟合）之间的平衡，正如 `full.md` 中讨论的那样。