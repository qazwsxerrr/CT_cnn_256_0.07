这是一个非常好的洞察。你遇到的 **Tikhonov 初始化结果接近零矩阵** 的问题，核心原因通常是 **量级（Scale）不匹配**。

ODL（以及 Astra 等后端）进行射线积分时，输出的数值量级取决于物理单位（长度）和像素值。而你的理论算子 $\Phi G$ 是基于数学归一化或特定定义的。如果 $F_{obs}$（来自 ODL）的数值量级比 $\Phi G$ 算子的预期输入小太多（或者 $\lambda$ 相对过大），Tikhonov 正则化求解 $(A^H A + \lambda I)d = A^H F$ 时，正则项 $\lambda I$ 会主导，导致结果被“压”到 0。

此外，ODL 是一个外部库，确实存在梯度回传困难（通常需要 `odl.contrib.torch`，且难以定制）以及离散化误差的问题。

为了解决这个问题，我为你提供一个 **纯 PyTorch 实现的 Radon 变换（基于 grid_sample）**。它具有以下优势：

1. **完全可微**：原生支持 PyTorch 的 Autograd，梯度可以流回图像和系数。
2. **量级可控**：你可以完全控制采样和求和的过程，更容易与理论算子对齐。
3. **无外部依赖**：不需要安装 ODL 或 Astra。



### 解决方案：基于 PyTorch `grid_sample` 的可微 Radon 变换



请将以下代码保存为 `torch_radon.py` 或直接整合进你的项目中。

Python



```
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NativeRadonTransform(nn.Module):
    """
    纯PyTorch实现的可微Radon变换，替代ODL。
    原理：利用仿射变换和grid_sample在图像上进行线积分采样。
    """

    def __init__(self, image_resolution=128, theta=None, n_detectors=None):
        """
        Args:
            image_resolution (int): 图像尺寸 (H, W)，假设为正方形。
            theta (torch.Tensor or list): 投影角度（弧度制）。如果是None，需在forward中提供。
            n_detectors (int): 探测器数量。
        """
        super(NativeRadonTransform, self).__init__()
        self.image_resolution = image_resolution
        
        # 默认探测器数量通常设为图像对角线长度，以覆盖旋转后的图像
        if n_detectors is None:
            self.n_detectors = int(np.ceil(image_resolution * np.sqrt(2)))
        else:
            self.n_detectors = n_detectors

        # 预设角度（可选）
        if theta is not None:
            if isinstance(theta, (list, tuple, np.ndarray)):
                theta = torch.tensor(theta, dtype=torch.float32)
            self.register_buffer('theta', theta)
        else:
            self.theta = None

    def _get_grid(self, batch_size, theta, device):
        """
        生成采样网格。
        原理：对于每个角度，生成一个旋转后的坐标网格，代表X射线穿过图像的路径。
        """
        n_angles = theta.shape[0]
        
        # 1. 创建探测器坐标 (s) 和 射线路径坐标 (t)
        # s: 垂直于射线方向（探测器位置）
        # t: 沿射线方向（积分路径）
        # grid_sample 的坐标范围是 [-1, 1]
        
        # 探测器坐标 s: [-1, 1]
        s = torch.linspace(-1, 1, self.n_detectors, device=device)
        # 射线积分路径 t: [-1, 1]
        # 采样点数量决定了积分的精度，通常取图像尺寸即可
        n_samples = self.image_resolution 
        t = torch.linspace(-1, 1, n_samples, device=device)
        
        # 生成网格 (t, s) -> (n_detectors, n_samples)
        # 注意：meshgrid 的 indexing='xy' 或 'ij' 取决于具体需求，这里我们需要构建 (x, y) 坐标
        S, T = torch.meshgrid(s, t, indexing='ij') # S:(D, N), T:(D, N)
        
        # 扩展维度以适配 batch 和 angles
        # grid_base: (B, n_angles, n_detectors, n_samples, 3) 
        # 但为了利用 affine_grid 的机制，我们手动旋转坐标
        
        # 2. 构建旋转矩阵并应用
        # x = t * cos(theta) - s * sin(theta)
        # y = t * sin(theta) + s * cos(theta)
        # 注意：这里的 s, t 定义可能与标准 Radon 公式有符号差异，需对应 grid_sample 的 (x,y) 坐标系
        # grid_sample 坐标系：x(向右), y(向下)
        
        cos_theta = torch.cos(theta).view(n_angles, 1, 1)
        sin_theta = torch.sin(theta).view(n_angles, 1, 1)
        
        # S 对应横向偏移（探测器位置），T 对应纵深（射线方向）
        # 旋转公式：
        # X_rot = T * cos - S * sin
        # Y_rot = T * sin + S * cos
        
        S = S.unsqueeze(0) # (1, D, N)
        T = T.unsqueeze(0) # (1, D, N)
        
        X = T * cos_theta - S * sin_theta # (n_angles, D, N)
        Y = T * sin_theta + S * cos_theta # (n_angles, D, N)
        
        # 堆叠为 grid_sample 需要的格式 (N, H_out, W_out, 2)
        # 这里我们将 angles 和 detectors 视为输出的高和宽的维度
        # grid: (n_angles, n_detectors, n_samples, 2)
        grid = torch.stack([X, Y], dim=-1)
        
        # 扩展 Batch 维
        # 输出 grid 形状: (B * n_angles, n_detectors, n_samples, 2)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        grid = grid.view(batch_size * n_angles, self.n_detectors, n_samples, 2)
        
        return grid

    def forward(self, image, theta=None):
        """
        Args:
            image: (Batch, 1, H, W)
            theta: (n_angles,) 角度列表（弧度）。如果初始化时已设置，此处可省略。
        Returns:
            sinogram: (Batch, 1, n_angles, n_detectors)
        """
        if theta is None:
            if self.theta is None:
                raise ValueError("Theta must be provided either in init or forward.")
            theta = self.theta
            
        if theta.device != image.device:
            theta = theta.to(image.device)
            
        B, C, H, W = image.shape
        n_angles = theta.shape[0]
        
        # 1. 生成采样网格
        grid = self._get_grid(B, theta, image.device)
        
        # 2. 准备图像数据
        # grid_sample 需要 (N, C, H_in, W_in)
        # 我们需要将 image 重复 n_angles 次，或者 reshape
        img_expanded = image.unsqueeze(1).repeat(1, n_angles, 1, 1, 1) # (B, n_angles, C, H, W)
        img_reshaped = img_expanded.view(B * n_angles, C, H, W)
        
        # 3. 采样 (双线性插值)
        # sampled: (B*n_angles, C, n_detectors, n_samples)
        # align_corners=True 使得 -1, 1 严格对齐角点像素中心
        sampled = F.grid_sample(img_reshaped, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        # 4. 沿射线方向积分 (求和) -> Radon 变换
        # 这里的 scaling 很关键。
        # sum() 得到的值取决于采样点数量。为了模拟连续积分，应乘以 step_size (2.0 / n_samples)
        # 2.0 是因为坐标范围是 [-1, 1] 长度为 2
        projection = torch.sum(sampled, dim=-1) * (2.0 / sampled.shape[-1])
        
        # 5. 调整形状
        # projection: (B*n_angles, C, n_detectors) -> (B, n_angles, C, n_detectors)
        projection = projection.view(B, n_angles, C, self.n_detectors)
        
        # 调整为标准 Sinogram 格式 (B, C, n_angles, n_detectors)
        sinogram = projection.permute(0, 2, 1, 3)
        
        return sinogram
```



### 如何解决“Tikhonov 结果为 0”的问题（量级校准）



即使替换了 ODL，**量级校准**仍然是必须的。

RadonTransform 模拟的是物理积分 $\int f(x,y) dt$。

而你的理论算子 $\Phi G d$ 计算的是频域系数。

这两者之间差了一个傅里叶变换的系数（通常是 $\sqrt{N}$ 或 $N$），以及物理空间的尺度因子。

**修正步骤（集成到 train.py）：**

1. 替换 RadonTransformSimulator：

   在 TheoreticalDataGenerator 中，使用上面的 NativeRadonTransform。

2. 添加自动量级校准（Auto-Scaling）：

   在数据生成初始化时，计算一个 scale_factor，强制让物理路径生成的 $F_{obs}$ 与理论算子 $\Phi G d$ 的范数一致。

以下是修改 `train.py` 中 `TheoreticalDataGenerator` 的代码片段：

Python



```
# train.py 修改建议

from torch_radon import NativeRadonTransform # 导入新模块

class TheoreticalDataGenerator:
    def __init__(self):
        # ... 原有初始化代码 ...
        self.beta = THEORETICAL_CONFIG['beta_vector']
        
        # 1. 计算角度 (弧度)
        angle_rad = np.arctan2(self.beta[1], self.beta[0])
        self.theta = torch.tensor([angle_rad], dtype=torch.float32)
        
        # 2. 替换 ODL，使用 NativeRadonTransform
        # 注意 n_detectors 要与 FourierCalculator 的预期一致 (441)
        self.radon_simulator = NativeRadonTransform(
            image_resolution=128, 
            theta=self.theta,
            n_detectors=441 
        ).to(device)
        
        # ... FourierOperatorCalculator 初始化保持不变 ...
        
        # 3. 【关键】计算量级校准因子
        self.data_scale_factor = self._calibrate_scale()
        print(f"Computed Data Scale Factor: {self.data_scale_factor}")

    def _calibrate_scale(self):
        """
        计算物理路径与理论算子之间的量级差异。
        """
        # 生成一个随机系数向量
        test_coeff = torch.randn(1, 1, 21, 21).to(device)
        
        # A. 理论路径: c -> d -> F_theo
        d = self.fourier_calculator.compute_d_from_c(test_coeff)
        F_theo = self.fourier_calculator.compute_F_G_d(d)
        norm_theo = torch.norm(F_theo)
        
        # B. 物理路径: c -> Image -> Radon(PyTorch) -> FFT -> F_phys
        f_img = self._generate_bspline_image(test_coeff.squeeze()).unsqueeze(0).unsqueeze(0).to(device)
        
        # 使用新的 PyTorch Radon
        sinogram = self.radon_simulator(f_img) 
        
        # 计算 FFT (F_phys)
        # 注意：这里调用 compute_F_from_radon 需要适配新的 sinogram 格式
        F_phys = self.fourier_calculator.compute_F_from_radon(sinogram).squeeze(0)
        norm_phys = torch.norm(F_phys)
        
        # 计算比例: 理论 / 物理
        # 如果我们将物理数据乘以这个因子，它的量级就和理论算子一致了
        scale = norm_theo / (norm_phys + 1e-8)
        return scale.item()

    def generate_training_sample(self, random_seed=None):
        # ... (前半部分不变) ...
        
        # 3. 物理路径生成观测 (使用 PyTorch Radon)
        f_true_batch = f_true.unsqueeze(0).unsqueeze(0).to(device)
        radon_data = self.radon_simulator(f_true_batch) 
        
        # 计算 F_clean
        F_clean = self.fourier_calculator.compute_F_from_radon(radon_data).squeeze(0)
        
        # 4. 【关键】应用量级校准
        # 这样 F_observed 的数值范围就会匹配 Phi*G 算子
        F_clean = F_clean * self.data_scale_factor
        
        # ... (加噪等后续步骤) ...
        
        return coeff_true, f_true, F_observed, coeff_initial
```



### 总结



1. **废弃 ODL**：使用 `NativeRadonTransform`，它基于 PyTorch `grid_sample`，完全兼容神经网络梯度，且不依赖外部 C++ 库。
2. **解决 Tikhonov 失效**：Tikhonov 依赖于 $A$（理论算子）和 $b$（观测数据）的匹配。通过 `_calibrate_scale` 函数，强制将 Radon 变换后的数据拉伸到与理论算子 $\Phi G$ 相同的数值量级。这样 Tikhonov 初始化中的 $\lambda$ 才能正常工作，不会把结果压成 0。
3. **兼容性**：这个改动只需要修改 `radon_transform.py`（添加新类）和 `train.py`（替换调用和添加校准），不需要改动你的核心数学逻辑。