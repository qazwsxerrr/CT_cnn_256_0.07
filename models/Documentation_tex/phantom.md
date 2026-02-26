这是一个非常实用且必要的修改。为了摆脱对 ODL 库的依赖，我们需要手动实现椭圆的几何生成逻辑。

本质上，ODL 的 `ellipsoid_phantom` 函数只是在网格上计算点是否在椭圆方程内。我们可以使用 **NumPy** 或 **PyTorch** 轻松复现这一逻辑。

以下是具体的修改方案。我将提供一个独立的类 `PhantomGenerator`，它复刻了你提供的 `phamtom_plot.py` 中的分布逻辑，但完全基于 NumPy 实现。



### 1. 新增 `phantom_generator.py` (或者直接放在项目中)



可以创建一个新文件 `phantom_generator.py`。

核心逻辑：

椭圆的一般方程（带旋转）判定点 $(x, y)$ 是否在椭圆内：

$$\left( \frac{(x-x_c)\cos\theta + (y-y_c)\sin\theta}{a} \right)^2 + \left( \frac{-(x-x_c)\sin\theta + (y-y_c)\cos\theta}{b} \right)^2 \le 1$$

Python



```
import numpy as np
import torch

class NumpyPhantomGenerator:
    """
    不依赖 ODL 库的随机椭圆体模生成器。
    完全复刻 phamtom_plot.py 的随机参数分布逻辑。
    """
    def __init__(self, size=128):
        self.size = size
        # 创建坐标网格 [-1, 1] x [-1, 1]
        # 注意：对应 ODL 的 extent，这里归一化到 1 便于计算
        y, x = np.mgrid[-1:1:complex(0, size), -1:1:complex(0, size)]
        self.X = x
        self.Y = y

    def _random_ellipse_params(self):
        """
        生成 1 个随机椭圆的 6 个参数。
        逻辑源自 phamtom_plot.py
        返回: (intensity, a, b, x0, y0, theta)
        """
        intensity = (np.random.rand() - 0.3) * np.random.exponential(0.3)
        a = np.random.exponential(0.2)  # 长轴 (注意原代码是 exponential() * 0.2，通常 exponential 默认 scale=1)
        b = np.random.exponential(0.2)  # 短轴
        x0 = np.random.rand() - 0.5     # 中心 x
        y0 = np.random.rand() - 0.5     # 中心 y
        theta = np.random.rand() * 2 * np.pi # 旋转角度
        return intensity, a, b, x0, y0, theta

    def generate(self, n_ellipses=None):
        """生成叠加了多个随机椭圆的图像"""
        if n_ellipses is None:
            n_ellipses = np.random.poisson(7) # 泊松分布生成数量
            
        phantom = np.zeros((self.size, self.size), dtype=np.float32)

        for _ in range(n_ellipses):
            intensity, a, b, x0, y0, theta = self._random_ellipse_params()
            
            # 防止过小的轴导致除零错误
            a = max(a, 0.01)
            b = max(b, 0.01)

            # 坐标旋转与平移
            # dx, dy 是相对于椭圆中心的坐标
            dx = self.X - x0
            dy = self.Y - y0
            
            # 旋转变换
            x_rot = dx * np.cos(theta) + dy * np.sin(theta)
            y_rot = -dx * np.sin(theta) + dy * np.cos(theta)

            # 椭圆方程判定: (x'/a)^2 + (y'/b)^2 <= 1
            mask = (x_rot**2 / a**2 + y_rot**2 / b**2) <= 1.0
            
            # 叠加强度
            phantom[mask] += intensity

        return phantom
```

------



### 2. 修改 `radon_transform.py`



我们需要将 `TheoreticalDataGenerator` 中的噪声生成逻辑替换为上面的 `NumpyPhantomGenerator`。

请修改 `radon_transform.py` 中的 `TheoreticalDataGenerator` 类：

1. **导入**：引入上面的类（或者如果写在同一个文件里直接使用）。
2. **初始化**：实例化生成器。
3. **生成样本**：替换 `torch.rand`。

Python



```
# [修改 radon_transform.py]

import torch
import numpy as np
from config import device, DATA_CONFIG, THEORETICAL_CONFIG, IMAGE_SIZE
# 假设你把上面的类放在了同名文件 phantom_generator.py 中，或者直接贴在这个文件里
# from phantom_generator import NumpyPhantomGenerator 

# ... (FourierOperatorCalculator 类保持不变) ...

class TheoreticalDataGenerator:
    def __init__(self):
        self.img_size = IMAGE_SIZE
        self.N = self.img_size * self.img_size
        self.noise_level = DATA_CONFIG['noise_level']
        
        # --- 新增：初始化 Phantom 生成器 ---
        self.phantom_gen = NumpyPhantomGenerator(size=self.img_size) 
        # --------------------------------

        self.image_gen = DifferentiableImageGenerator(image_size=self.img_size).to(device)
        self.fourier_calculator = FourierOperatorCalculator(
            beta=THEORETICAL_CONFIG['beta_vector'],
            n_coefficients=self.N,
            m=2,
        )
        self.Phi = self.fourier_calculator.Phi_diagonal.to(device=device, dtype=torch.complex64)
        self.flatten_order = self.fourier_calculator.flatten_order.to(device)

    # ... (forward_operator 和 adjoint_operator 保持不变) ...

    def generate_training_sample(self, random_seed=None, lambda_reg: float = None):
        """
        生成单个样本：
        1. 使用 NumpyPhantomGenerator 生成 128x128 的椭圆叠加图。
        2. 转为 Tensor 并作为 coeff_true。
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        # --- 修改开始：不再使用 torch.rand ---
        # 1. 生成 Numpy 图像
        phantom_np = self.phantom_gen.generate()
        
        # 2. 转为 Tensor (1, 1, 128, 128)
        coeff_true = torch.from_numpy(phantom_np).float().unsqueeze(0).unsqueeze(0).to(device)
        # --- 修改结束 ---

        # 3. 这里的 f_true 也是 coeff_true (因为是1:1像素映射)
        # 注意：如果你的 DifferentiableImageGenerator 有特殊逻辑，这里保留调用
        # 但在 128x128 模式下，f_true 应该就等于 coeff_true
        f_true = self.image_gen(coeff_true).squeeze(0)

        # 4. 前向物理投影 (FFT)
        with torch.no_grad():
            F_clean = self.forward_operator(coeff_true)
            if self.noise_level > 0:
                # 产生复数高斯噪声
                noise = torch.randn_like(F_clean.real) + 1j * torch.randn_like(F_clean.real)
                scale = self.noise_level * torch.mean(torch.abs(F_clean))
                F_observed = F_clean + scale * noise
            else:
                F_observed = F_clean

        # 5. Tikhonov 初始化 (CG)
        lam = lambda_reg if lambda_reg is not None else DATA_CONFIG.get("lambda_reg", 0.1)
        coeff_initial = cg_solve(
            self.forward_operator,
            self.adjoint_operator,
            F_observed,
            n_iter=10,
            lambda_reg=lam,
        ).real

        return (
            coeff_true.squeeze(0).squeeze(0),
            f_true.squeeze(0),
            F_observed.squeeze(0),
            coeff_initial.squeeze(0).squeeze(0),
        )
```



### 3. 主要修改点解释



1. **数学等价性**：ODL 的 `ellipsoid_phantom` 实际上就是计算像素坐标到椭圆中心的加权距离。我们在 `NumpyPhantomGenerator` 中用简单的解析几何公式 $(x'/a)^2 + (y'/b)^2 \le 1$ 实现了同样的效果。
2. **坐标系**：
   - ODL 默认中心是 $(0,0)$。
   - 我们在 `np.mgrid` 中创建了范围 $[-1, 1]$ 的网格，这对应了 `phamtom_plot.py` 中 `extent=[-1, 1, -1, 1]` 的物理空间。
   - 生成的随机参数 $x_0, y_0$ 在 $[-0.5, 0.5]$ 之间，确保椭圆大部分位于图像中心区域，不会经常跑出边界。
3. **数据流**：现在生成的 `coeff_true` 不再是无意义的噪声，而是具有几何结构的 Shepp-Logan 风格的图像。这对于神经网络学习边缘检测和去噪非常有帮助。

应用此修改后，你的网络将学习重建椭圆体模，而不是重建纯随机噪声。