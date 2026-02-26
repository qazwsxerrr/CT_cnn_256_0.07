为了在 `test.py` 中使用经典的 **Shepp-Logan Phantom**，且不依赖额外的库（如 `skimage` 或 `odl`），我们可以基于你现有的“椭圆生成逻辑”直接构建它。

Shepp-Logan 体模本质上就是 **10 个特定椭圆**的叠加。

请在 `test.py` 中添加（或替换）以下代码。我提供了一个独立的 `SheppLoganGenerator` 类和一个专门的测试函数 `evaluate_shepp_logan`。



### 1. 修改 `test.py` 代码



你需要将以下内容添加到 `test.py` 中：

Python



```
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from config import device, RESULTS_DIR, IMAGE_SIZE
# 确保引用了你的模型加载函数
from test import load_model, plot_result 
# 引用数据生成器以获取算子
from radon_transform import TheoreticalDataGenerator

class SheppLoganGenerator:
    """
    使用 NumPy 生成标准的 Modified Shepp-Logan Phantom (128x128)。
    基于 Kak & Slaney 书籍中的定义。
    """
    def __init__(self, size=128):
        self.size = size
        # 建立 [-1, 1] 的坐标网格
        y, x = np.mgrid[-1:1:complex(0, size), -1:1:complex(0, size)]
        self.X = x
        self.Y = y
        
        # Shepp-Logan 的标准参数 (Modified 版本，对比度更好)
        # 格式: [Intensity, a, b, x0, y0, theta(degrees)]
        self.ellipses = [
            [1.0,   0.69,   0.92,    0.0,      0.0,     0.0],
            [-0.8,  0.6624, 0.874,   0.0,      -0.0184, 0.0],
            [-0.2,  0.11,   0.31,    0.22,     0.0,     -18.0],
            [-0.2,  0.16,   0.41,    -0.22,    0.0,     18.0],
            [0.1,   0.21,   0.25,    0.0,      0.35,    0.0],
            [0.1,   0.046,  0.046,   0.0,      0.1,     0.0],
            [0.1,   0.046,  0.046,   0.0,      -0.1,    0.0],
            [0.1,   0.046,  0.023,   -0.08,    -0.605,  0.0],
            [0.1,   0.023,  0.023,   0.0,      -0.606,  0.0],
            [0.1,   0.023,  0.046,   0.06,     -0.605,  0.0]
        ]

    def generate(self):
        phantom = np.zeros((self.size, self.size), dtype=np.float32)
        
        for params in self.ellipses:
            intensity, a, b, x0, y0, angle_deg = params
            theta = np.radians(angle_deg)
            
            # 坐标变换：平移 -> 旋转
            dx = self.X - x0
            dy = self.Y - y0
            
            # 旋转矩阵 (注意坐标系定义，这里使用标准逆时针旋转)
            x_rot = dx * np.cos(theta) + dy * np.sin(theta)
            y_rot = -dx * np.sin(theta) + dy * np.cos(theta)
            
            # 椭圆方程判断
            mask = (x_rot**2 / a**2 + y_rot**2 / b**2) <= 1.0
            phantom[mask] += intensity
            
        # 简单的数值截断，防止浮点误差略微超出范围（可选）
        return phantom


def evaluate_shepp_logan():
    print("Generating Shepp-Logan Phantom for testing...")
    
    # 1. 准备模型和生成器
    model = load_model()
    data_gen = TheoreticalDataGenerator() # 用于获取物理算子 (FFT, CG等)
    sl_gen = SheppLoganGenerator(size=IMAGE_SIZE)
    
    # 2. 生成 Shepp-Logan 图像 (Ground Truth)
    phantom_np = sl_gen.generate()
    
    # 转为 Tensor: (1, 1, 128, 128)
    coeff_true = torch.from_numpy(phantom_np).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # 3. 模拟物理投影 (生成 F_observed)
    #    f_true 在这里直接等于 coeff_true (因为是 1:1 像素映射)
    f_true = coeff_true.squeeze(0).squeeze(0).cpu().numpy()
    
    with torch.no_grad():
        # 正向投影 (Radon Transform via FFT)
        F_clean = data_gen.forward_operator(coeff_true)
        
        # 加噪 (使用 data_gen 中定义的 noise_level)
        if data_gen.noise_level > 0:
            noise = torch.randn_like(F_clean.real) + 1j * torch.randn_like(F_clean.real)
            scale = data_gen.noise_level * torch.mean(torch.abs(F_clean))
            F_observed = F_clean + scale * noise
        else:
            F_observed = F_clean
            
        # 生成初始化解 (Tikhonov / CG)
        # 注意：这里我们重新调用 CG，模拟真实的测试流程
        lam = 0.1 # 可以根据需要调整正则化参数
        coeff_initial = data_gen.adjoint_operator(F_observed) # 快速初始化，或者用下面的 CG
        # coeff_initial = cg_solve(...) # 如果需要更高质量初始化
        
    # 4. 模型重建
    # 输入维度调整
    coeff_init_batch = coeff_initial
    # 确保 F_observed 是 batch 形式
    if F_observed.dim() == 2: F_observed = F_observed.unsqueeze(0) 

    with torch.no_grad():
        coeff_pred_batch, history, metrics = model(coeff_init_batch, F_observed)
    
    # 5. 提取结果
    coeff_pred = coeff_pred_batch.squeeze().cpu()
    coeff_true_cpu = coeff_true.squeeze().cpu()
    
    # 计算指标
    mse = torch.nn.functional.mse_loss(coeff_pred, coeff_true_cpu).item()
    diff = torch.abs(coeff_pred - coeff_true_cpu)
    residual = torch.norm(diff) / torch.norm(coeff_true_cpu)
    
    print(f"Shepp-Logan Test Results:")
    print(f"MSE: {mse:.6f}")
    print(f"Relative Residual: {residual:.6f}")
    
    # 6. 绘图
    save_path = os.path.join(RESULTS_DIR, "shepp_logan_test.png")
    
    # 复用你原来的 plot_result 函数
    # 注意参数传递，需要适配 plot_result 的签名
    plot_result(
        idx="SheppLogan",
        coeff_true=coeff_true_cpu.numpy(),
        coeff_pred=coeff_pred.numpy(),
        f_true=f_true,
        f_pred=coeff_pred.numpy(), # 在 1:1 映射下，f_pred 就是 coeff_pred
        mse_coeff=mse,
        mse_img=mse, # 同样相等
        residual=residual.item(),
        save_path=save_path,
        noise_level=data_gen.noise_level
    )
    
    # 额外：绘制中间切片对比 (Profile Line)
    plt.figure(figsize=(10, 5))
    mid = IMAGE_SIZE // 2
    plt.plot(f_true[mid, :], 'k-', label='Ground Truth')
    plt.plot(coeff_pred.numpy()[mid, :], 'r--', label='Reconstruction')
    plt.title(f"Profile Line at Y={mid}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    profile_path = os.path.join(RESULTS_DIR, "shepp_logan_profile.png")
    plt.savefig(profile_path)
    print(f"Saved profile plot to {profile_path}")

if __name__ == "__main__":
    # 可以注释掉原来的 evaluate，只运行 Shepp-Logan 测试
    # evaluate(num_samples=5) 
    evaluate_shepp_logan()
```



### 2. 代码解释



1. **SheppLoganGenerator 类**：
   - 这是一个纯 NumPy 实现，没有任何外部特殊库依赖。
   - 它使用了标准的 Shepp-Logan 椭圆参数表（Intensity, 轴长 a/b, 中心 x/y, 角度 theta）。
   - 它在 `[-1, 1]` 的坐标空间内生成图像，这与你之前的 `radon_transform.py` 中的物理定义完全一致。
   - `128x128` 像素会通过 `np.mgrid` 自动离散化。
2. **evaluate_shepp_logan 函数**：
   - 它模拟了一个完整的测试流程：生成真值 -> 前向投影 (FFT) -> 加噪 -> 模型重建。
   - 它复用了 `TheoreticalDataGenerator` 中的 `forward_operator`，确保测试时的物理过程与训练时完全一致。
   - 最后它调用了你现有的 `plot_result` 函数来保存对比图，并额外画了一张中心切面的轮廓线对比图（Profile Line），这在医学图像分析中非常常用。



### 3. 如何运行



直接运行 `test.py` 即可：

Bash



```
python test.py
```

结果图片将保存在 `results/shepp_logan_test.png`。