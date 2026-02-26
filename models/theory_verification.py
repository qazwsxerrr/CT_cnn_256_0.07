"""
Float32 Verification Script
验证：在 Float32 精度下，即使无噪声，极小的 Lambda 也无法实现完美还原。
原因：数值截断和精度下溢。
"""

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

# 导入配置和模块
from config import IMAGE_SIZE, THEORETICAL_CONFIG, device, RESULTS_DIR
from radon_transform import (
    TheoreticalDataGenerator,
    FourierOperatorCalculator,
    SheppLoganGenerator
)

matplotlib.use("Agg")

# ==========================================
# 关键修改：强制使用单精度 (Float32)
# ==========================================
dtype_real = torch.float32
dtype_complex = torch.complex64

class Float32Generator(TheoreticalDataGenerator):
    """
    强制单精度 (Float32)、无噪声的生成器
    """
    def __init__(self):
        self.img_size = IMAGE_SIZE
        self.N = self.img_size * self.img_size
        self.noise_mode = "none" # 强制无噪声
        self.noise_level = 0.0
        self.target_snr_db = None

        self.phantom_gen = SheppLoganGenerator(size=self.img_size)

        print("Initializing Standard Precision Fourier Operator (Complex64)...")
        self.fourier_calculator = FourierOperatorCalculator(
            beta=THEORETICAL_CONFIG['beta_vector'],
            n_coefficients=self.N,
            m=2,
        )

        # 强制转换为 complex64
        self.Phi = self.fourier_calculator.Phi_diagonal.to(device=device, dtype=dtype_complex)
        self.flatten_order = self.fourier_calculator.flatten_order.to(device)

    def forward_operator(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        """单精度正向投影"""
        if coeff_matrix.dim() == 3:
            coeff_matrix = coeff_matrix.unsqueeze(0)
        b = coeff_matrix.shape[0]
        coeff_flat = coeff_matrix.view(b, -1)
        # 输入转单精度
        d_ordered = coeff_flat[:, self.flatten_order].to(dtype_complex)
        Gd = torch.fft.fft(d_ordered, dim=-1)
        return Gd * self.Phi.unsqueeze(0)

def solve_float32(generator, F_observed, lambda_reg):
    """
    单精度求解器
    """
    F_observed = F_observed.to(dtype_complex)
    if F_observed.dim() == 1:
        F_observed = F_observed.unsqueeze(0)

    Phi = generator.Phi
    N = generator.N

    # Tikhonov 公式
    # 注意：在 Float32 下，如果 Phi 很小，|Phi|^2 会直接下溢变成 0
    numerator = N * Phi.conj().unsqueeze(0) * F_observed
    denominator = N * (Phi.abs() ** 2) + lambda_reg

    # 加上极小的 epsilon 防止除零错误 (NaN)，模拟真实计算中的保护措施
    # 但在lambda=0测试中，我们不仅想看报错，还想看数值行为，所以这里暂时不加，让它自然计算
    # 如果出现 NaN，说明发生了 0/0
    u = numerator / denominator

    # 如果产生了 NaN (0/0)，将其置为 0 (通常的处理方式)
    if torch.isnan(u).any():
        # print(f"Warning: NaN detected at lambda={lambda_reg}. Replacing with 0.")
        u = torch.nan_to_num(u, nan=0.0)

    d_complex = torch.fft.ifft(u, dim=-1)
    d_real = d_complex.real

    batch_size = F_observed.shape[0]
    out = torch.zeros(batch_size, N, device=F_observed.device, dtype=dtype_real)
    out.scatter_(1, generator.flatten_order.unsqueeze(0).expand(batch_size, -1), d_real)
    return out.view(batch_size, 1, IMAGE_SIZE, IMAGE_SIZE)

def main():
    print("="*60)
    print("SEARCH EXPERIMENT: Noiseless + Float32 (Standard Precision)")
    print("Objective: Test if Float32 can achieve 0 error with small Lambda")
    print("="*60)

    # 1. 准备数据
    gen = Float32Generator()

    # 分析 Float32 下的算子最小值
    phi_abs = gen.Phi.abs()
    print(f"Operator Statistics (Float32):")
    print(f"  Max: {phi_abs.max().item():.4e}")
    print(f"  Min: {phi_abs.min().item():.4e}")
    # 注意：Float32 的最小正规数约为 1e-38，但精度只有 1e-7
    # 如果 |Phi|^2 小于 1e-7 且 Lambda 约为 1e-7，数值误差就会占主导

    # 真值
    phantom_np = gen.phantom_gen.generate()
    c_true = torch.from_numpy(phantom_np).to(device=device, dtype=dtype_real).unsqueeze(0).unsqueeze(0)

    # 观测值
    F_clean = gen.forward_operator(c_true)

    # 2. 定义搜索空间
    # 包含典型值、极小值和 0
    lambdas = [1.0, 1e-3, 1e-7, 1e-15,1e-25,1e-35,1e-40,1e-45,1e-50, 0.0]

    print("\n" + "-" * 65)
    print(f"{'Lambda':<15} | {'RES (Relative Error)':<20} | {'MSE':<20}")
    print("-" * 65)

    best_res = float('inf')
    best_lam = None
    best_img = None

    for lam in lambdas:
        c_est = solve_float32(gen, F_clean, lambda_reg=lam)

        diff = c_est - c_true
        norm_diff = torch.norm(diff)
        norm_true = torch.norm(c_true)
        res = (norm_diff / norm_true).item()
        mse = torch.mean(diff ** 2).item()

        print(f"{lam:<15.1e} | {res:<20.4e} | {mse:<20.4e}")

        if res < best_res:
            best_res = res
            best_lam = lam
            best_img = c_est.detach().cpu().numpy().squeeze()

    # 3. 结果可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    c_true_np = c_true.squeeze().cpu().numpy()

    vmin, vmax = c_true_np.min(), c_true_np.max()

    axes[0].imshow(c_true_np, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth")

    axes[1].imshow(best_img, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Float32 Best Result\n$\lambda$={best_lam}, RES={best_res:.4f}")

    # 误差图
    err = np.abs(c_true_np - best_img)
    im2 = axes[2].imshow(err, cmap="hot", origin="lower")
    axes[2].set_title("Error Map")
    plt.colorbar(im2, ax=axes[2])

    save_path = os.path.join(RESULTS_DIR, "float32_verification.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nResult image saved to: {save_path}")

    if best_res > 0.1:
        print("\n>> 结论验证：即使无噪声，Float32 精度也无法将误差降至 0。")
        print("   原因：算子高频值在 Float32 下发生了严重的精度丢失或下溢。")

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    main()