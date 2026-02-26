"""Standalone Tikhonov reconstruction demo (FFT + direct diagonal solve)."""

import os
from typing import Optional, Tuple
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np

# 移除对 test 的依赖，直接使用核心模块
from radon_transform import TheoreticalDataGenerator, SheppLoganGenerator, direct_tikhonov_solve
from config import device, DATA_CONFIG, RESULTS_DIR

matplotlib.use("Agg")


def get_shepp_logan_data(generator, seed=None):
    """直接生成 Shepp-Logan 验证数据"""
    if seed is not None:
        torch.manual_seed(seed)

    # 使用 generator 内部的机制生成数据
    # generate_training_sample 返回: (coeff_true, f_true, F_observed, coeff_initial)
    # 注意：generate_training_sample 内部会根据 target_snr_db 或 noise_level 添加噪声
    c_true, f_true, F_obs, c_init = generator.generate_training_sample(random_seed=seed)

    # 调整维度以适配 Tikhonov求解器 [H, W] -> [1, 1, H, W] 或保持原样
    # direct_tikhonov_solve 期望 F_observed 是 [Batch, N] 或 [N]
    return c_true, F_obs


def main(
    lambda_list: Optional[list] = None,
    target_snr_db: Optional[float] = None,
):
    print(f"Running Tikhonov reconstruction on device: {device}")

    # 1. 设置默认参数
    if lambda_list is None:
        # 默认粗搜索范围：从 1e-8 到 1000
        lambda_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

    if target_snr_db is None:
        target_snr_db = DATA_CONFIG.get("target_snr_db", 30.0)

    # 2. 初始化生成器
    generator = TheoreticalDataGenerator()

    # 获取当前的噪声配置用于显示逻辑
    current_noise_mode = DATA_CONFIG.get("noise_mode", "snr")
    current_noise_level = DATA_CONFIG.get("noise_level", 0.05)

    # 根据模式配置 generator 和打印信息
    if current_noise_mode == "snr":
        # 仅在 SNR 模式下应用命令行传入的 target_snr_db
        generator.target_snr_db = target_snr_db
        print(f"Settings: Mode={current_noise_mode}, Target SNR={target_snr_db}dB")
    else:
        # 其他模式使用 config 中的 level
        # 注意：generator 初始化时已经加载了 config，这里主要是为了确保逻辑一致性
        generator.noise_mode = current_noise_mode
        generator.noise_level = current_noise_level
        print(f"Settings: Mode={current_noise_mode}, Level(δ)={current_noise_level}")

    # 强制使用 Shepp-Logan 幻影
    generator.phantom_gen = SheppLoganGenerator()

    # 3. 生成数据
    coeff_true, F_obs = get_shepp_logan_data(generator, seed=42)
    # coeff_true: [H, W], F_obs: [H*W]
    print(f"Data generated. Shape: {coeff_true.shape}")

    # ----------------------------------------------------------------
    # 第一阶段：粗搜索 (Coarse Search)
    # ----------------------------------------------------------------
    print("\n" + "="*40)
    print("STAGE 1: Coarse Search")
    print("="*40)
    print(f"{'Lambda':<12} | {'RES':<12}")
    print("-" * 30)

    best_res = float("inf")
    best_lambda = None
    best_est = None

    # 确保 lambda_list 排序，方便后续精细搜索
    lambda_list = sorted(lambda_list)

    for lam in lambda_list:
        # 求解
        coeff_est = direct_tikhonov_solve(
            generator,
            F_obs.to(device),
            lambda_reg=lam,
        ).real.squeeze()

        # 计算 RES (Residual Error / Relative Error)
        # RES = ||x_pred - x_true|| / ||x_true||
        diff_norm = torch.norm(coeff_est - coeff_true.to(device))
        true_norm = torch.norm(coeff_true.to(device))
        res = (diff_norm / true_norm).item()

        print(f"{lam:<12.1e} | {res:<12.6f}")

        if res < best_res:
            best_res = res
            best_lambda = lam
            best_est = coeff_est

    print("-" * 30)
    print(f"Coarse Best Lambda: {best_lambda} with RES: {best_res:.6f}")

    # ----------------------------------------------------------------
    # 第二阶段：精细搜索 (Fine Search)
    # ----------------------------------------------------------------
    # 在最佳 lambda 附近的区间进行更密集的搜索

    idx = lambda_list.index(best_lambda)
    low_bound = lambda_list[max(0, idx - 1)]
    high_bound = lambda_list[min(len(lambda_list) - 1, idx + 1)]

    # 如果最佳值在边界，或者区间太小，适当扩展
    if low_bound == best_lambda: low_bound = best_lambda * 0.1
    if high_bound == best_lambda: high_bound = best_lambda * 10.0

    print("\n" + "="*40)
    print(f"STAGE 2: Fine Search in [{low_bound:.1e}, {high_bound:.1e}]")
    print("="*40)
    print(f"{'Lambda':<12} | {'RES':<12}")
    print("-" * 30)

    # 在对数尺度上生成 20 个点
    fine_lambdas = np.logspace(np.log10(low_bound), np.log10(high_bound), 20)

    final_best_res = best_res
    final_best_lambda = best_lambda
    # best_est 保持第一阶段的最佳，除非第二阶段找到更好的

    for lam in fine_lambdas:
        coeff_est = direct_tikhonov_solve(
            generator,
            F_obs.to(device),
            lambda_reg=lam,
        ).real.squeeze()

        diff_norm = torch.norm(coeff_est - coeff_true.to(device))
        true_norm = torch.norm(coeff_true.to(device))
        res = (diff_norm / true_norm).item()

        # 简单的打印过滤
        marker = "*" if res < final_best_res else " "
        print(f"{lam:<12.4e} | {res:<12.6f} {marker}")

        if res < final_best_res:
            final_best_res = res
            final_best_lambda = lam
            best_est = coeff_est

    print("-" * 30)
    print(f"FINAL BEST LAMBDA: {final_best_lambda:.6e}")
    print(f"FINAL BEST RES   : {final_best_res:.6f}")

    # ----------------------------------------------------------------
    # 保存结果
    # ----------------------------------------------------------------
    if best_est is not None:
        best_est_np = best_est.detach().cpu().numpy()
        coeff_true_np = coeff_true.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        vmin, vmax = coeff_true_np.min(), coeff_true_np.max()

        # origin="lower" 矫正显示方向 (使 Index 0 在底部)
        im0 = axes[0].imshow(coeff_true_np, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axes[0].set_title("True Coeff (Shepp-Logan)")
        plt.colorbar(im0, ax=axes[0])

        # 根据噪声模式动态生成标题字符串
        if current_noise_mode == "snr":
            noise_str = f"SNR={target_snr_db:.1f}dB"
        else:
            # 使用简写以节省空间
            mode_map = {"additive": "Add", "multiplicative": "Mult"}
            mode_str = mode_map.get(current_noise_mode, current_noise_mode)
            noise_str = f"{mode_str} $\delta$={current_noise_level}"

        im1 = axes[1].imshow(best_est_np, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Best Est (λ={final_best_lambda:.2e})\n{noise_str}, RES={final_best_res:.4f}")
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        out_path = os.path.join(RESULTS_DIR, "tikhonov_best_res.png")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nHeatmap saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tikhonov reconstruction with auto-lambda search")
    parser.add_argument("--target_snr_db", type=float, default=None, help="Override target SNR in dB (only for SNR mode)")
    args = parser.parse_args()

    main(target_snr_db=args.target_snr_db)