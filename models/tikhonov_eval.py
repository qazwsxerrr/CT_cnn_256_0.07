"""Standalone Tikhonov reconstruction demo (FFT + direct diagonal solve)."""

import os
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np

from radon_transform import TheoreticalDataGenerator, SheppLoganGenerator, direct_tikhonov_solve
from config import device, DATA_CONFIG, RESULTS_DIR
from test import build_shepp_logan_sample

matplotlib.use("Agg")


@torch.no_grad()
def solve_one(
    generator: TheoreticalDataGenerator,
    lambda_reg: float,
    target_snr_db: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Generate one sample and run Tikhonov via direct diagonal solve (FFT operator)."""
    generator.target_snr_db = target_snr_db
    coeff_true, F_obs, _ = build_shepp_logan_sample(generator, seed=None)
    coeff_true = coeff_true.squeeze(0).squeeze(0)

    coeff_est = direct_tikhonov_solve(
        generator,
        F_obs.to(device),
        lambda_reg=lambda_reg,
    ).real.squeeze()

    mse = torch.nn.functional.mse_loss(
        coeff_est.to(torch.float32),
        coeff_true.to(device, dtype=torch.float32),
    ).item()
    diff_sq_sum = torch.sum(torch.abs(coeff_est - coeff_true.to(coeff_est)) ** 2)
    true_sq_sum = torch.sum(torch.abs(coeff_true) ** 2)
    res = torch.sqrt(diff_sq_sum / true_sq_sum).item()

    return (
        coeff_true.cpu().numpy(),
        coeff_est.cpu().numpy(),
        mse,
        res,
    )


def main(
    lambda_list: Optional[list] = None,
    target_snr_db: Optional[float] = None,
):
    print(f"Running Tikhonov reconstruction on device: {device}")
    if lambda_list is None:
        lambda_list = [1e-8,1e-7,1e-6,1e-5,1e-4, 1e-3,0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    if target_snr_db is None:
        target_snr_db = DATA_CONFIG.get("target_snr_db", 30.0)
    print(f"Settings: target_snr_db={target_snr_db}")

    generator = TheoreticalDataGenerator()
    generator.target_snr_db = target_snr_db
    generator.phantom_gen = SheppLoganGenerator()
    coeff_true, F_obs, _ = build_shepp_logan_sample(generator, seed=None)
    coeff_true = coeff_true.squeeze(0).squeeze(0)
    print(f"Data generated with target_snr_db: {target_snr_db}")

    # Scan lambdas
    print(f"\n{'Lambda':<12} | {'MSE':<12} | {'RSE':<12}")
    print("-" * 45)
    mse_best = float("inf")
    best_lambda = None
    best_est = None
    for lam in lambda_list:
        coeff_est = direct_tikhonov_solve(
            generator,
            F_obs.to(device),
            lambda_reg=lam,
        ).real.squeeze()
        mse = torch.mean((coeff_est - coeff_true.to(device)) ** 2).item()
        diff_norm = torch.norm(coeff_est - coeff_true.to(coeff_est))
        true_norm = torch.norm(coeff_true)
        rse = (diff_norm / true_norm).item()
        print(f"{lam:<12.1e} | {mse:<12.6f} | {rse:<12.6f}")
        if mse < mse_best:
            mse_best = mse
            best_lambda = lam
            best_est = coeff_est.detach().cpu().numpy()
    print("-" * 45)
    print(f"Best Lambda: {best_lambda} with MSE: {mse_best:.6f}")

    # Save heatmaps
    coeff_true_np = coeff_true.cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    vmin, vmax = coeff_true_np.min(), coeff_true_np.max()
    im0 = axes[0].imshow(coeff_true_np, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("True Coeff (Shepp-Logan)")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(best_est, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Best Est (lambda={best_lambda:.1e})")
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "tikhonov_best_shepp_logan.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tikhonov reconstruction demo")
    parser.add_argument("--lambda_list", type=str, default="", help="Comma-separated lambdas to scan; empty uses default list")
    parser.add_argument("--target_snr_db", type=float, default=DATA_CONFIG.get("target_snr_db", 30.0), help="Target SNR in dB for complex measurements")
    args = parser.parse_args()

    if args.lambda_list:
        lambdas = [float(x) for x in args.lambda_list.split(",")]
    else:
        lambdas = None

    main(lambda_list=lambdas, target_snr_db=args.target_snr_db)
