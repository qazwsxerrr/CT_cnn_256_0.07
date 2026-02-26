"""Compare Tikhonov reconstructions at λ=1e-5, config λ, and best-found λ for current noise settings.

- Uses the noise mode/level (or target SNR) from models/config.py.
- Generates a single Shepp-Logan sample with a fixed seed.
- Runs two reconstructions:
    1) λ = 1e-5 (fixed baseline)
    2) Using DATA_CONFIG['lambda_reg'] (config lambda)
    3) Searching for the best lambda (coarse + fine) that minimizes RES
- Saves a 1x3 plot: λ=1e-5 | Config λ | Best λ, with RES/λ annotated.
"""

import argparse
import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import DATA_CONFIG, IMAGE_SIZE, RESULTS_DIR
from radon_transform import (
    SheppLoganGenerator,
    TheoreticalDataGenerator,
    direct_tikhonov_solve,
)


def compute_res(recon: torch.Tensor, target: torch.Tensor) -> float:
    """RES = ||recon - target|| / ||target||."""
    diff_sq_sum = torch.sum(torch.abs(recon - target) ** 2)
    true_sq_sum = torch.sum(torch.abs(target) ** 2) + 1e-12
    return torch.sqrt(diff_sq_sum / true_sq_sum).item()


def search_best_lambda(
    generator: TheoreticalDataGenerator,
    coeff_true: torch.Tensor,
    F_obs: torch.Tensor,
    lambda_list: list,
    fine_points: int = 20,
) -> Dict[str, torch.Tensor]:
    """Two-stage search (coarse + fine) for lambda minimizing RES."""
    lambda_list = sorted(lambda_list)
    best_res = float("inf")
    best_lambda = None
    best_est = None

    def solve_and_res(lam: float) -> (torch.Tensor, float):
        coeff_est = direct_tikhonov_solve(generator, F_obs, lambda_reg=lam).real.squeeze()
        res = compute_res(coeff_est, coeff_true)
        return coeff_est, res

    # Coarse search
    for lam in lambda_list:
        coeff_est, res = solve_and_res(lam)
        if res < best_res:
            best_res = res
            best_lambda = lam
            best_est = coeff_est

    # Fine search around best lambda
    idx = lambda_list.index(best_lambda)
    low_bound = lambda_list[max(0, idx - 1)]
    high_bound = lambda_list[min(len(lambda_list) - 1, idx + 1)]
    if low_bound == best_lambda:
        low_bound = best_lambda * 0.1
    if high_bound == best_lambda:
        high_bound = best_lambda * 10.0

    fine_lambdas = np.logspace(np.log10(low_bound), np.log10(high_bound), fine_points)
    for lam in fine_lambdas:
        coeff_est, res = solve_and_res(lam)
        if res < best_res:
            best_res = res
            best_lambda = lam
            best_est = coeff_est

    return {
        "best_lambda": best_lambda,
        "best_res": best_res,
        "best_est": best_est,
    }


def run_case(seed: int, fine_points: int, lambda_list: list) -> Dict[str, np.ndarray]:
    """Generate one sample using current config noise settings and compare λ=1e-5 vs config λ vs best λ."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Configure generator with current config noise settings
    generator = TheoreticalDataGenerator()
    generator.phantom_gen = SheppLoganGenerator(size=IMAGE_SIZE)
    current_mode = DATA_CONFIG.get("noise_mode", "snr")
    current_level = DATA_CONFIG.get("noise_level", 0.1)
    current_snr = DATA_CONFIG.get("target_snr_db", 30.0)

    generator.noise_mode = current_mode
    generator.noise_level = current_level
    generator.target_snr_db = current_snr

    config_lambda = DATA_CONFIG.get("lambda_reg", 0.01)

    coeff_true, _, F_obs, _ = generator.generate_training_sample(
        random_seed=seed,
        lambda_reg=config_lambda,
    )
    coeff_true = coeff_true.squeeze()
    F_obs = F_obs  # already flattened

    # Fixed lambda reconstruction (λ = 1e-5)
    fixed_lambda = 1e-5
    recon_fixed = direct_tikhonov_solve(generator, F_obs, lambda_reg=fixed_lambda).real.squeeze()
    res_fixed = compute_res(recon_fixed, coeff_true)

    # Config lambda reconstruction
    recon_config = direct_tikhonov_solve(generator, F_obs, lambda_reg=config_lambda).real.squeeze()
    res_config = compute_res(recon_config, coeff_true)

    # Best lambda search
    search_result = search_best_lambda(
        generator=generator,
        coeff_true=coeff_true,
        F_obs=F_obs,
        lambda_list=lambda_list,
        fine_points=fine_points,
    )
    recon_best = search_result["best_est"]
    best_lambda = search_result["best_lambda"]
    best_res = search_result["best_res"]

    return {
        "true_img": coeff_true.cpu().numpy(),
        "fixed_img": recon_fixed.cpu().numpy(),
        "fixed_res": res_fixed,
        "fixed_lambda": fixed_lambda,
        "config_img": recon_config.cpu().numpy(),
        "best_img": recon_best.cpu().numpy(),
        "config_res": res_config,
        "config_lambda": config_lambda,
        "best_res": best_res,
        "best_lambda": best_lambda,
        "noise_mode": current_mode,
        "noise_level": current_level,
        "target_snr_db": current_snr,
    }


def plot_results(results: Dict[str, np.ndarray], save_path: str) -> None:
    """Plot λ=1e-5 | Config λ | Best λ in one row with labels, scaled by true image range."""
    true_vals = results["true_img"].ravel()
    vmin, vmax = true_vals.min(), true_vals.max()
    if vmin == vmax:  # degenerate case
        vmin, vmax = np.percentile(true_vals, [1, 99])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    axes[0].imshow(results["fixed_img"], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[0].set_title(f"λ=1e-5\nRES={results['fixed_res']:.4f}")
    axes[0].axis("off")

    axes[1].imshow(results["config_img"], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[1].set_title(f"Config λ={results['config_lambda']:.2e}\nRES={results['config_res']:.4f}")
    axes[1].axis("off")

    axes[2].imshow(results["best_img"], cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[2].set_title(f"Best λ={results['best_lambda']:.2e}\nRES={results['best_res']:.4f}")
    axes[2].axis("off")

    noise_str = (
        f"Mode={results['noise_mode']}, "
        f"level={results['noise_level']}" if results["noise_mode"] != "snr"
        else f"Mode=SNR, target_snr_db={results['target_snr_db']}"
    )
    plt.suptitle(
        f"Tikhonov: λ=1e-5 vs Config λ vs Best λ ({noise_str})",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout(pad=0.3, w_pad=0.4)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison to {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Tikhonov reconstructions at config lambda vs best lambda for current noise settings."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed to keep phantom/noise identical.",
    )
    parser.add_argument(
        "--fine_points",
        type=int,
        default=20,
        help="Number of fine-search points (log-spaced) around the best coarse lambda.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    coarse_lambda_list = [
        1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5,
        1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0,
    ]

    results = run_case(
        seed=args.seed,
        fine_points=args.fine_points,
        lambda_list=coarse_lambda_list,
    )

    print("==== Config λ vs Best λ (Tikhonov) ====")
    print(f"Noise mode: {results['noise_mode']} | level: {results['noise_level']} | target_snr_db: {results['target_snr_db']}")
    print(f"Fixed  λ: {results['fixed_lambda']:.3e} | RES: {results['fixed_res']:.6f}")
    print(f"Config λ: {results['config_lambda']:.3e} | RES: {results['config_res']:.6f}")
    print(f"Best   λ: {results['best_lambda']:.3e} | RES: {results['best_res']:.6f}")

    save_path = os.path.join(RESULTS_DIR, "tikhonov_config_vs_best.png")
    plot_results(results, save_path)


if __name__ == "__main__":
    main()
