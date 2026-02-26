import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import initialize_model
from radon_transform import TheoreticalDataGenerator, SheppLoganGenerator
from config import device, BEST_MODEL_PATH, MODEL_PATH, RESULTS_DIR, IMAGE_SIZE, DATA_CONFIG


def load_model():
    model = initialize_model()
    load_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No checkpoint found at {BEST_MODEL_PATH} or {MODEL_PATH}")

    checkpoint = torch.load(load_path, map_location=device, weights_only=True)
    loaded_state = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    filtered = {k: v for k, v in loaded_state.items() if k in model_state and model_state[k].shape == v.shape}
    skipped = [k for k in loaded_state.keys() if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    model.eval()
    print(f"Loaded checkpoint: {load_path}")
    if skipped:
        print(f"Skipped mismatched keys: {skipped}")
    return model


def plot_result(idx, f_true, f_init, f_pred, res_init, res_pred, save_path, target_snr_db):
    """Plot true image, Tikhonov init image, and reconstruction image with RES."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmin, vmax = f_true.min(), f_true.max()

    im0 = axes[0].imshow(f_true, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[0].set_title("True Image")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(f_init, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[1].set_title(f"Tikhonov Init\nRES={res_init:.4f}")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(f_pred, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
    axes[2].set_title(f"Reconstruction\nRES={res_pred:.4f}")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.suptitle(f"Sample {idx} | λ={0.1}", y=1.08)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{idx}] saved: {save_path}")


def build_shepp_logan_sample(generator: TheoreticalDataGenerator, seed: Optional[int] = None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    coeff_true, _, F_obs, coeff_init = generator.generate_training_sample(
        random_seed=seed,
        lambda_reg=DATA_CONFIG.get("lambda_reg", 0.01),
    )
    return coeff_true, F_obs, coeff_init


def evaluate_shepp_logan(target_snr_db: float = DATA_CONFIG.get("target_snr_db", 30.0), num_samples: int = 100):
    model = load_model()
    generator = TheoreticalDataGenerator()
    generator.target_snr_db = target_snr_db
    generator.phantom_gen = SheppLoganGenerator(size=IMAGE_SIZE)
    generator.image_gen = generator.image_gen.to(device)

    res_init_list = []
    res_pred_list = []
    last_plot_data = None

    for i in range(num_samples):
        coeff_true, F_obs, coeff_init = build_shepp_logan_sample(generator, seed=None)
        coeff_true_cpu = coeff_true.squeeze().cpu()
        coeff_init_cpu = coeff_init.squeeze().cpu()

        F_obs_batch = F_obs.unsqueeze(0).to(device)
        coeff_init_batch = coeff_init.unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            coeff_pred_batch, _, _ = model(coeff_init_batch, F_obs_batch)

        coeff_pred = coeff_pred_batch.squeeze().detach().cpu()
        diff_sq_sum_pred = torch.sum(torch.abs(coeff_pred - coeff_true_cpu) ** 2)
        diff_sq_sum_init = torch.sum(torch.abs(coeff_init_cpu - coeff_true_cpu) ** 2)
        true_sq_sum = torch.sum(torch.abs(coeff_true_cpu) ** 2)
        res_pred = torch.sqrt(diff_sq_sum_pred / true_sq_sum).item()
        res_init = torch.sqrt(diff_sq_sum_init / true_sq_sum).item()

        res_init_list.append(res_init)
        res_pred_list.append(res_pred)
        last_plot_data = (
            coeff_true_cpu.numpy(),
            coeff_init_cpu.numpy(),
            coeff_pred.numpy(),
            res_init,
            res_pred,
        )

    mean_res_init = sum(res_init_list) / len(res_init_list)
    mean_res_pred = sum(res_pred_list) / len(res_pred_list)

    if last_plot_data is not None:
        f_true_np, f_init_np, f_pred_np, res_init_last, res_pred_last = last_plot_data
        save_path = os.path.join(RESULTS_DIR, "shepp_logan_last.png")
        plot_result(
            idx="SheppLogan_last",
            f_true=f_true_np,
            f_init=f_init_np,
            f_pred=f_pred_np,
            res_init=res_init_last,
            res_pred=res_pred_last,
            save_path=save_path,
            target_snr_db=None,
        )

    print("==== Shepp-Logan Evaluation (Mean over samples) ====")
    print(f"Mean RES (init): {mean_res_init:.6f}")
    print(f"Mean RES (pred): {mean_res_pred:.6f}")


if __name__ == "__main__":
    evaluate_shepp_logan(target_snr_db=DATA_CONFIG.get("target_snr_db", 30.0))
