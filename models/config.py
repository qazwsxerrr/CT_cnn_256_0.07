# -*- coding: utf-8 -*-
"""Project configuration for CT_cnn."""

import os
import torch
import numpy as np

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

IMAGE_SIZE = 256

# Theoretical training parameters.
THEORETICAL_CONFIG = {
    "beta_vector": (1, IMAGE_SIZE),
    "regularizer_type": "tv",
    "n_iter": 15,
    "n_memory_units": 8,
    "learning_rate": 1e-3,
}

n_data = 5
n_memory = THEORETICAL_CONFIG["n_memory_units"]
n_iter = THEORETICAL_CONFIG["n_iter"]
n_train = 3000

starter_learning_rate = THEORETICAL_CONFIG["learning_rate"]
learning_rate_decay = 0.95
learning_rate_step = 1000

DATA_CONFIG = {
    # Noise Configuration
    # Options: "snr" (Legacy), "additive", "multiplicative"
    "noise_mode": "additive",
    # Noise level (delta) for additive/multiplicative modes
    "noise_level": 0.1,
    # Legacy SNR target (used only if noise_mode is "snr")
    "target_snr_db": 60.0,

    "lambda_reg": 1.0e-02,
    "validation_pattern_type": "cardinal",
    "validation_seed": 42,
    "test_dataset_size": 20,
    "preserve_basis_structure": True,
    "enforce_region_constraint": False,
    "region_constraint_tolerance": 1e-6,
    "use_theoretical_mapping": True,
    "val_batch_size": n_data,
    "val_reproducible": False,
    "use_physical_tikhonov": False,
}

TRAINING_CONFIG = {
    "batch_size": n_data,
    "validation_interval": 10,
    "save_interval": 1000,
    "early_stopping_patience": 500,
    "gradient_clip_value": 1.0,
    "use_mixed_precision": False,
    "image_loss_weight": 0.1,
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_PATH = os.path.join(MODEL_DIR, "theoretical_ct_model.pth")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "theoretical_ct_best_model.pth")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "training.log")
TRAINING_PLOT_PATH = os.path.join(LOG_DIR, "training_progress.png")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

LOGGING_CONFIG = {
    "log_dir": LOG_DIR,
    "log_level": "INFO",
    "log_to_file": True,
    "log_to_console": True,
}


def print_config():
    """Print the current configuration for quick inspection."""
    print("=" * 60)
    print("THEORETICAL CT RECONSTRUCTION CONFIGURATION")
    print("=" * 60)
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Beta vector: {THEORETICAL_CONFIG['beta_vector']}")
    print(f"Regularizer type: {THEORETICAL_CONFIG['regularizer_type']}")
    print(f"Optimization iterations: {THEORETICAL_CONFIG['n_iter']}")
    print(f"Memory units: {THEORETICAL_CONFIG['n_memory_units']}")
    print(f"Device: {device}")
    print(f"Noise Mode: {DATA_CONFIG['noise_mode']}")
    if DATA_CONFIG['noise_mode'] in ["additive", "multiplicative"]:
        print(f"Noise Level (delta): {DATA_CONFIG['noise_level']}")
    else:
        print(f"Target SNR (dB): {DATA_CONFIG['target_snr_db']}")
    print(f"Training iterations: {n_train}")
    print(f"Batch size: {n_data}")
    print(f"Learning rate: {starter_learning_rate}")
    print(f"Training patience: {TRAINING_CONFIG['early_stopping_patience']}")
    print(f"Model save path: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()