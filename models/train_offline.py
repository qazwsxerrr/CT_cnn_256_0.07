import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import time
import os
import sys
import logging
import matplotlib
import random
import numpy as np

# 强制使用非交互式后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# [Windows 多进程防刷屏补丁]
if os.name == 'nt' and 'data_fidelity_error' not in globals():
    import multiprocessing

    if multiprocessing.current_process().name != 'MainProcess':
        class NullWriter:
            def write(self, s): pass

            def flush(self): pass


        sys.stdout = NullWriter()
        sys.stderr = NullWriter()

from model import initialize_model
from config import (
    n_data, n_train, starter_learning_rate,
    device, MODEL_PATH, BEST_MODEL_PATH, CHECKPOINT_DIR,
    TRAINING_CONFIG, LOGGING_CONFIG, DATA_DIR
)

# ============================================================================

# 配置区域 (新增)
# ============================================================================
# 梯度累积步数
ACCUMULATION_STEPS = 4


# ============================================================================
# 0. 辅助函数
# ============================================================================
def seed_everything(seed=42):
    """固定所有随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Global seed set to {seed}")


def tv_loss(img):
    """计算全变分损失 (Total Variation Loss) - 像素级平均"""
    batch_size = img.shape[0]
    h_x = img.shape[2]
    w_x = img.shape[3]
    w_variance = torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    h_variance = torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    num_elements = batch_size * h_x * w_x
    return (w_variance + h_variance) / num_elements


class CTDataset(Dataset):
    def __init__(self, pt_file_path):
        print(f"Loading dataset from {pt_file_path}...")
        try:
            data = torch.load(pt_file_path, weights_only=True)
        except TypeError:
            data = torch.load(pt_file_path)

        self.coeff_true = data['coeff_true']
        self.F_observed = data['F_observed']
        self.coeff_initial = data['coeff_initial']
        self.length = self.coeff_true.shape[0]
        print(f"Loaded {self.length} samples.")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (
            self.coeff_true[idx],
            self.F_observed[idx],
            self.coeff_initial[idx]
        )


class TheoreticalTrainerOffline:
    def __init__(self):
        self._setup_logging()
        seed_everything(42)

        self.model = initialize_model()
        self.scaler = torch.amp.GradScaler('cuda')

        train_path = os.path.join(DATA_DIR, "train_dataset.pt")
        val_path = os.path.join(DATA_DIR, "val_dataset.pt")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Please run generate_data.py first! Missing: {train_path}")

        train_dataset = CTDataset(train_path)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=n_data,  # 物理 Batch Size (显存限制)
            shuffle=True,  # Shuffle 对梯度累积很重要
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        if os.path.exists(val_path):
            val_dataset = CTDataset(val_path)
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=n_data,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
        else:
            self.val_loader = None

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=starter_learning_rate,
            weight_decay=1e-4
        )

        lr_lambda = lambda step: 1.0 / (1.0 + step / 500.0)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        self.current_iter = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.last_val_res = 0.0

        self.training_history = {
            'train_loss': [], 'val_loss': [], 'train_res': [], 'val_res': [],
            'learning_rate': [], 'data_fidelity_error': []
        }

        self.criterion_l1 = nn.L1Loss()

        self.logger.info(f"Offline trainer initialized on {device}.")
        self.logger.info(f"Physical Batch Size: {n_data}")
        self.logger.info(f"Accumulation Steps: {ACCUMULATION_STEPS}")
        self.logger.info(f"Effective Batch Size: {n_data * ACCUMULATION_STEPS}")
        self.logger.info("Using Composite Loss: L1 + 0.001 * TV (Normalized)")

    def _setup_logging(self):
        log_dir = LOGGING_CONFIG['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger('TheoreticalCTTrainer')
        self.logger.setLevel(getattr(logging, LOGGING_CONFIG['log_level']))
        self.logger.handlers.clear()
        if LOGGING_CONFIG['log_to_console']:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        if LOGGING_CONFIG['log_to_file']:
            file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
            file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def _validate(self):
        if self.val_loader is None:
            return 0.0, 0.0, {}

        self.model.eval()
        total_val_loss = 0.0
        total_val_res = 0.0
        num_batches = 0

        with torch.no_grad():
            for i, (coeff_true, F_observed, coeff_initial) in enumerate(self.val_loader):
                coeff_true = coeff_true.to(device, non_blocking=True)
                F_observed = F_observed.to(device, non_blocking=True)
                coeff_initial = coeff_initial.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    coeff_pred, _, metrics = self.model(coeff_initial, F_observed)

                    diff_sq_sum = torch.sum(torch.abs(coeff_pred - coeff_true) ** 2)
                    true_sq_sum = torch.sum(torch.abs(coeff_true) ** 2)
                    val_res = torch.sqrt(diff_sq_sum / true_sq_sum)

                    l1_val = self.criterion_l1(coeff_pred, coeff_true)
                    tv_val = tv_loss(coeff_pred)
                    val_loss = l1_val + 0.001 * tv_val

                total_val_loss += val_loss.item()
                total_val_res += val_res.item()
                num_batches += 1
                if i >= 10: break

        self.model.train()
        return total_val_loss / num_batches, total_val_res / num_batches, metrics

    def _save_training_plots(self):
        if len(self.training_history['train_loss']) == 0: return
        try:
            start_idx = 10 if len(self.training_history['train_loss']) > 50 else 0
            plt.figure(figsize=(10, 10))
            plt.subplot(2, 1, 1)
            plt.plot(self.training_history['train_loss'][start_idx:], label='Train Loss')
            plt.plot(self.training_history['val_loss'], label='Val Loss')
            plt.title(f'Loss (Effective Batch: {n_data * ACCUMULATION_STEPS})')
            plt.legend();
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(self.training_history['train_res'][start_idx:], label='Train RES')
            plt.plot(self.training_history['val_res'], label='Val RES')
            plt.title('Residual Error')
            plt.legend();
            plt.grid(True)

            plt.savefig(os.path.join(LOGGING_CONFIG['log_dir'], 'training_progress.png'))
            plt.close()
        except Exception as e:
            self.logger.error(f"Error saving plots: {e}")

    def train(self):
        self.logger.info("Starting OFFLINE training with GRADIENT ACCUMULATION...")
        self.logger.info(f"Target total iterations (updates): {n_train}")

        total_start_time = time.time()
        LOG_INTERVAL = TRAINING_CONFIG['validation_interval']

        # 初始化梯度累积变量
        self.optimizer.zero_grad()
        accum_loss = 0.0
        accum_res = 0.0

        # 确保循环能持续进行，直到达到 update 次数
        while self.current_iter < n_train:
            for batch_idx, (coeff_true, F_observed, coeff_initial) in enumerate(self.train_loader):
                # 达到目标迭代次数（更新次数）则退出
                if self.current_iter >= n_train: break

                iter_start_time = time.time()

                # --- 1. 数据搬运 ---
                coeff_true = coeff_true.to(device, non_blocking=True)
                F_observed = F_observed.to(device, non_blocking=True)
                coeff_initial = coeff_initial.to(device, non_blocking=True)

                # --- 2. 前向传播 ---
                with torch.amp.autocast('cuda'):
                    coeff_pred, _, metrics = self.model(coeff_initial, F_observed)

                    # 计算指标 (仅用于记录，不参与梯度)
                    with torch.no_grad():
                        diff_sq_sum = torch.sum(torch.abs(coeff_pred - coeff_true) ** 2)
                        true_sq_sum = torch.sum(torch.abs(coeff_true) ** 2)
                        res_error = torch.sqrt(diff_sq_sum / true_sq_sum)

                    l1_val = self.criterion_l1(coeff_pred, coeff_true)
                    tv_val = tv_loss(coeff_pred)
                    loss = l1_val + 0.001 * tv_val

                    # [关键修改] Loss 除以累积步数
                    loss = loss / ACCUMULATION_STEPS

                # --- 3. 反向传播 (攒梯度) ---
                self.scaler.scale(loss).backward()

                # 累积用于显示的 Loss (还原回原始量级)
                accum_loss += loss.item() * ACCUMULATION_STEPS
                accum_res += res_error.item()

                # --- 4. 梯度更新 (每 ACCUMULATION_STEPS 次执行一次) ---
                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    # 梯度裁剪
                    if TRAINING_CONFIG.get('gradient_clip_value', 0) > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            TRAINING_CONFIG['gradient_clip_value']
                        )

                    # 真正的参数更新
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # 计算这几个 batch 的平均 Loss/Res 用于日志
                    avg_loss = accum_loss / ACCUMULATION_STEPS
                    avg_res = accum_res / ACCUMULATION_STEPS

                    # 重置累积变量
                    accum_loss = 0.0
                    accum_res = 0.0

                    # 记录日志
                    self.training_history['train_loss'].append(avg_loss)
                    self.training_history['train_res'].append(avg_res)
                    self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

                    iter_time = time.time() - iter_start_time

                    # 验证与保存 (逻辑稍微调整，基于 update 次数)
                    if self.current_iter % LOG_INTERVAL == 0:
                        val_loss, val_res, _ = self._validate()
                        self.training_history['val_loss'].append(val_loss)
                        self.training_history['val_res'].append(val_res)
                        self.last_val_res = val_res

                        current_lr = self.optimizer.param_groups[0]['lr']
                        self.logger.info(
                            f"Iter: {self.current_iter:4d} | "
                            f"Loss: {avg_loss:.5f} | "
                            f"Train RES: {avg_res:.5f} | "
                            f"Val RES: {val_res:.5f} | "
                            f"LR: {current_lr:.8f} | "
                            f"Time: {iter_time:.3f}s"
                        )

                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.patience_counter = 0
                            self._save_checkpoint(is_best=True)
                            self.logger.info(f"New best model saved. Loss: {self.best_val_loss:.6f}")
                        else:
                            self.patience_counter += 1

                    if self.current_iter % 100 == 0:
                        self._save_checkpoint()

                    if self.current_iter % 500 == 0 and self.current_iter > 0:
                        self._save_training_plots()

                    # 增加迭代次数计数 (指的是 Update 次数)
                    self.current_iter += 1

                    if (self.patience_counter >= TRAINING_CONFIG['early_stopping_patience'] and
                            TRAINING_CONFIG['early_stopping_patience'] > 0):
                        self.logger.info("Early stopping triggered")
                        return

        total_time = time.time() - total_start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self._save_checkpoint()
        self._save_training_plots()

    def _save_checkpoint(self, is_best=False):
        checkpoint = {
            'iter': self.current_iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
        }
        torch.save(checkpoint, MODEL_PATH)
        if is_best:
            torch.save(checkpoint, BEST_MODEL_PATH)
        if self.current_iter % 500 == 0:  # 稍微减少 checkpoint 频率以节省空间
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_iter_{self.current_iter}.pth')
            torch.save(checkpoint, checkpoint_path)


if __name__ == '__main__':
    trainer = TheoreticalTrainerOffline()
    trainer.train()