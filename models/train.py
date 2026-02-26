import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import time
import os
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import initialize_model, count_parameters
from radon_transform import TheoreticalDataGenerator
from config import (
    n_data, n_train, starter_learning_rate,
    device, MODEL_PATH, BEST_MODEL_PATH, CHECKPOINT_DIR,
    TRAINING_CONFIG, DATA_CONFIG, LOGGING_CONFIG
)


class TheoreticalTrainer:
    def __init__(self):
        self._setup_logging()
        self.model = initialize_model()
        self.data_generator = TheoreticalDataGenerator()
        self.image_gen = self.data_generator.image_gen
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
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'data_fidelity_error': [],
            'update_difference': []
        }
        self.logger.info("Theoretical trainer initialized successfully")

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

    def _generate_training_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = n_data
        return self.data_generator.generate_batch(batch_size, random_seed=self.current_iter)

    def _validate(self):
        val_bs = DATA_CONFIG.get('val_batch_size', n_data)
        seed = self.current_iter if DATA_CONFIG.get('val_reproducible', False) else None
        coeff_true_val, _, F_observed_val, coeff_initial_val = self.data_generator.generate_batch(
            batch_size=val_bs, random_seed=seed
        )
        coeff_true_val = coeff_true_val.to(device)
        F_observed_val = F_observed_val.to(device)
        coeff_initial_val = coeff_initial_val.to(device)
        if not torch.is_complex(F_observed_val):
            F_observed_val = torch.complex(F_observed_val, torch.zeros_like(F_observed_val))
        self.model.eval()
        with torch.no_grad():
            coeff_pred, history, metrics = self.model(coeff_initial_val, F_observed_val)
            diff_sq_sum = torch.sum(torch.abs(coeff_pred - coeff_true_val) ** 2)
            true_sq_sum = torch.sum(torch.abs(coeff_true_val) ** 2)
            val_loss = torch.sqrt(diff_sq_sum / (true_sq_sum ))
        self.model.train()
        return val_loss.item(), metrics

    def train(self):
        self.logger.info("Starting theoretical CT reconstruction training...")
        self.logger.info(f"Total iterations: {n_train}")
        self.logger.info(f"Batch size: {n_data}")
        self.logger.info(f"Model parameters: {count_parameters(self.model):,}")
        total_start_time = time.time()
        for self.current_iter in range(n_train):
            iter_start_time = time.time()
            coeff_true, f_true, F_observed, coeff_initial = self._generate_training_batch()
            coeff_true = coeff_true.to(device)
            f_true = f_true.to(device)
            F_observed = F_observed.to(device)
            coeff_initial = coeff_initial.to(device)
            if not torch.is_complex(F_observed):
                F_observed = torch.complex(F_observed, torch.zeros_like(F_observed))
            self.optimizer.zero_grad()
            coeff_pred, history, metrics = self.model(coeff_initial, F_observed)
            diff_sq_sum = torch.sum(torch.abs(coeff_pred - coeff_true) ** 2)
            true_sq_sum = torch.sum(torch.abs(coeff_true) ** 2)
            loss = torch.sqrt(diff_sq_sum / true_sq_sum)
            loss.backward()
            if TRAINING_CONFIG.get('gradient_clip_value', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    TRAINING_CONFIG['gradient_clip_value']
                )
            self.optimizer.step()
            self.scheduler.step()
            # 记录训练指标用于画图
            self.training_history['train_loss'].append(loss.item())
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            if metrics is not None:
                self.training_history['data_fidelity_error'].append(metrics.get('data_fidelity_error', 0.0))
                self.training_history['update_difference'].append(metrics.get('update_difference', 0.0))
            iter_time = time.time() - iter_start_time
            if self.current_iter % TRAINING_CONFIG['validation_interval'] == 0:
                val_loss, val_metrics = self._validate()
                # 记录验证损失
                self.training_history['val_loss'].append(val_loss)
                self.logger.info(
                    f"Iter: {self.current_iter:4d} | "
                    f"Train RES: {loss.item():.6f} | Val RES: {val_loss:.6f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.8f} | "
                    f"Time: {iter_time:.3f}s | Data Fidelity Error: {metrics['data_fidelity_error']:.6f} | "
                    f"Update Difference: {metrics['update_difference']:.6f}"
                )
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                if (self.patience_counter >= TRAINING_CONFIG['early_stopping_patience'] and
                    TRAINING_CONFIG['early_stopping_patience'] > 0):
                    self.logger.info(f"Early stopping triggered after {self.current_iter} iterations")
                    break
            if self.current_iter % TRAINING_CONFIG['save_interval'] == 0:
                self._save_checkpoint()
            if self.current_iter % 500 == 0 and self.current_iter > 0:
                self._save_training_plots()
        total_time = time.time() - total_start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        self._save_checkpoint()
        self._save_training_plots()

    def _save_checkpoint(self, is_best=False):
        checkpoint = {
            'iter': self.current_iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
        }
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR, f'checkpoint_iter_{self.current_iter}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            torch.save(checkpoint, BEST_MODEL_PATH)
            self.logger.info(f"New best model saved with validation loss: {self.best_val_loss:.6f}")
        torch.save(checkpoint, MODEL_PATH)

    def _save_training_plots(self):
        if len(self.training_history['train_loss']) == 0:
            return
        fig = None
        try:
            start_idx = 150 if len(self.training_history['train_loss']) > 150 else 0

            def _slice(seq):
                return seq[start_idx:] if len(seq) > start_idx else seq

            train_loss = _slice(self.training_history['train_loss'])
            val_loss = _slice(self.training_history['val_loss'])
            lr_hist = _slice(self.training_history['learning_rate'])
            data_err = _slice(self.training_history['data_fidelity_error'])
            upd_diff = _slice(self.training_history['update_difference'])

            if len(train_loss) == 0:
                return
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes[0, 0].plot(train_loss, label='Train Loss')
            if len(val_loss) > 0:
                axes[0, 0].plot(val_loss, label='Val Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            axes[0, 1].plot(lr_hist)
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].grid(True)
            axes[1, 0].plot(data_err)
            axes[1, 0].set_title('Data Fidelity Error')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Error')
            axes[1, 0].grid(True)
            axes[1, 1].plot(upd_diff)
            axes[1, 1].set_title('Theoretical vs Learned Update Difference')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Difference')
            axes[1, 1].grid(True)
            plt.tight_layout()
            plot_path = os.path.join(LOGGING_CONFIG['log_dir'], 'training_progress.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Training plots saved to {plot_path}")
        except Exception as e:
            self.logger.error(f"Error saving training plots: {e}")
        finally:
            if fig is not None:
                plt.close(fig)

    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_iter = checkpoint['iter']
            self.best_val_loss = checkpoint['best_val_loss']
            self.training_history = checkpoint['training_history']
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
            self.logger.info(f"Resuming from iteration {self.current_iter}")
        else:
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")


def main():
    print("=" * 60)
    print("THEORETICAL CT RECONSTRUCTION TRAINING")
    print("=" * 60)
    trainer = TheoreticalTrainer()
    resume_path = None
    if resume_path and os.path.exists(resume_path):
        trainer.load_checkpoint(resume_path)
    trainer.train()
    print("Theoretical training completed successfully!")


if __name__ == '__main__':
    main()
