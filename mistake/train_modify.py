import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import (TheoreticalCTNet, initialize_model, count_parameters,
                   CoefficientMapping, RadonFourierOperator2D)
from box_spline import CardinalBSpline2D
from radon_transform import RadonTransformSimulator, FourierOperatorCalculator
from config import (
    n_data, n_train, starter_learning_rate,
    device, MODEL_PATH, BEST_MODEL_PATH, CHECKPOINT_DIR,
    TRAINING_CONFIG, DATA_CONFIG, LOGGING_CONFIG, THEORETICAL_CONFIG
)


# ============================================================================
# 基于算法理论的数据生成器
# ============================================================================
class TheoreticalDataGenerator:
    """基于算法理论的数据生成器 - 实现完整的Radon变换和F = Φ G d流程"""

    def __init__(self):
        self.beta = THEORETICAL_CONFIG['beta_vector']
        self.height, self.width = 21, 21
        self.noise_level = DATA_CONFIG['noise_level']
        self.image_shape = (128, 128)  # CT图像尺寸

        # 基数B样条生成器 - 实现φ(x,y) = B₂(x) · B₁(y)
        self.bspline_generator = CardinalBSpline2D()

        # Radon变换模拟器 - 模拟CT扫描过程
        try:
            self.radon_simulator = RadonTransformSimulator(
                beta=self.beta,
                image_shape=self.image_shape,
                n_angles=1,  # 单角投影
                n_detectors=441  # 441个探测器
            )
        except RuntimeError:
            # 若缺少 ODL torch 桥接，使用 None 标记以便后续退化为 ΦGd
            self.radon_simulator = None

        # Fourier算子计算器 - 计算F = Φ G d
        self.fourier_calculator = FourierOperatorCalculator(
            beta=self.beta,
            n_coefficients=441,
            m=2
        )

        # 系数映射
        self.mapping = CoefficientMapping(self.beta, (self.height, self.width))

        # 物理算子（保留用于兼容性）
        self.operator = RadonFourierOperator2D(
            self.beta, self.height, self.width
        ).to(device)

    def generate_training_sample(self, random_seed=None):
        """
        生成单个训练样本 - 基于完整的算法理论

        实现完整流程：
        1. 生成系数c_k ~ N(0,1)
        2. 生成图像f(x,y) = Σ c_{i,j} · φ_{i,j}(x,y)
        3. 模拟CT扫描：R_α f（Radon变换）
        4. 计算观测向量F = Φ G d

        Returns:
            coeff_true: (21, 21) 真实系数矩阵 c_{i,j}，使用标准正态分布
            f_true: (128, 128) 基于基数B样条的真实图像
            F_observed: (441,) 观测向量，通过Radon变换和F = Φ G d计算得到
            coeff_initial: (21, 21) 初始系数估计（零初始化）
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        # 1. 真实系数矩阵
        coeff_true = self._generate_normal_coefficients()

        # 2. 基于基数B样条生成真实图像 f(x,y)
        f_true = self._generate_bspline_image(coeff_true, random_seed)

        # 3. 模拟CT扫描：Radon变换（若 ODL 不可用则退化为 ΦGd 一致算子）
        if self.radon_simulator is not None:
            try:
                f_true_batch = f_true.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
                radon_data = self.radon_simulator.forward_radon_transform(f_true_batch)

                # 4. 添加噪声（模拟实际CT扫描）
                if self.noise_level > 0:
                    radon_data = self.radon_simulator.add_noise(radon_data, self.noise_level)

                # 5. 计算观测向量 F = Φ G d
                F_observed = self.fourier_calculator.compute_F_from_radon(radon_data)
            except RuntimeError:
                # 无 ODL 桥接：使用与训练算子一致的 ΦGd 生成观测
                F_observed = self.operator(coeff_true.unsqueeze(0).unsqueeze(0).to(device)).squeeze(0)
        else:
            # 无 ODL 桥接：使用与训练算子一致的 ΦGd 生成观测
            F_observed = self.operator(coeff_true.unsqueeze(0).unsqueeze(0).to(device)).squeeze(0)

        # 压缩维度以便训练
        F_observed = F_observed.squeeze(0)  # (441,)

        # 6. 初始估计
        coeff_initial = self._tikhonov_init(F_observed)

        return coeff_true, f_true, F_observed, coeff_initial

    def _generate_normal_coefficients(self):
        """
        生成使用标准正态分布的系数矩阵c_k

        根据f(x,y) = Σ_{i=0}^{20} Σ_{j=0}^{20} c_{i,j} · φ_{i,j}(x,y)的要求，
        系数c_{i,j}使用标准正态分布N(0,1)

        Returns:
            coeff_matrix: (21, 21) 系数矩阵，每个元素服从N(0,1)分布
        """
        # 生成441个系数，使用标准正态分布N(0,1)
        # 这对应21×21=441个基函数的系数c_{i,j}
        coeff_matrix = torch.randn(self.height, self.width)

        return coeff_matrix

    def _generate_bspline_image(self, coeff_matrix, random_seed=None):
        """
        基于基数B样条生成图像 - 使用修正后的数学方法

        根据simple_output_2.py的修正：
        f(x,y) = Σ_{i=0}^{20} Σ_{j=0}^{20} c_{i,j} · φ_{i,j}(x,y)
        其中 φ_{i,j}(x,y) = B₂(x-i) · B₁(y-j)
        修正：使用coeff_matrix.t().flatten()生成图像

        Args:
            coeff_matrix: (21, 21) 系数矩阵 c_{i,j}
            random_seed: 随机种子

        Returns:
            f_image: (128, 128) 生成的图像
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # 修正：使用coeff_matrix.t().flatten()生成图像（与simple_output_2.py一致）
        coefficients_for_image = coeff_matrix.flatten().numpy()  # 441个系数，转置后flatten

        # 使用基数B样条生成器生成图像
        # 实现f(x,y) = Σ_{i=0}^{20} Σ_{j=0}^{20} c_{i,j} · φ_{i,j}(x,y)
        shape = (128, 128)  # 标准CT图像尺寸

        f_image = self.bspline_generator.generate_cardinal_pattern(
            shape=shape,
            coefficients=coefficients_for_image,
            region=((2, 20), (1, 20)),  # 区域E = [2,20] × [1,20]
            enforce_region_constraint=True,  # 在区域E外强制为0
            random_seed=random_seed
        )

        return torch.from_numpy(f_image).float()

    def _tikhonov_init(self, F_observed: torch.Tensor, lambda_reg: float = 0.1) -> torch.Tensor:
        """
        使用 Tikhonov 正则化求解线性系统，生成系数初始估计。
        A = diag(Φ) · G，求解 (AᴴA + λI)d = AᴴF，取实部作为初始系数 (21,21)。
        """
        F_obs = F_observed.to(self.operator.G_matrix.device)
        if not torch.is_complex(F_obs):
            F_obs = torch.complex(F_obs, torch.zeros_like(F_obs))

        A = self.operator.G_matrix * self.operator.phi_diag.unsqueeze(1)  # (441,441) complex
        A_H = torch.conj(A).T
        normal = A_H @ A + lambda_reg * torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
        rhs = A_H @ F_obs

        d_est = torch.linalg.solve(normal, rhs)  # (441,)
        coeff_init = d_est.real.view(self.height, self.width)
        return coeff_init

    def generate_batch(self, batch_size, random_seed=None):
        """生成训练批次：完整流程包括F = Φ G d"""
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        coeff_true_list = []
        f_true_list = []
        F_observed_list = []
        coeff_initial_list = []

        for i in range(batch_size):
            coeff_true, f_true, F_observed, coeff_initial = self.generate_training_sample(
                random_seed=None if random_seed is None else random_seed + i
            )
            coeff_true_list.append(coeff_true)
            f_true_list.append(f_true)
            F_observed_list.append(F_observed)
            coeff_initial_list.append(coeff_initial)

        # 转换为批次张量
        coeff_true_batch = torch.stack(coeff_true_list).unsqueeze(1)  # (B, 1, 21, 21)
        f_true_batch = torch.stack(f_true_list).unsqueeze(1)  # (B, 1, 128, 128)
        F_observed_batch = torch.stack(F_observed_list).unsqueeze(1)  # (B, 1, 441)
        coeff_initial_batch = torch.stack(coeff_initial_list).unsqueeze(1)  # (B, 1, 21, 21)

        return coeff_true_batch, f_true_batch, F_observed_batch, coeff_initial_batch


# ============================================================================
# 基于算法理论的训练器
# ============================================================================
class TheoreticalTrainer:
    """基于算法理论的训练器"""

    def __init__(self):
        # 设置日志系统
        self._setup_logging()

        # 初始化模型
        self.model = initialize_model()

        # 数据生成器
        self.data_generator = TheoreticalDataGenerator()

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=starter_learning_rate,
            weight_decay=1e-4
        )

        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=100,
            min_lr=1e-6
        )

        # 训练状态
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

        # 生成验证数据
        self._generate_validation_data()

        self.logger.info("Theoretical trainer initialized successfully")

    def _setup_logging(self):
        """设置日志系统"""
        log_dir = LOGGING_CONFIG['log_dir']
        os.makedirs(log_dir, exist_ok=True)

        # 创建日志器
        self.logger = logging.getLogger('TheoreticalCTTrainer')
        self.logger.setLevel(getattr(logging, LOGGING_CONFIG['log_level']))

        # 清除现有处理器
        self.logger.handlers.clear()

        # 控制台处理器
        if LOGGING_CONFIG['log_to_console']:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # 文件处理器
        if LOGGING_CONFIG['log_to_file']:
            file_handler = logging.FileHandler(
                os.path.join(log_dir, 'training.log')
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def _generate_validation_data(self):
        """生成固定的验证数据"""
        self.logger.info("Generating validation data...")

        # 生成固定验证数据 - 使用完整的算法理论
        coeff_true_val, f_true_val, F_observed_val, coeff_initial_val = self.data_generator.generate_training_sample(
            random_seed=DATA_CONFIG['validation_seed']
        )

        # 转换为批次张量
        self.coeff_true_val = coeff_true_val.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 21, 21)
        self.f_true_val = f_true_val.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 128, 128)
        self.F_observed_val = F_observed_val.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 441)
        self.coeff_initial_val = coeff_initial_val.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 21, 21)

        self.logger.info(f"Validation data generated: coeff={self.coeff_true_val.shape}, f={self.f_true_val.shape}, F={self.F_observed_val.shape}")

    def _generate_training_batch(self, batch_size=None):
        """生成训练批次"""
        if batch_size is None:
            batch_size = n_data

        return self.data_generator.generate_batch(batch_size, random_seed=self.current_iter)

    def _validate(self):
        """执行验证：基于完整算法理论的验证"""
        self.model.eval()
        with torch.no_grad():
            coeff_pred, history, metrics = self.model(
                self.coeff_initial_val, self.F_observed_val
            )
            # 验证损失：||c_k(pred) - c_k(true)||₂²
            val_loss = nn.functional.mse_loss(coeff_pred, self.coeff_true_val)

        self.model.train()
        return val_loss.item(), metrics

    def _save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'iter': self.current_iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
        }

        # 保存常规检查点
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR, f'checkpoint_iter_{self.current_iter}.pth'
        )
        torch.save(checkpoint, checkpoint_path)

        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, BEST_MODEL_PATH)
            self.logger.info(f"New best model saved with validation loss: {self.best_val_loss:.6f}")

        # 保存最新模型
        torch.save(checkpoint, MODEL_PATH)

    def _log_training_info(self, train_loss, val_loss, metrics, iter_time):
        """记录训练信息"""
        current_lr = self.optimizer.param_groups[0]['lr']

        # 存储历史
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss)
        self.training_history['learning_rate'].append(current_lr)
        self.training_history['data_fidelity_error'].append(metrics['data_fidelity_error'])
        self.training_history['update_difference'].append(metrics['update_difference'])

        # 记录信息
        self.logger.info(
            f"Iter: {self.current_iter:4d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"LR: {current_lr:.6f} | "
            f"Time: {iter_time:.3f}s | "
            f"Data Fidelity Error: {metrics['data_fidelity_error']:.6f} | "
            f"Update Difference: {metrics['update_difference']:.6f}"
        )

    def _save_training_plots(self):
        """保存训练进度图表"""
        if len(self.training_history['train_loss']) == 0:
            return

        fig = None
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # 损失图
            axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
            axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # 学习率图
            axes[0, 1].plot(self.training_history['learning_rate'])
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].grid(True)

            # 数据保真误差
            axes[1, 0].plot(self.training_history['data_fidelity_error'])
            axes[1, 0].set_title('Data Fidelity Error')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Error')
            axes[1, 0].grid(True)

            # 更新差异
            axes[1, 1].plot(self.training_history['update_difference'])
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

    def train(self):
        """主训练循环"""
        self.logger.info("Starting theoretical CT reconstruction training...")
        self.logger.info(f"Total iterations: {n_train}")
        self.logger.info(f"Batch size: {n_data}")
        self.logger.info(f"Model parameters: {count_parameters(self.model):,}")

        total_start_time = time.time()

        for self.current_iter in range(n_train):
            iter_start_time = time.time()

            # 生成训练批次 - 使用完整的算法理论
            coeff_true, f_true, F_observed, coeff_initial = self._generate_training_batch()

            # 移动到设备
            coeff_true = coeff_true.to(device)
            f_true = f_true.to(device)
            F_observed = F_observed.to(device)
            coeff_initial = coeff_initial.to(device)

            # 确保频域数据是复数类型
            if not torch.is_complex(F_observed):
                F_observed = torch.complex(F_observed, torch.zeros_like(F_observed))

            # 清零梯度
            self.optimizer.zero_grad()

            # 前向传播：神经网络预测的系数c_k(pred)
            coeff_pred, history, metrics = self.model(coeff_initial, F_observed)

            # 计算损失：真实系数c_k(true)和神经网络输出结果c_k(pred)的2范数损失
            # loss = ||c_k(pred) - c_k(true)||₂²
            loss = nn.functional.mse_loss(coeff_pred, coeff_true)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if TRAINING_CONFIG.get('gradient_clip_value', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    TRAINING_CONFIG['gradient_clip_value']
                )

            # 更新参数
            self.optimizer.step()

            iter_time = time.time() - iter_start_time

            # 验证
            if self.current_iter % TRAINING_CONFIG['validation_interval'] == 0:
                val_loss, val_metrics = self._validate()

                # 学习率调度
                self.scheduler.step(val_loss)

                # 记录信息
                self._log_training_info(loss.item(), val_loss, val_metrics, iter_time)

                # 检查改进
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1

                # 早停
                if (self.patience_counter >= TRAINING_CONFIG['early_stopping_patience'] and
                    TRAINING_CONFIG['early_stopping_patience'] > 0):
                    self.logger.info(f"Early stopping triggered after {self.current_iter} iterations")
                    break

            # 保存检查点
            if self.current_iter % TRAINING_CONFIG['save_interval'] == 0:
                self._save_checkpoint()

            # 保存训练图表
            if self.current_iter % 500 == 0 and self.current_iter > 0:
                self._save_training_plots()

        total_time = time.time() - total_start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")

        # 最终保存
        self._save_checkpoint()
        self._save_training_plots()

    def load_checkpoint(self, checkpoint_path):
        """从检查点加载训练状态"""
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
    """主训练函数"""
    print("=" * 60)
    print("THEORETICAL CT RECONSTRUCTION TRAINING")
    print("=" * 60)

    # 创建训练器
    trainer = TheoreticalTrainer()

    # 可选择从检查点恢复
    resume_path = None  # 设置检查点路径以恢复训练
    if resume_path and os.path.exists(resume_path):
        trainer.load_checkpoint(resume_path)

    # 开始训练
    trainer.train()

    print("Theoretical training completed successfully!")


if __name__ == '__main__':
    main()
