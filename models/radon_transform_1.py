"""FFT-based Fourier operator with rigorous Periodic Summation logic."""

import torch
import numpy as np
from typing import Optional, List

from box_spline import CardinalBSpline2D
from image_generator import DifferentiableImageGenerator
from config import device, DATA_CONFIG, THEORETICAL_CONFIG, IMAGE_SIZE


class NumpyPhantomGenerator:
    # ... (保持不变，省略以节省空间) ...
    def __init__(self, size: int = IMAGE_SIZE):
        self.size = size
        y, x = np.mgrid[-1:1:complex(0, size), -1:1:complex(0, size)]
        self.X = x
        self.Y = y

    def _random_ellipse_params(self):
        intensity = (np.random.rand() - 0.3) * np.random.exponential(0.3)
        a = max(np.random.exponential(0.2), 0.01)
        b = max(np.random.exponential(0.2), 0.01)
        x0 = np.random.rand() - 0.5
        y0 = np.random.rand() - 0.5
        theta = np.random.rand() * 2 * np.pi
        return intensity, a, b, x0, y0, theta

    def generate(self, n_ellipses: Optional[int] = None) -> np.ndarray:
        if n_ellipses is None:
            n_ellipses = np.random.poisson(100)
        phantom = np.zeros((self.size, self.size), dtype=np.float32)
        for _ in range(n_ellipses):
            intensity, a, b, x0, y0, theta = self._random_ellipse_params()
            dx = self.X - x0
            dy = self.Y - y0
            x_rot = dx * np.cos(theta) + dy * np.sin(theta)
            y_rot = -dx * np.sin(theta) + dy * np.cos(theta)
            mask = (x_rot ** 2) / (a ** 2) + (y_rot ** 2) / (b ** 2) <= 1.0
            phantom[mask] += intensity
        return phantom


class SheppLoganGenerator:
    # ... (保持不变，省略以节省空间) ...
    def __init__(self, size: int = IMAGE_SIZE):
        self.size = size
        y, x = np.mgrid[-1:1:complex(0, size), -1:1:complex(0, size)]
        self.X = x
        self.Y = y
        self.ellipses: List[List[float]] = [
            [1.0, 0.69, 0.92, 0.0, 0.0, 0.0],
            [-0.8, 0.6624, 0.874, 0.0, -0.0184, 0.0],
            [-0.2, 0.11, 0.31, 0.22, 0.0, -18.0],
            [-0.2, 0.16, 0.41, -0.22, 0.0, 18.0],
            [0.1, 0.21, 0.25, 0.0, 0.35, 0.0],
            [0.1, 0.046, 0.046, 0.0, 0.1, 0.0],
            [0.1, 0.046, 0.046, 0.0, -0.1, 0.0],
            [0.1, 0.046, 0.023, -0.08, -0.605, 0.0],
            [0.1, 0.023, 0.023, 0.0, -0.606, 0.0],
            [0.1, 0.023, 0.046, 0.06, -0.605, 0.0],
        ]

    def generate(self, *args, **kwargs) -> np.ndarray:
        phantom = np.zeros((self.size, self.size), dtype=np.float32)
        for intensity, a, b, x0, y0, angle_deg in self.ellipses:
            theta = np.radians(angle_deg)
            dx = self.X - x0
            dy = self.Y - y0
            x_rot = dx * np.cos(theta) + dy * np.sin(theta)
            y_rot = -dx * np.sin(theta) + dy * np.cos(theta)
            mask = (x_rot ** 2) / (a ** 2) + (y_rot ** 2) / (b ** 2) <= 1.0
            phantom[mask] += intensity
        return phantom


def direct_tikhonov_solve(operator, F_observed: torch.Tensor, lambda_reg: float = 0.1) -> torch.Tensor:
    if F_observed.dim() == 1:
        F_observed = F_observed.unsqueeze(0)
    Phi = operator.Phi
    N = operator.N
    numerator = N * Phi.conj().unsqueeze(0) * F_observed
    denominator = N * (Phi.abs() ** 2) + lambda_reg
    u = numerator / denominator
    d_complex = torch.fft.ifft(u, dim=-1)
    d_real = d_complex.real
    batch_size = F_observed.shape[0]
    out = torch.zeros(batch_size, N, device=F_observed.device, dtype=d_real.dtype)
    out.scatter_(1, operator.flatten_order.unsqueeze(0).expand(batch_size, -1), d_real)
    return out.view(batch_size, 1, operator.img_size, operator.img_size)


class FourierOperatorCalculator:
    """Compute Phi with strict Periodization/Aliasing logic."""

    def __init__(self, beta=THEORETICAL_CONFIG["beta_vector"], n_coefficients=None, m=2):
        # 严格使用原始 beta (1, 256)，不搞两套参数
        self.beta = torch.tensor(beta, dtype=torch.float32)

        if n_coefficients is None:
            n_coefficients = IMAGE_SIZE * IMAGE_SIZE
        self.n_coefficients = n_coefficients
        self.m = m
        self._compute_k_order()
        self._compute_Phi_matrix()

    def _compute_Phi_matrix(self):
        """
        利用泊松求和公式的离散推论：
        计算高频采样点的物理响应时，应计算其折叠回基带（Baseband）的响应。
        """
        N = self.n_coefficients
        # 1. 原始采样频率 (归一化单位)
        freq = torch.fft.fftfreq(N, d=1.0).to(torch.float64)

        # 2. 严格按照理论公式计算高频坐标
        # Arg = freq * beta
        # 这里会产生很大的值 (例如 256.0, 512.0 等)
        arg_x_high = freq * self.beta[0].double()
        arg_y_high = freq * self.beta[1].double()

        # 3. 【理论修正核心】 频谱折叠 (Aliasing / Periodization)
        # 将高频坐标映射回 [-0.5, 0.5] 的基带区间
        # 这一步在数学上等效于：认为基函数响应 \hat{\phi} 是周期的
        # torch.remainder(x + 0.5, 1.0) - 0.5 是标准的 wrap-around 操作
        arg_x_folded = torch.remainder(arg_x_high + 0.5, 1.0) - 0.5
        arg_y_folded = torch.remainder(arg_y_high + 0.5, 1.0) - 0.5

        # 4. 计算基带上的 Sinc 响应
        # 我们的基函数是 B-Spline (Box)，其基带响应就是 Sinc
        # 注意：这里我们手动计算 Sinc 以确保完全掌控
        # Sinc(x) = sin(pi*x) / (pi*x)
        # 这里的 arg_folded 已经是归一化频率
        phi_x = torch.sinc(arg_x_folded)
        phi_y = torch.sinc(arg_y_folded)

        # 5. 相位校正 (可选，但推荐)
        # 加上线性相位项以保持 shift-invariant 特性
        phase_x = torch.exp(-1j * np.pi * arg_x_folded)
        phase_y = torch.exp(-1j * np.pi * arg_y_folded)

        # 组合
        Phi = (phi_x * phase_x) * (phi_y * phase_y)
        self.Phi_diagonal = Phi.to(torch.complex128)

    def _compute_k_order(self):
        side = int(np.sqrt(self.n_coefficients))
        xs = torch.arange(side, dtype=torch.float64)
        ys = torch.arange(side, dtype=torch.float64)
        self.X, self.Y = torch.meshgrid(xs, ys, indexing="xy")

        # 使用原始 beta 排序
        beta_vec = self.beta.double()
        beta_dot = (beta_vec[0] * self.X + beta_vec[1] * self.Y).reshape(-1)
        self.flatten_order = torch.argsort(beta_dot).to(torch.long)


class TheoreticalDataGenerator:
    # ... (保持不变，与前文一致) ...
    def __init__(self):
        self.img_size = IMAGE_SIZE
        self.N = self.img_size * self.img_size
        self.target_snr_db = DATA_CONFIG.get("target_snr_db", None)
        self.phantom_gen = NumpyPhantomGenerator(size=self.img_size)
        self.image_gen = DifferentiableImageGenerator(image_size=self.img_size).to(device)

        self.fourier_calculator = FourierOperatorCalculator(
            beta=THEORETICAL_CONFIG['beta_vector'],
            n_coefficients=self.N,
            m=2,
        )

        self.Phi = self.fourier_calculator.Phi_diagonal.to(device=device, dtype=torch.complex64)
        self.flatten_order = self.fourier_calculator.flatten_order.to(device)

    def forward_operator(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        if coeff_matrix.dim() == 3:
            coeff_matrix = coeff_matrix.unsqueeze(0)
        b = coeff_matrix.shape[0]
        coeff_flat = coeff_matrix.view(b, -1)
        d_ordered = coeff_flat[:, self.flatten_order].to(torch.complex64)
        Gd = torch.fft.fft(d_ordered, dim=-1)
        return Gd * self.Phi.unsqueeze(0)

    def adjoint_operator(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.dim() == 3:
            residual = residual.squeeze(1)
        PhiH_R = residual * self.Phi.conj().unsqueeze(0)
        d_complex = torch.fft.ifft(PhiH_R, dim=-1) * self.N
        d_real = d_complex.real
        b = residual.shape[0]
        out = torch.zeros(b, self.N, device=residual.device, dtype=d_real.dtype)
        out.scatter_(1, self.flatten_order.unsqueeze(0).expand(b, -1), d_real)
        return out.view(b, 1, self.img_size, self.img_size)

    def generate_training_sample(self, random_seed=None, lambda_reg: float = None):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        phantom_np = self.phantom_gen.generate()
        coeff_true = torch.from_numpy(phantom_np).float().unsqueeze(0).unsqueeze(0).to(device)
        f_true = self.image_gen(coeff_true).squeeze(0)

        with torch.no_grad():
            F_clean = self.forward_operator(coeff_true)
            if self.target_snr_db is not None:
                signal_energy = torch.sum(torch.abs(F_clean) ** 2)
                M = F_clean.numel()
                sigma_squared = signal_energy / (M * (10 ** (self.target_snr_db / 10.0)))
                sigma = torch.sqrt(sigma_squared)
                noise_std = sigma / 1.41421356
                noise_real = torch.randn_like(F_clean.real) * noise_std
                noise_imag = torch.randn_like(F_clean.imag) * noise_std
                noise = noise_real + 1j * noise_imag
                F_observed = F_clean + noise
            else:
                F_observed = F_clean

        lam = lambda_reg if lambda_reg is not None else DATA_CONFIG.get("lambda_reg", 0.1)
        coeff_initial = direct_tikhonov_solve(self, F_observed, lambda_reg=lam).real

        return (
            coeff_true.squeeze(0).squeeze(0),
            f_true.squeeze(0),
            F_observed.squeeze(0),
            coeff_initial.squeeze(0).squeeze(0),
        )

    def generate_batch(self, batch_size, random_seed=None, lambda_reg: float = None):
        # ... (保持不变) ...
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        coeff_true_list = []
        f_true_list = []
        F_observed_list = []
        coeff_initial_list = []
        start_seed = random_seed if random_seed is not None else 0
        for i in range(batch_size):
            c_true, f_true, F_obs, c_init = self.generate_training_sample(
                random_seed=start_seed + i if random_seed is not None else None,
                lambda_reg=lambda_reg,
            )
            coeff_true_list.append(c_true)
            f_true_list.append(f_true)
            F_observed_list.append(F_obs)
            coeff_initial_list.append(c_init)
        coeff_true_batch = torch.stack(coeff_true_list).unsqueeze(1)
        f_true_batch = torch.stack(f_true_list).unsqueeze(1)
        F_observed_batch = torch.stack(F_observed_list)
        coeff_initial_batch = torch.stack(coeff_initial_list).unsqueeze(1)
        return coeff_true_batch, f_true_batch, F_observed_batch, coeff_initial_batch