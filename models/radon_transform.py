"""FFT-based Fourier operator for pixel basis (Phi = B1_x * B1_y)."""

import torch
import numpy as np
from typing import Optional, List

from box_spline import CardinalBSpline2D
from image_generator import DifferentiableImageGenerator
from config import device, DATA_CONFIG, THEORETICAL_CONFIG, IMAGE_SIZE


class NumpyPhantomGenerator:
    """ random ellipse phantom generator (IMAGE_SIZE x IMAGE_SIZE by default)."""

    def __init__(self, size: int = IMAGE_SIZE):
        self.size = size
        # grid in [-1, 1] for both axes
        y, x = np.mgrid[-1:1:complex(0, size), -1:1:complex(0, size)]
        self.X = x
        self.Y = y

    def _random_ellipse_params(self):
        """
        Returns: (intensity, a, b, x0, y0, theta)
        a, b: semi-axes; x0, y0: center; theta: rotation (radians).
        """
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

#   SheppLoganGenerator
class SheppLoganGenerator:
    """Standard Shepp-Logan phantom generator."""

    def __init__(self, size: int = IMAGE_SIZE):
        self.size = size
        y, x = np.mgrid[-1:1:complex(0, size), -1:1:complex(0, size)]
        self.X = x
        self.Y = y
        # intensity, a, b, x0, y0, theta_deg
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

    # *args, **kwargs 是为了兼容 generate_training_sample 的调用方式
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
    """
    Solve (A^H A + lambda I)x = A^H y exploiting diagonal structure of A^H A.
    A = Phi * FFT * Permute, so A^H A is diagonal in permuted Fourier domain.
    """
    if F_observed.dim() == 1:
        F_observed = F_observed.unsqueeze(0)
    Phi = operator.Phi
    N = operator.N
    # A^H y in permuted Fourier domain: N * Phi^H * y
    numerator = N * Phi.conj().unsqueeze(0) * F_observed
    denominator = N * (Phi.abs() ** 2) + lambda_reg
    u = numerator / denominator  # solution in permuted Fourier domain
    d_complex = torch.fft.ifft(u, dim=-1)  # inverse FFT (includes 1/N)
    d_real = d_complex.real
    batch_size = F_observed.shape[0]
    out = torch.zeros(batch_size, N, device=F_observed.device, dtype=d_real.dtype)
    out.scatter_(1, operator.flatten_order.unsqueeze(0).expand(batch_size, -1), d_real)
    return out.view(batch_size, 1, operator.img_size, operator.img_size) if hasattr(operator, "img_size") else out.view(batch_size, 1, int(np.sqrt(N)), int(np.sqrt(N)))


class FourierOperatorCalculator:
    """Compute Phi diagonal and frequency ordering for the FFT operator."""

    def __init__(self, beta=THEORETICAL_CONFIG["beta_vector"], n_coefficients=None, m=2):
        self.beta = torch.tensor(beta, dtype=torch.float32)
        if n_coefficients is None:
            n_coefficients = IMAGE_SIZE * IMAGE_SIZE
        self.n_coefficients = n_coefficients
        self.m = m
        self._compute_k_order()
        self._compute_Phi_matrix()

    def _compute_Phi_matrix(self):
        N = self.n_coefficients
        # Use fftfreq to align Phi with the FFT output ordering (handles negative frequencies in the tail).
        freq = torch.fft.fftfreq(N, d=1.0).to(torch.float64)
        arg_x = (2.0 * np.pi * freq) * self.beta[0].double()
        arg_y = (2.0 * np.pi * freq) * self.beta[1].double()
        bspline = CardinalBSpline2D()
        B1_x = np.asarray(bspline.B1_hat_complex(arg_x.numpy()), dtype=np.complex128)
        B1_y = np.asarray(bspline.B1_hat_complex(arg_y.numpy()), dtype=np.complex128)
        Phi = B1_x * B1_y
        self.Phi_diagonal = torch.from_numpy(Phi).to(torch.complex128)

    def _compute_k_order(self):
        side = int(np.sqrt(self.n_coefficients))
        xs = torch.arange(side, dtype=torch.float64)
        ys = torch.arange(side, dtype=torch.float64)
        self.X, self.Y = torch.meshgrid(xs, ys, indexing="xy")
        beta_vec = self.beta.double()
        beta_dot = (beta_vec[0] * self.X + beta_vec[1] * self.Y).reshape(-1)
        self.flatten_order = torch.argsort(beta_dot).to(torch.long)


class TheoreticalDataGenerator:
    """
    Use FFT to apply F = Phi * FFT(d) and a direct diagonal Tikhonov init,
    with coeffs generated from a random ellipse phantom .
    """

    def __init__(self):
        self.img_size = IMAGE_SIZE
        self.N = self.img_size * self.img_size

        # Load Noise Configuration
        self.noise_mode = DATA_CONFIG.get("noise_mode", "snr")
        self.noise_level = DATA_CONFIG.get("noise_level", 0.05) # delta
        self.target_snr_db = DATA_CONFIG.get("target_snr_db", 60.0)

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
        """A(d) = Phi * FFT(sort(d))."""
        if coeff_matrix.dim() == 3:
            coeff_matrix = coeff_matrix.unsqueeze(0)
        b = coeff_matrix.shape[0]
        coeff_flat = coeff_matrix.view(b, -1)
        d_ordered = coeff_flat[:, self.flatten_order].to(torch.complex64)
        Gd = torch.fft.fft(d_ordered, dim=-1)
        return Gd * self.Phi.unsqueeze(0)

    def adjoint_operator(self, residual: torch.Tensor) -> torch.Tensor:
        """A^H(r) = unsort(IFFT(Phi^H * r) * N)."""
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

            # --- NOISE GENERATION LOGIC ---
            if self.noise_mode == "multiplicative":
                # F_obs = F + delta * F * rand, where rand ~ U[-1, 1]
                # Assuming rand is real-valued multiplicative noise (scaling)
                rand_u = 2.0 * torch.rand_like(F_clean.real) - 1.0 # Uniform [-1, 1]
                noise = self.noise_level * F_clean * rand_u
                F_observed = F_clean + noise


            elif self.noise_mode == "additive":
                # F_obs = F + delta * n, where n ~ N(0, 1)
                mean_amplitude = torch.mean(torch.abs(F_clean))
                # Standard normal noise (assuming independent real/imag parts for complex signal)
                # Note: standard randn has std=1.
                noise_real = torch.randn_like(F_clean.real)
                noise_imag = torch.randn_like(F_clean.imag)
                noise = self.noise_level  * (noise_real + 1j * noise_imag)
                F_observed = F_clean + noise

            elif self.noise_mode == "snr" and self.target_snr_db is not None:
                # Legacy SNR based noise
                signal_energy = torch.sum(torch.abs(F_clean) ** 2)
                M = F_clean.numel()
                sigma_squared = signal_energy / (M * (10 ** (self.target_snr_db / 10.0)))
                sigma = torch.sqrt(sigma_squared)
                noise_std = sigma / 1.41421356  # split variance across real/imag parts
                noise_real = torch.randn_like(F_clean.real) * noise_std
                noise_imag = torch.randn_like(F_clean.imag) * noise_std
                noise = noise_real + 1j * noise_imag
                F_observed = F_clean + noise
            else:
                # No noise
                F_observed = F_clean
            # ------------------------------

        lam = lambda_reg if lambda_reg is not None else DATA_CONFIG.get("lambda_reg", 0.1)
        coeff_initial = direct_tikhonov_solve(self, F_observed, lambda_reg=lam).real

        return (
            coeff_true.squeeze(0).squeeze(0),
            f_true.squeeze(0),
            F_observed.squeeze(0),
            coeff_initial.squeeze(0).squeeze(0),
        )

    def generate_batch(self, batch_size, random_seed=None, lambda_reg: float = None):
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