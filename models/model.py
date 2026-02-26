import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import device, THEORETICAL_CONFIG, TRAINING_CONFIG, IMAGE_SIZE
from radon_transform import FourierOperatorCalculator


# ============================================================================
# 1. Coefficient mapping (row-major flatten)
# ============================================================================
class CoefficientMapping:
    def __init__(self, beta=THEORETICAL_CONFIG["beta_vector"], E_plus_shape=(IMAGE_SIZE, IMAGE_SIZE)):
        self.beta = torch.tensor(beta, dtype=torch.long)
        self.beta_norm = torch.norm(self.beta.float(), p=2)
        self.alpha = self.beta.float() / self.beta_norm
        self.E_plus_shape = E_plus_shape
        self.height, self.width = E_plus_shape
        self.N = self.height * self.width
        self._create_one_to_one_mapping()

    def _create_one_to_one_mapping(self):
        self.k_to_d_mapping = {}
        self.d_to_k_mapping = {}
        for kx in range(self.height):
            for ky in range(self.width):
                k = (kx, ky)
                d_index = kx * self.width + ky
                self.k_to_d_mapping[k] = d_index
                self.d_to_k_mapping[d_index] = k

    def coeff_to_vector(self, coeff_matrix):
        return coeff_matrix.flatten()

    def vector_to_coeff(self, d_vector):
        return d_vector.view(self.height, self.width)

    def flatten_batch(self, coeff_batch):
        return coeff_batch.view(coeff_batch.shape[0], -1)

    def unflatten_batch(self, d_batch):
        return d_batch.view(d_batch.shape[0], 1, self.height, self.width)

    def verify_mapping_consistency(self):
        coeff_matrix = torch.randn(self.E_plus_shape)
        d_vector = self.coeff_to_vector(coeff_matrix)
        recovered_coeff = self.vector_to_coeff(d_vector)
        error = torch.norm(coeff_matrix - recovered_coeff)
        return error.item()


# ============================================================================
# 2. FFT-based frequency-domain operator (Phi diag + FFT)
# ============================================================================
class RadonFourierOperator2D(nn.Module):
    """
    FFT-based A = Phi * FFT(d) for pixel basis (B1*B1). Provides forward and adjoint.
    """

    def __init__(self, beta=THEORETICAL_CONFIG["beta_vector"], height=IMAGE_SIZE, width=IMAGE_SIZE, m=2, normalize=True):
        super().__init__()
        self.height = height
        self.width = width
        self.N = height * width
        calc = FourierOperatorCalculator(beta=beta, n_coefficients=self.N, m=m)
        self.register_buffer("Phi", calc.Phi_diagonal.to(device=device, dtype=torch.complex64))
        self.register_buffer("flatten_order", calc.flatten_order.to(device=device, dtype=torch.long))
        self.scale_factor = 1.0

    def forward(self, coeff_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward: d -> F_pred (complex), shape (B, N).
        """
        if coeff_matrix.dim() == 3:
            coeff_matrix = coeff_matrix.unsqueeze(1)
        batch_size = coeff_matrix.shape[0]
        coeff_flat = coeff_matrix.view(batch_size, -1)
        d_ordered = coeff_flat[:, self.flatten_order].to(torch.complex64)
        F_pred = torch.fft.fft(d_ordered, dim=-1) * self.Phi.unsqueeze(0)
        return F_pred * self.scale_factor

    def adjoint(self, residual: torch.Tensor) -> torch.Tensor:
        """
        Adjoint: Residual -> Re{A^H Residual}, returns shape (B,1,H,W).
        """
        if residual.dim() == 3 and residual.shape[1] == 1:
            residual = residual.squeeze(1)
        res_norm = residual * self.scale_factor
        PhiH_R = res_norm * self.Phi.conj().unsqueeze(0)
        d_grad_complex = torch.fft.ifft(PhiH_R, dim=-1) * self.N
        d_grad_real = d_grad_complex.real
        batch_size = residual.shape[0]
        grad_output = torch.zeros(batch_size, self.N, device=residual.device, dtype=d_grad_real.dtype)
        grad_output.scatter_(1, self.flatten_order.unsqueeze(0).expand(batch_size, -1), d_grad_real)
        return grad_output.view(batch_size, 1, self.height, self.width)


# ============================================================================
# 3. Theoretical gradient descent
# ============================================================================
class TheoreticalGradientDescent(nn.Module):
    def __init__(self, beta=THEORETICAL_CONFIG["beta_vector"], height=IMAGE_SIZE, width=IMAGE_SIZE,
                 regularizer_type='tikhonov', lambda_reg=0.01):
        super().__init__()
        self.operator = RadonFourierOperator2D(beta, height, width)
        self.regularizer_type = regularizer_type
        self.lambda_reg = lambda_reg
        self.step_size = 1e-2
        self.register_buffer('laplace_kernel', torch.tensor(
            [[0.0, -1.0, 0.0],
             [-1.0, 4.0, -1.0],
             [0.0, -1.0, 0.0]]
        ).view(1, 1, 3, 3))

    def compute_data_fidelity_gradient(self, coeff_matrix, F_observed):
        if F_observed.dim() == 3 and F_observed.shape[1] == 1:
            F_observed = F_observed.squeeze(1)
        F_pred = self.operator(coeff_matrix)
        F_obs = F_observed.to(dtype=F_pred.dtype)
        residual = F_pred - F_obs
        gradient = self.operator.adjoint(residual)
        return 2.0 * gradient

    def compute_regularization_gradient(self, coeff_matrix):
        if self.regularizer_type == 'dirichlet':
            return self._dirichlet_gradient(coeff_matrix)
        elif self.regularizer_type == 'tikhonov':
            return 2 * coeff_matrix
        elif self.regularizer_type == 'tv':
            return self._tv_gradient(coeff_matrix)
        else:
            return torch.zeros_like(coeff_matrix)

    def _tv_gradient(self, coeff_matrix):
        eps = coeff_matrix.new_tensor(1e-6)
        grad_x, grad_y = self._forward_gradient(coeff_matrix)
        grad_norm = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + eps)
        grad_x_norm = grad_x / grad_norm
        grad_y_norm = grad_y / grad_norm
        div_grad = self._divergence(grad_x_norm, grad_y_norm)
        return div_grad

    def _dirichlet_gradient(self, coeff_matrix):
        padded = F.pad(coeff_matrix, (1, 1, 1, 1), mode='replicate')
        return F.conv2d(padded, self.laplace_kernel.to(coeff_matrix), padding=0)

    def _forward_gradient(self, x):
        grad_x = torch.zeros_like(x)
        grad_y = torch.zeros_like(x)
        grad_y[:, :, :-1, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
        grad_x[:, :, :, :-1] = x[:, :, :, 1:] - x[:, :, :, :-1]
        return grad_x, grad_y

    def _divergence(self, grad_x, grad_y):
        div = torch.zeros_like(grad_x)
        div[:, :, 0, :] += grad_y[:, :, 0, :]
        div[:, :, 1:, :] += grad_y[:, :, 1:, :] - grad_y[:, :, :-1, :]
        div[:, :, :, 0] += grad_x[:, :, :, 0]
        div[:, :, :, 1:] += grad_x[:, :, :, 1:] - grad_x[:, :, :, :-1]
        return div

    def gradient_descent_step(self, coeff_matrix, F_observed):
        data_grad = self.compute_data_fidelity_gradient(coeff_matrix, F_observed)
        reg_grad = self.compute_regularization_gradient(coeff_matrix)
        total_grad = data_grad + self.lambda_reg * reg_grad
        updated_coeff = coeff_matrix - self.step_size * total_grad
        return updated_coeff


# ============================================================================
# 4. Learned gradient descent (CNN updates)
# ============================================================================
class LearnedGradientDescent(nn.Module):
    def __init__(self, beta=THEORETICAL_CONFIG["beta_vector"], height=IMAGE_SIZE, width=IMAGE_SIZE,
                 regularizer_type='tikhonov', n_iter=10, n_memory=5):
        super().__init__()
        self.n_iter = n_iter
        self.n_memory = n_memory
        self.height = height
        self.width = width
        init_image_w = TRAINING_CONFIG.get('image_loss_weight', 0.0)
        self.use_image_loss = init_image_w > 0
        self.operator = RadonFourierOperator2D(beta, height, width)
        self.theoretical_gd = TheoreticalGradientDescent(
            beta, height, width, regularizer_type
        )
        self.update_network = self._build_update_network()
        self.reg_lambda = nn.Parameter(torch.tensor(1e-3))
        self.step_size = nn.Parameter(torch.tensor(1e-3))
        init_s_coeff = torch.tensor(0.0)
        if self.use_image_loss:
            init_s_image = torch.tensor(-np.log(init_image_w))
        else:
            init_s_image = torch.tensor(0.0)
        self.loss_params = nn.Parameter(torch.stack([init_s_coeff, init_s_image]).float())

    def _build_update_network(self):
        input_channels = 3 + self.n_memory
        # 增加通道数到 64 以提升表达能力

        return nn.Sequential(
            # --- 输入归一化 ---
            nn.InstanceNorm2d(input_channels, affine=True),

            # --- 特征提取 (感受野 3x3) ---
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64, affine=True),  # 层间归一化
            nn.ReLU(inplace=True),

            # --- 扩大感受野 (Dilation=2, 感受野 7x7) ---
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),

            # --- 再次扩大 (Dilation=4, 感受野 15x15) ---
            # 这一层对消除大范围条纹伪影至关重要
            nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),

            # [新增] 增加一层 dilation=8，感受野增加 16
            nn.Conv2d(64, 64, kernel_size=3, padding=8, dilation=8),
            nn.InstanceNorm2d(64, affine=True), nn.ReLU(inplace=True),

            # --- 输出层 ---
            nn.Conv2d(64, 1 + self.n_memory, kernel_size=3, padding=1),
        )

    def forward(self, coeff_initial, F_observed):
        if F_observed.dim() == 3 and F_observed.shape[1] == 1:
            F_observed = F_observed.squeeze(1)
        batch_size = coeff_initial.shape[0]
        coeff_current = coeff_initial.clone()
        memory = torch.zeros(batch_size, self.n_memory, self.height, self.width,
                             device=coeff_initial.device)
        history = [coeff_current.clone()]

        for _ in range(self.n_iter):
            data_grad = self.theoretical_gd.compute_data_fidelity_gradient(
                coeff_current, F_observed
            )
            reg_grad = self.theoretical_gd.compute_regularization_gradient(
                coeff_current
            ) * self.reg_lambda

            cnn_input = torch.cat([
                coeff_current,
                data_grad,
                reg_grad,
                memory
            ], dim=1)

            cnn_output = self.update_network(cnn_input)
            raw_update = cnn_output[:, 0:1, :, :]
            new_memory = cnn_output[:, 1:, :, :]
            learned_update = raw_update * self.step_size
            coeff_current = coeff_current - learned_update
            memory = torch.relu(new_memory)
            history.append(coeff_current.clone())

        return coeff_current, history

    def get_theoretical_vs_learned_updates(self, coeff_current, F_observed):
        if F_observed.dim() == 3 and F_observed.shape[1] == 1:
            F_observed = F_observed.squeeze(1)
        theoretical_update = self.theoretical_gd.gradient_descent_step(
            coeff_current, F_observed
        ) - coeff_current
        data_grad = self.theoretical_gd.compute_data_fidelity_gradient(
            coeff_current, F_observed
        )
        reg_grad = self.theoretical_gd.compute_regularization_gradient(
            coeff_current
        ) * self.reg_lambda
        memory = torch.zeros(coeff_current.shape[0], self.n_memory,
                             self.height, self.width, device=coeff_current.device)
        cnn_input = torch.cat([coeff_current, data_grad, reg_grad, memory], dim=1)
        cnn_output = self.update_network(cnn_input)
        learned_update = cnn_output[:, 0:1, :, :] * self.step_size
        return theoretical_update, learned_update


# ============================================================================
# 5. Full CT network
# ============================================================================
class TheoreticalCTNet(nn.Module):
    def __init__(self, beta=THEORETICAL_CONFIG["beta_vector"], height=IMAGE_SIZE, width=IMAGE_SIZE,
                 regularizer_type='tikhonov', n_iter=10, n_memory=5):
        super().__init__()
        self.beta = beta
        self.height = height
        self.width = width
        init_image_w = TRAINING_CONFIG.get('image_loss_weight', 0.0) if 'TRAINING_CONFIG' in globals() else 0.0
        self.use_image_loss = init_image_w > 0
        init_s_coeff = torch.tensor(0.0)
        init_s_image = torch.tensor(-np.log(init_image_w)) if self.use_image_loss else torch.tensor(0.0)
        self.loss_params = nn.Parameter(torch.stack([init_s_coeff, init_s_image]).float())
        self.optimizer = LearnedGradientDescent(
            beta, height, width, regularizer_type, n_iter, n_memory
        )
        self.mapping = CoefficientMapping(beta, (height, width))

    def forward(self, coeff_initial, F_observed):
        if F_observed.dim() == 3 and F_observed.shape[1] == 1:
            F_observed = F_observed.squeeze(1)
        coeff_final, history = self.optimizer(coeff_initial, F_observed)
        metrics = self._compute_optimization_metrics(
            coeff_initial, coeff_final, F_observed, history
        )
        return coeff_final, history, metrics

    def _compute_optimization_metrics(self, coeff_initial, coeff_final, F_observed, history):
        metrics = {}
        # [关键修改] 强制整个 metrics 计算都在 no_grad 下进行
        # 这样就不会占用显存去存梯度，也不会拖慢训练速度
        with torch.no_grad():
            F_final = self.optimizer.operator(coeff_final)
            data_fidelity_error = torch.norm(F_final - F_observed, dim=-1).mean()
            metrics['data_fidelity_error'] = data_fidelity_error.item()

            coeff_change = torch.norm(coeff_final - coeff_initial, dim=(2, 3)).mean()
            metrics['coefficient_change'] = coeff_change.item()

            if self.optimizer.theoretical_gd.regularizer_type == 'tikhonov':
                reg_value = torch.norm(coeff_final, dim=(2, 3)) ** 2
                metrics['regularization_value'] = reg_value.mean().item()
            elif self.optimizer.theoretical_gd.regularizer_type == 'dirichlet':
                grad_y = torch.diff(coeff_final, dim=2, prepend=coeff_final[:, :, -1:])
                grad_x = torch.diff(coeff_final, dim=3, prepend=coeff_final[:, :, :, -1:])
                reg_value = 0.5 * (grad_x.pow(2) + grad_y.pow(2)).sum(dim=(2, 3))
                metrics['regularization_value'] = reg_value.mean().item()

            # 这部分计算非常重，只在验证时算，或者每隔很久算一次
            # 为了极速训练，建议先注释掉，或者只在 cpu 上算
            # theoretical_update, learned_update = self.optimizer.get_theoretical_vs_learned_updates(
            #    coeff_final, F_observed
            # )
            # update_difference = torch.norm(theoretical_update - learned_update)
            metrics['update_difference'] = 0.0  # update_difference.item()

        return metrics


# ============================================================================
# 6. Helpers
# ============================================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_model():
    beta = THEORETICAL_CONFIG['beta_vector']
    regularizer_type = THEORETICAL_CONFIG['regularizer_type']
    n_iter = THEORETICAL_CONFIG['n_iter']
    n_memory = THEORETICAL_CONFIG['n_memory_units']

    model = TheoreticalCTNet(
        beta=beta,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        regularizer_type=regularizer_type,
        n_iter=n_iter,
        n_memory=n_memory
    ).to(device)

    print(f"Model initialized on device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Using theoretical GD block")
    print(f"Regularizer type: {regularizer_type}")
    print(f"Optimization iterations: {n_iter}")
    print(f"Memory units: {n_memory}")

    return model


if __name__ == "__main__":
    model = initialize_model()
    batch_size = 2
    beta = THEORETICAL_CONFIG['beta_vector']
    mapping = CoefficientMapping(beta)
    N = mapping.N
    x_0 = torch.randn(batch_size, 1, IMAGE_SIZE, IMAGE_SIZE).to(device)
    y_fake = torch.randn(batch_size, N, dtype=torch.cfloat).to(device)
    with torch.no_grad():
        coeff_pred, history, metrics = model(x_0, y_fake)
        print(f"input shape: {x_0.shape}")
        print(f"output shape: {coeff_pred.shape}")
        print(f"observed shape: {y_fake.shape}")
        print(f"iterations: {len(history)-1}")
        print(f"data fidelity error: {metrics['data_fidelity_error']:.6f}")
        print(f"update difference: {metrics['update_difference']:.6f}")
        mapping_error = mapping.verify_mapping_consistency()
        print(f"mapping error: {mapping_error:.6f} (should be ~0)")
    print("Simplified LGD model test successful!")
