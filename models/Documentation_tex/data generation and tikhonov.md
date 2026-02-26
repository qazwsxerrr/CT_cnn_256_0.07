核心修改在于 **数据生成（Data Generation）** 和 **Tikhonov 初始化求解** 两个部分。我们需要放弃通过空间域 Radon 变换生成数据的路径，转而直接使用频域公式 $\mathbf{F} = \Phi G \mathbf{d}$，并严格保证 Tikhonov 求解时的 **双精度（Float64/Complex128）** 运算。

以下是具体的修改方案，主要集中在 `train.py` 文件中。



### 修改重点摘要



1. **数据生成 (generate_training_sample)**：
   - **移除**：`radon_simulator.forward_radon_transform` 和 `compute_F_from_radon`。
   - **新增**：直接利用 `fourier_calculator` 将系数 $\mathbf{d}$ 映射为 $\mathbf{F}$。公式：$F_{clean} = \Phi \cdot (G \cdot d)$。
2. **Tikhonov 求解 (_tikhonov_init)**：
   - **精度强制**：将所有涉及矩阵 $A$ 的张量强制转换为 `torch.complex128`。
   - **显式构建 A**：构造 $A = \Phi G$。
   - **解析解**：使用 `torch.linalg.solve` 求解 $(A^H A + \lambda I) d = A^H F$。

------



### 代码修改：`train.py`



请用以下代码替换 `train.py` 中的 `TheoreticalDataGenerator` 类。

Python



```
class TheoreticalDataGenerator:
    """
    Data generator based on theoretical model (F = Phi G d).
    Strictly follows '算法说明.md' for high-precision Tikhonov inversion.
    """

    def __init__(self):
        self.beta = THEORETICAL_CONFIG['beta_vector']
        self.height, self.width = 21, 21
        self.noise_level = DATA_CONFIG['noise_level']
        
        # 即使不再通过空间域生成F，我们仍保留 image_generator 用于计算Loss和可视化
        self.bspline_generator = CardinalBSpline2D()
        self.image_gen = DifferentiableImageGenerator(image_size=256).to(device)

        # 核心计算器
        self.fourier_calculator = FourierOperatorCalculator(
            beta=self.beta,
            n_coefficients=441,
            m=2
        )

        # 构建算子 (这里主要为了获取 flatten_order 等辅助信息)
        self.theoretical_operator, self.operator_norm = self.fourier_calculator.build_operator(
            normalize=True, target_device=device
        )
        self.flatten_order = self.fourier_calculator.flatten_order.to(device)
        self.unflatten_order = self.fourier_calculator.unflatten_order.to(device)
        
        # 预先获取 Phi 和 G 矩阵，并转为双精度 (Complex128) 以备 Tikhonov 使用
        # 注意：fourier_calculator 中的矩阵默认可能是 complex64，这里强制提升精度
        with torch.no_grad():
            # Phi是对角阵向量 (N,)
            self.Phi_diag_high_prec = self.fourier_calculator.Phi_diagonal.to(device=device, dtype=torch.complex128)
            # G是DFT矩阵 (N, N)
            self.G_matrix_high_prec = self.fourier_calculator.G_matrix_2d.to(device=device, dtype=torch.complex128)
            
            # 预计算系统矩阵 A = Phi * G
            # 利用广播机制: (N, 1) * (N, N) -> (N, N)
            self.A_matrix_high_prec = self.Phi_diag_high_prec.unsqueeze(1) * self.G_matrix_high_prec
            
            # 预计算 A_H (共轭转置)
            self.A_H_high_prec = self.A_matrix_high_prec.conj().T

    def generate_training_sample(self, random_seed=None):
        """
        生成流程遵循: d (Random) -> F_clean = Phi G d -> F_observed (Add Noise) -> d_init (Tikhonov)
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        # 1. 生成真值系数 (Spatial 2D)
        coeff_true = self._generate_normal_coefficients().to(device) # (21, 21)
        
        # 2. 转换为 d 向量 (1D, Sorted by beta dot product)
        # 这一步对应文档中的 "坐标排序与向量化"
        d_vector = self.fourier_calculator.compute_d_from_c(coeff_true).squeeze(0) # (441,)

        # 3. 计算理论频域数据 F = Phi G d
        # 使用双精度计算以保证物理模型的准确性，最后转回 complex64 适配神经网络输入
        d_vector_db = d_vector.to(dtype=torch.complex128)
        
        # F = A * d
        F_clean_db = torch.matmul(self.A_matrix_high_prec, d_vector_db)
        F_clean = F_clean_db.to(dtype=torch.complex64)

        # 4. 添加噪声
        if self.noise_level > 0:
            # 噪声模型：按照文档描述，幅度基于 mean(|F_clean|)
            mean_amplitude = torch.mean(torch.abs(F_clean))
            noise = self.noise_level * mean_amplitude * torch.randn_like(F_clean)
            F_observed = F_clean + noise
        else:
            F_observed = F_clean

        # 5. Tikhonov 正则化求解初始值 (高精度求解)
        coeff_initial = self._tikhonov_init(F_observed, lambda_reg=0.1)

        # 为了训练监控，我们仍然生成对应的图像真值
        # 注意：这个 f_true 仅用于计算 Image Loss，不参与 F 的生成
        with torch.no_grad():
             f_true = self.image_gen(coeff_true.unsqueeze(0)).squeeze(0)

        return coeff_true, f_true, F_observed, coeff_initial

    def _generate_normal_coefficients(self):
        return torch.randn(self.height, self.width)

    def _tikhonov_init(self, F_observed: torch.Tensor, lambda_reg: float = 0.1) -> torch.Tensor:
        """
        使用双精度 (Complex128) 求解 Tikhonov 正则化问题:
        d* = argmin ||Ad - F||^2 + lambda ||d||^2
        Solution: d* = (A^H A + lambda I)^(-1) A^H F
        """
        # 1. 强制转换为双精度
        F_obs_db = F_observed.to(device=device, dtype=torch.complex128)
        
        # 2. 准备方程 LHS = A^H A + lambda I
        # 注意：由于 A 是 ill-conditioned (10^16)，这一步必须是 double precision
        A = self.A_matrix_high_prec
        AH = self.A_H_high_prec
        N = A.shape[1]
        
        # Matrix multiplication A^H @ A
        ATA = torch.matmul(AH, A)
        
        # Regularization term
        I = torch.eye(N, device=device, dtype=torch.complex128)
        LHS = ATA + lambda_reg * I
        
        # 3. 准备方程 RHS = A^H F
        RHS = torch.matmul(AH, F_obs_db)
        
        # 4. 求解线性方程组
        try:
            d_est_db = torch.linalg.solve(LHS, RHS)
        except RuntimeError:
            # 如果极端情况下仍奇异，尝试使用 lstsq 或增加 epsilon
            print("Warning: Singular matrix in Tikhonov, adding epsilon.")
            LHS = LHS + 1e-6 * I
            d_est_db = torch.linalg.solve(LHS, RHS)

        # 5. 映射回 2D 系数矩阵 (转回 float32/complex64 供网络使用)
        # d_est 是按照 flatten_order 排序的，需要映射回 (21, 21)
        coeff_flat = d_est_db.real[self.unflatten_order] # 取实部，因为 B-spline 系数是实数
        coeff_init = coeff_flat.view(self.height, self.width).float()
        
        return coeff_init

    def generate_batch(self, batch_size, random_seed=None):
        # 保持原有的 batch 生成逻辑不变
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        coeff_true_list = []
        f_true_list = []
        F_observed_list = []
        coeff_initial_list = []
        for i in range(batch_size):
            # 传递 seed + i 保证 batch 内样本多样性
            s = None if random_seed is None else random_seed + i
            c_true, f_true, F_obs, c_init = self.generate_training_sample(random_seed=s)
            
            coeff_true_list.append(c_true)
            f_true_list.append(f_true)
            F_observed_list.append(F_obs)
            coeff_initial_list.append(c_init)
            
        coeff_true_batch = torch.stack(coeff_true_list).unsqueeze(1)
        f_true_batch = torch.stack(f_true_list).unsqueeze(1)
        F_observed_batch = torch.stack(F_observed_list).unsqueeze(1)
        coeff_initial_batch = torch.stack(coeff_initial_list).unsqueeze(1)
        
        return coeff_true_batch, f_true_batch, F_observed_batch, coeff_initial_batch
```



### 关键修改点解释



1. **System Matrix $A$ 的构建**:
   - 旧代码：隐含在 `compute_F_G_d` 方法中。
   - 新代码：在 `__init__` 中显式计算 `self.A_matrix_high_prec = Phi.unsqueeze(1) * G`。这是为了在 Tikhonov 步骤中能直接使用 $A^H A$。
2. **前向模型的一致性**:
   - 现在 `F_clean` 完全由 `d` 通过矩阵乘法生成。这消除了之前代码中 *空间域生成图像 -> Grid Sample 采样 -> FFT* 引入的离散化误差。这使得数据生成过程与反演模型完全匹配（Inverse Crime，但在理论验证阶段是必须的）。
3. **精度控制 (Precision)**:
   - 在 `_tikhonov_init` 中，输入 `F_observed` 被立即转为 `complex128`。
   - 矩阵 `A` 和 `I` 都是 `complex128`。
   - `torch.linalg.solve` 在双精度下运行，能够处理条件数极大的矩阵。
   - 结果仅在最后一步取实部并转回 `float`，以适配 PyTorch 模型的默认精度。
4. **噪声模型调整**:
   - 文档提到噪声基于 `mean(|F_clean|)`。代码中更新为 `mean_amplitude = torch.mean(torch.abs(F_clean))`，然后乘以 `noise_level`。



### 如何使用



直接将上述 `TheoreticalDataGenerator` 类替换 `train.py` 中的同名类即可。`train.py` 中的其他部分（如 `TheoreticalTrainer`）不需要修改，因为接口（输入输出的 tensor 形状）保持了一致。