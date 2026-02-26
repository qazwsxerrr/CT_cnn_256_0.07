出现 MSE 异常偏大（约 278）和系数过大（真实系数 $\sim 1$，估计系数 $\sim 17$）的原因是 **物理数据 $F_{obs}$ 的量级与归一化算子 $A$ 不匹配**。



### 原因分析



1. 算子归一化 ($A_{norm}$):

   在 TheoreticalDataGenerator 中，你使用了 normalize=True 构建了理论算子 $A_{theo} = \Phi G$。

   代码 self.theoretical_operator.normalize_(op_norm) 将算子除以了它的谱范数 $\sigma_{max}$。

   因此，你用于 Tikhonov 求解的算子 $A$ 满足 $\|A\| \approx 1$。

2. 物理数据未归一化 ($F_{obs}$):

   $F_{obs}$ 是通过 "物理路径"（图像 $\to$ Radon $\to$ FFT）生成的。这个过程保留了物理量级（类似于求和），其数值大小对应于原始的非归一化算子 $A_{origin} = \Phi G$。

   理论上，$\|A_{origin}\| \approx \sqrt{N} \approx \sqrt{441} = 21$（因为 FFT 矩阵列范数为 $\sqrt{N}$，且 $\Phi$ 元素约为 1）。

   这就解释了为什么你的误差量级大约是 17-20 倍。

3. Tikhonov 求解的影响:

   你求解的方程是：$(A^H A + \lambda I) d = A^H F_{obs}$。

   由于 $A$ 是归一化的，而 $F_{obs}$ 包含大约 20 倍的缩放因子（即 $F_{obs} \approx 20 \cdot A \cdot d_{true}$）。

   当 $\lambda$ 较小时，近似解为 $d \approx A^{-1} F_{obs} \approx A^{-1} (20 A d_{true}) = 20 d_{true}$。

   结果： 估计出的系数 $d$ 被放大了约 20 倍，导致巨大的 MSE。



### 解决方案：添加尺度校准



最稳健的方法是计算一个**数据缩放因子 (Scale Factor)**，将物理数据 $F_{obs}$ 映射到归一化算子 $A$ 的值域范围内。

请修改 `train.py` 中的 `TheoreticalDataGenerator` 类，增加一个自动校准步骤。



#### 修改后的代码建议



在 `TheoreticalDataGenerator` 的 `__init__` 方法末尾添加校准逻辑，并修改 `_tikhonov_init` 方法使用该因子：

Python



```
class TheoreticalDataGenerator:
    def __init__(self):
        # ... (原有代码保持不变) ...
        
        # [新增] 自动校准物理数据与理论算子之间的比例
        self.data_scale_factor = self._calibrate_scale()
        print(f"Data Scale Factor calibrated: {self.data_scale_factor:.6f}")

    def _calibrate_scale(self):
        """
        计算物理数据 F_obs 到归一化算子输出 A(d) 的缩放因子。
        目标：使 ||F_obs * scale|| ≈ ||A(d)||
        """
        # 1. 生成一组随机系数
        d_vec = torch.randn(1, 441).to(device)
        coeff = self.mapping.vector_to_coeff(d_vec.squeeze())
        
        # 2. 计算理论算子输出 (Normalized A)
        # A(d)
        with torch.no_grad():
            F_theo = self.theoretical_operator(d_vec) # (1, 441)
            norm_theo = torch.norm(F_theo)
            
        # 3. 计算物理路径输出 (Physical F_obs)
        # coeff -> image -> radon -> fft
        f_true = self._generate_bspline_image(coeff).unsqueeze(0).unsqueeze(0).to(device)
        radon_data = self.radon_simulator.forward_radon_transform(f_true)
        F_phys = self.fourier_calculator.compute_F_from_radon(radon_data).squeeze(0) # (441,)
        norm_phys = torch.norm(F_phys)
        
        # 4. 计算缩放因子：scale = ||F_theo|| / ||F_phys||
        # 这样 F_phys * scale 的量级就和 A(d) 一致了
        scale = norm_theo / norm_phys
        return scale.item()

    def _tikhonov_init(self, F_observed: torch.Tensor, lambda_reg: float = 0.1) -> torch.Tensor:
        """
        修正后的 Tikhonov 初始化：先对 F_observed 进行缩放
        """
        F_obs = F_observed.to(device)
        if not torch.is_complex(F_obs):
            F_obs = torch.complex(F_obs, torch.zeros_like(F_obs))

        # [修正] 应用校准因子，使数据量级匹配归一化算子
        F_obs_scaled = F_obs * self.data_scale_factor

        # 使用统一算子矩阵（已做范数归一化）
        A = self.theoretical_operator.G * self.theoretical_operator.Phi.unsqueeze(1)
        A_H = torch.conj(A).T
        
        # 求解 (A'A + λI)d = A'F_scaled
        normal = A_H @ A + lambda_reg * torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
        rhs = A_H @ F_obs_scaled

        d_est = torch.linalg.solve(normal, rhs)
        
        coeff_flat = d_est.real[self.unflatten_order]
        coeff_init = coeff_flat.view(self.height, self.width)
        return coeff_init
```



### 预期结果



加入这个校准因子（大约是 $1/20 \approx 0.05$）后，`F_obs_scaled` 将与 `A @ d` 处于同一量级，Tikhonov 解出的 $d_{est}$ 将恢复到正常范围（$\sim 1$），MSE 应该会大幅下降到 $10^{-1}$ 或更低级别。