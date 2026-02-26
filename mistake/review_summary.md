# mistake.md 复核摘要与建议（未改代码）

- **Φ/G 按理论改为 1D 频率（沿 α）**：当前 FourierOperatorCalculator 用 21×21 频率网格构造 Φ/G，与单角 Radon 的 1D 公式不符。建议改为 `xi_j = 2π j/N * ||β||`，`Phi_diag[j] = B2_hat(α_x xi_j) * B1_hat(α_y xi_j)`，`G[j,k] = exp(-1j * xi_j * (α_x x_k + α_y y_k))`（k 按统一的展平顺序），确保 F=ΦGd 与 Radon+FFT 理论一致。
- **统一展平顺序，消除 transpose 补丁**：现在数据生成对系数做了 `coeff_matrix.t().flatten()`，而 G 的列索引用 y-major。需全局统一：要么将系数展平固定为行优先（k = y*21 + x）并同步调整 G 的列索引；要么修改 B 样条生成遍历顺序以匹配现有 G，避免不同模块各用一套顺序导致 ΦGd 和真实 Radon FFT 对不齐。
- **B 样条频域公式与稳定性**：`B1_hat_complex/B2_hat_complex` 公式不标准，xi→0 需手工兜底。建议改为 sinc 形式：`B1_hat(ξ)=exp(-1j*ξ/2)*2*sinc(ξ/(2π))`，`B2_hat = B1_hat**2`（无条件稳定），再用于上面 1D Φ。
- **2π 频率刻度一致性**：若使用 DFT 频率 `xi_j = 2π j/N * ||β||`，指数项中不应再遗漏/重复 2π；调整 Φ/G 时一并统一。
- **次要清理**：`_compute_sampling_frequencies` 当前未用，可在重构 Φ/G 后合并或删除，避免误导；config 中未使用的 ODL 几何配置可标注占位以免混淆。
