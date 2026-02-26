结论先说：**整体框架（前向/伴随、TV/Dir 正则、LGD 展开）和理论思路基本一致**，但有几处“关键实现”与理论不符或容易出错，建议尽快按下面的修改修正。改完这几处，训练/验证会更稳定，且与“单角-频域公式 (F=\Phi,G,d)”严格对齐。

------

# 一眼看出的不一致/bug（按优先级）

## 1) 频域算子 (\Phi) 与 (G) 的构造与理论不符（应为 1D 频率沿 (\alpha)）

- **问题**：`FourierOperatorCalculator` 里把频率做成了**二维网格** ((\xi_x,\xi_y))，并据此构造 (\Phi[i,j]=\widehat B_2(\xi_x)\widehat B_1(\xi_y))、(G[i,j,k]=e^{-i,\xi\cdot \text{displacement}})。理论上，**单角 Radon**只需**一维频率** (\xi_j=\tfrac{2\pi j}{N}|\beta|*2)，且(\Phi_j=\hat\phi(\xi_j\alpha)=\widehat B_2(\alpha_x\xi_j)\widehat B_1(\alpha_y\xi_j))，(G*{j,k}=e^{-i,\xi_j(\alpha\cdot k)})。当前实现既没有用 (\xi_j\alpha)，也遗漏了与 (2\pi) 和 (\alpha!\cdot!k) 相关的刻度/相位关系。
- **影响**：等式 (F=\Phi,G,d) 数值上不再等价于“Radon→一维 FFT”的理论结果；即使能训练，学习到的是“另一个问题”。
- **建议修改**（思路）：
  - 用**一维**频率向量 `xi = 2π * torch.arange(N)/N * beta_norm`；
  - 计算 (\Phi) 为**长度 (N)** 的对角：`Phi_diag[j] = B2_hat(alpha_x*xi[j]) * B1_hat(alpha_y*xi[j])`；
  - 计算 (G) 为**(N×441)**：`G[j, k] = exp(-1j * xi[j] * (alpha_x * x_k + alpha_y * y_k))`，其中 (k) 对应网格点 ((x_k,y_k)\in{0..20}^2)；
  - 保持 `coeff_matrix.view(-1)` 的**行优先（y 行，x 列）**展平顺序，与你现在 `k = y*21 + x` 的实现一致（见下一条）。

## 2) (k) 的展平顺序在多个模块里不一致（易出现错位）

- `G` 的列索引用的是 `k = idx_y*21 + idx_x`（**y 优先**）；而 `CardinalBSpline2D.generate_cardinal_pattern` 的系数遍历是外层 `kx`（x），内层 `ky`（y），即 `k = kx*21 + ky`（**x 优先**），为此你在数据生成里临时做了 `coeff_matrix.t().flatten()` 来“对齐”，这很脆弱。
- **建议二选一**：
  - **更稳**：把 `generate_cardinal_pattern` 的遍历改为外层 `ky`、内层 `kx`（即 `k = ky*21 + kx`），然后**删掉** `train.py` 里那句 `coeff_matrix.t().flatten()` 的转置（直接 `flatten()` 即可）；或
  - **维持现状**：保持 `generate_cardinal_pattern` 不改，但统一规定**所有**用到 (d)/(c) 的展平均使用 `coeff_matrix.t().flatten()`；并把 `FourierOperatorCalculator._compute_G_matrix` 的 `k` 改成 `k = idx_x*21 + idx_y` 来匹配。
- 原因：一处用 y-major、一处用 x-major，会导致 (G) 的列与 (d) 的元素**错位**，从而破坏 (F=\Phi G d) 的可比对性。

## 3) ( \widehat B_1(\xi),\widehat B_2(\xi) ) 的公式与数值稳定性

- **问题**：`B1_hat_complex/B2_hat_complex` 使用了 `exp(-i*ξ/2) * exp(-i*ξ) / (iξ)` 之类的式子；这与**标准公式**不一致，而且在 (\xi\to 0) 时需要硬塞极限值。标准上（支撑在 ((0,1])）：
   [
   \widehat{B_1}(\xi)=\frac{1-e^{-i\xi}}{i\xi}
   =e^{-i\xi/2}\cdot \frac{2\sin(\xi/2)}{\xi},\quad
   \widehat{B_2}(\xi)=\widehat{B_1}(\xi)^2
   =e^{-i\xi}\cdot\Big(\tfrac{2\sin(\xi/2)}{\xi}\Big)^2 .
   ]
   用 `sinc` 形式可以**自然**处理 (\xi\to 0) 的极限，无需 if。
- **建议修改**：把两函数改为 NumPy 纯矢量版：
  - `B1_hat(ξ) = np.exp(-1j*ξ/2) * 2 * np.sinc(ξ/(2*np.pi))`（注意 `np.sinc` 的定义是 (\mathrm{sinc}(x)=\sin(\pi x)/(\pi x))）；
  - `B2_hat(ξ) = (B1_hat(ξ))**2`。
- 这样在 `FourierOperatorCalculator` 里就能**稳健**地计算 (\Phi_j)（上一条的 1D 公式）。

## 4) 理论梯度下降里**正则权重 (\lambda)**没有生效

- `TheoreticalGradientDescent.gradient_descent_step` 里把 `reg_grad` 直接加到了 `data_grad`，但**没有乘** `self.lambda_reg`；而你的 LGD 路径里是用 `self.reg_lambda` 乘过的，所以两条“理论 vs 学习”对比会失真。建议：在 `gradient_descent_step` 中改为
   `total_grad = data_grad + self.lambda_reg * reg_grad`；并在 `get_theoretical_vs_learned_updates` 的“理论步”对齐同样逻辑。

## 5) (G) 的相位里遗漏了 (2\pi) 的刻度

- 你现在的 `exp(-1j * phase)` 里 `phase = xi⋅displacement`，而 `xi` 的单位/范围并未含 (2\pi)。若按 DFT 约定取 (\xi_j=\tfrac{2\pi j}{N}|\beta|_2)，则指数里**无需**再额外乘 (2\pi)；反之若使用“非归一化的”频率网格，就应该把 (2\pi) 显式乘上。建议按第 1 条统一成 (\xi_j=\tfrac{2\pi j}{N}|\beta|_2)。

## 6) ODL/几何的小问题（非阻断）

- `config.py` 里自建几何的 `Flat1dDetector/ParallelBeamGeometry` 参数组合看起来不会被训练主流程用到（训练使用了 `radon_transform.py` 中的 `Parallel2dGeometry`），但那段配置大概率无效/不一致；建议删除或标注“仅占位”，以免混淆。

------

# 建议的最小改动（示例片段）

> 下面是按你现有代码风格的“可直接替换”的思路（给出关键要点，便于你粘贴实现）。

### A. `box_spline.py`：稳定而正确的 (\widehat B_1,\widehat B_2)（替换两函数）



```python
# 替换 B1_hat_complex / B2_hat_complex 为 NumPy 纯函数（支持标量或向量）
def B1_hat_complex(self, xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=np.float64)
    # np.sinc(x) = sin(pi x)/(pi x)
    return np.exp(-1j * xi / 2.0) * 2.0 * np.sinc(xi / (2.0 * np.pi))

def B2_hat_complex(self, xi: np.ndarray) -> np.ndarray:
    B1 = self.B1_hat_complex(xi)
    return B1 * B1
```

### B. `radon_transform.py`：用 1D 频率沿 (\alpha) 构造 (\Phi) 与 (G)



```python
def _compute_Phi_matrix(self):
    N = self.n_coefficients
    j = torch.arange(N, dtype=torch.float64)
    xi = 2.0 * np.pi * j / N * self.beta_norm.item()  # (N,)
    ax, ay = self.alpha.double().tolist()

    # 调用上面修正过的 B1_hat / B2_hat（可用 numpy 后再转 torch）
    from box_spline import CardinalBSpline2D
    bs = CardinalBSpline2D()
    B2 = np.asarray(bs.B2_hat_complex(ax * xi.numpy()), dtype=np.complex128)
    B1 = np.asarray(bs.B1_hat_complex(ay * xi.numpy()), dtype=np.complex128)
    Phi = B2 * B1                                    # (N,)
    self.Phi_diagonal = torch.from_numpy(Phi).to(torch.complex64)  # (N,)
    self.Phi_matrix = None  # 不再使用 2D 网格

def _compute_G_matrix(self):
    N = self.n_coefficients
    j = torch.arange(N, dtype=torch.float64)
    xi = 2.0 * np.pi * j / N * self.beta_norm        # (N,)
    ax, ay = self.alpha.double()
    xs = torch.arange(21, dtype=torch.float64)
    ys = torch.arange(21, dtype=torch.float64)
    X, Y = torch.meshgrid(xs, ys, indexing='xy')     # X: x, Y: y  (21,21)
    # 展平顺序与 coeff_matrix.view(-1) 一致：k = y*21 + x
    kdot = (ax * X + ay * Y).reshape(-1)             # (441,)
    phase = torch.outer(xi, kdot)                    # (N, 441)
    G = torch.exp(-1j * phase)                       # (N, 441)
    self.G_matrix_2d = G.to(torch.complex64)
```

> 如果你选择把 `generate_cardinal_pattern` 的系数遍历改为 **y 外 x 内**，记得同步删掉 `train.py` 里对系数的 `.t()`；否则就把上面 `kdot` 的展平改成 `k = x*21 + y` 的次序来对齐你当前的数据生成路径。

### C. `model.py`：让 (\lambda) 在“理论步”生效



```python
def gradient_descent_step(self, coeff_matrix, F_observed):
    data_grad = self.compute_data_fidelity_gradient(coeff_matrix, F_observed)
    reg_grad  = self.compute_regularization_gradient(coeff_matrix)
    total_grad = data_grad + self.lambda_reg * reg_grad   # <-- 乘 λ
    return coeff_matrix - self.step_size * total_grad
```

并在 `get_theoretical_vs_learned_updates` 里计算“理论步”时，保持**同样**的 `lambda_reg` 乘法。

------

# 其它小建议（可选，提升鲁棒性/效率）

- `FourierOperatorCalculator` 里 `_compute_sampling_frequencies` 目前**未被使用**；合并进上面的 1D 频率定义，避免读者误解。
- 三/四重 for-loop 生成 (G) 与 (\Phi) 的地方已用**张量广播**替代（见上面 B 片段），速度会快很多。
- `RadonTransformSimulator` 里选择 `astra_cpu` 时建议提供更友好的降级信息（你已经做了 `try/except`，很好），并把验证函数里“1% 相对误差”阈值改成**在新公式下重新评估**。
- `config.py` 里不再使用的 ODL 几何可移除或注释为“示例”，避免与 `radon_transform.py` 的几何混淆。

------

# 小结（你关心的关键点）

- 只要把 **(\Phi)/(G) 改为 1D（沿 (\alpha)）**、**B-spline 的频域公式改对**、**(\lambda)** 放进“理论步”，再**统一展平顺序**，你的代码就与我们前面总结的理论严格一致了；训练得到的“学习步”也会更可解释。
- 上面给的代码片段就是“最小改动集”，按你的当前结构直接替换即可。

如果你愿意，我可以**按你当前的展平约定**给出完整可替换的 `FourierOperatorCalculator` 实现（含单元测试：验证 `F_from_radon` 与 `Phi G d` 的相对误差收敛到 ~1e-6）。

—
 引文：`box_spline.py`（基函数与 FFT 实现、系数遍历方式）；`config.py`（几何与配置占位）；`model.py`（LGD/理论步、(\lambda) 权重、伴随实现）；`radon_transform.py`（(\Phi,G) 构造、Radon 仿真与 (F) 计算）；`train.py`（数据生成路径、`coeff_matrix.t().flatten()` 的权宜做法）；`util.py`（仅系数数据生成器，不影响主线）。