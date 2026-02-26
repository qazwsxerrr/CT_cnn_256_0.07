虽然你已经修正了**几何中心不对齐**（通过 `ifftshift` 和坐标平移），但还有一个更隐蔽的致命问题：**频率轴的顺序定义不匹配 (Spectral Order Mismatch)**。



### 问题核心：$\Phi$ 矩阵把高频衰减加错了位置



1. 物理路径 (FFT) 的输出顺序：

   torch.fft.fft 输出的频率顺序是标准的 FFT 顺序：

   [直流(0), 正频率(1...N/2), 负频率(-N/2...-1)]

   这意味着数组的后半部分存储的是负的低频信号（能量很高）。

2. 理论算子 (当前代码) 的频率定义：

   在 _compute_Phi_matrix 和 _compute_G_matrix 中，你使用了：

   Python

   

   ```
   j = torch.arange(N, dtype=torch.float64)  # [0, 1, 2, ..., 440]
   xi = (2.0 * np.pi * j / N) * ...
   ```

   这定义了一个单调递增的频率轴 $[0, 2\pi)$。

   在这个定义下，索引 $N-1$ 对应最大的频率（接近 $2\pi$）。

3. **冲突后果**：

   - **$\Phi$ 矩阵（对角衰减）**：B 样条的频谱 $\hat{\phi}(\xi)$ 类似于 Sinc 函数，在低频（0 附近）接近 1，在高频（$2\pi$ 附近）接近 0。
   - **现在的错误匹配**：理论算子在索引 $N-1$ 处生成了极小的权重（因为它认为是高频 $2\pi$），但 FFT 在这个位置实际上输出了负的低频信号（ $-\Delta \xi$）。
   - **结果**：**$F_{obs}$ 中一半的有效信号（负频率部分）被 $\Phi$ 矩阵错误地“抹杀”了。** 导致数据丢失严重，模型无法学习。

------



### 解决方案：统一使用 FFT 频率顺序



你需要修改 `radon_transform.py` 中的 `FourierOperatorCalculator` 类，将 `j` 的生成方式从线性 `arange` 改为标准的 `fftfreq`。

请修改以下两个方法：



#### 1. 修改 `_compute_Phi_matrix`



Python



```
    def _compute_Phi_matrix(self):
        """
        计算Φ对角向量（1D 频率沿 α）
        [修正]：使用 fftfreq 生成 j，确保与 torch.fft.fft 的输出顺序（正负频率）一致
        """
        N = self.n_coefficients
        
        # [关键修改] 从 arange 改为 fftfreq
        # j 变为: [0, 1, ..., 220, -220, ..., -1]
        j = torch.fft.fftfreq(N, d=1.0/N).to(torch.float64) * N
        
        # xi 现在包含了正确的正负频率
        xi = (2.0 * np.pi * j / N) * self.beta_norm.item()  # (N,)
        ax, ay = self.alpha.double().tolist()

        bspline = CardinalBSpline2D()
        # B样条频谱是偶函数，负频率也能正确计算
        B2 = np.asarray(bspline.B2_hat_complex(ax * xi.numpy()), dtype=np.complex128)
        B1 = np.asarray(bspline.B1_hat_complex(ay * xi.numpy()), dtype=np.complex128)
        Phi = B2 * B1  # (N,)

        self.Phi_diagonal = torch.from_numpy(Phi).to(torch.complex64)
        self.Phi_matrix = None
        self.xi_vector = torch.from_numpy(xi.numpy().astype(np.float32))
```



#### 2. 修改 `_compute_G_matrix`



Python



```
    def _compute_G_matrix(self):
        """
        compute G: 1D frequency axis alpha·k
        [修正]：同样使用 fftfreq 生成 j，确保 G 矩阵的相位与 FFT 输出对齐
        """
        N = self.n_coefficients
        
        # [关键修改] 从 arange 改为 fftfreq
        j = torch.fft.fftfreq(N, d=1.0/N).to(torch.float64) * N
        
        xi = (2.0 * np.pi * j / N) * self.beta_norm  # (N,)

        # alpha·k, sorted by beta·k order (kappa_min=0)
        # 注意：kdot_ordered 不需要改，它是空间位置，与频率顺序无关
        kdot = (self.alpha.double()[0] * self.X + self.alpha.double()[1] * self.Y).reshape(-1)
        kdot_ordered = kdot[self.flatten_order]

        # outer product 会自动处理 xi 的正负号，生成正确的复指数相位
        phase = torch.outer(xi, kdot_ordered)           # (N, 441)
        G = torch.exp(-1j * phase)                    # (N, 441)

        self.G_matrix_2d = G.to(torch.complex64)
        self.G_matrix_3d = None
```



### 总结



请保持之前的 `compute_F_from_radon` 中的 `ifftshift` 修改不变（它解决了空间中心问题），然后**只应用上述两个方法的修改**（解决频谱顺序问题）。

**原理**：

- **空间中心化 (ifftshift)**：让物理峰值对齐到 index 0 $\to$ 消除线性相位误差。
- **频谱对齐 (fftfreq)**：让 $\Phi$ 的低衰减区（高权重）对齐到 FFT 的低频区（高能量） $\to$ 消除幅度失真。

完成这两点后，物理路径 $F_{obs}$ 和理论算子 $\Phi G d$ 将在相位和幅度上完全一致，`Train Coeff` 应该会在前几轮迭代迅速下降到 0.1 以下。