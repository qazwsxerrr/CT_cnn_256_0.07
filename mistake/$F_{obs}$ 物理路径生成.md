这是一份将 $F_{obs}$ 生成方式从“理论算子路径”修改为“物理仿真路径”的完整实施大纲。

修改的核心目标是在 `train.py` 中替换数据生成逻辑，利用 `radon_transform.py` 中已有的物理仿真功能。

------



### **修改大纲：$F_{obs}$ 物理路径生成**





#### **第一阶段：初始化准备 (修改 train.py)**



在这一步，我们需要在训练脚本中引入物理仿真器，并在数据生成类中进行实例化。

1. **引入模块**

   - 在 `train.py` 头部导入 `RadonTransformSimulator`。
   - **位置**: `train.py` 的 import 区域。
   - **动作**: `from radon_transform import RadonTransformSimulator`。

2. **实例化仿真器**

   - 在 `TheoreticalDataGenerator` 类的 `__init__` 方法中初始化仿真器。

   - **位置**: `train.py` -> `TheoreticalDataGenerator.__init__`。

   - **代码逻辑**:

     Python

     

     ```
     # 新增代码
     self.radon_simulator = RadonTransformSimulator(
         beta=self.beta,
         image_shape=self.image_shape,  # (128, 128)
         n_angles=1,                    # 单角度扫描
         n_detectors=441                # 探测器数量需匹配系数数量
     )
     ```

   - **依据**: `RadonTransformSimulator` 类定义及其初始化参数。

------



#### **第二阶段：核心逻辑替换 (修改 train.py)**



这是最关键的步骤，将原有的矩阵乘法路径替换为“图像->Radon->FFT”路径。

1. **定位目标方法**
   - 找到 `train.py` 中的 `TheoreticalDataGenerator.generate_training_sample` 方法。
2. **保留前半部分**
   - 保留生成真实系数 `coeff_true` 的代码。
   - 保留生成真实图像 `f_true` 的代码。
3. **替换 $F_{obs}$ 生成逻辑**
   - **旧逻辑 (删除或注释)**:
     - `d_true = ...`
     - `F_clean = self.theoretical_operator(d_true)`
   - **新逻辑 (插入)**:
     1. **维度调整**: 将 `f_true` (128, 128) 调整为 `(1, 1, 128, 128)` 以适配 ODL 输入要求。
     2. **Radon 变换**: 调用 `self.radon_simulator.forward_radon_transform(f_input)`。
     3. **物理转频域**: 调用 `self.fourier_calculator.compute_F_from_radon(radon_data)`。
     4. **维度还原**: 将结果 `squeeze` 回 `(441,)` 或保持 `(1, 441)` 供后续加噪使用。
4. **保留后半部分**
   - 保留噪声添加逻辑（`noise_level` 处理）。
   - 保留 `coeff_initial` 的 Tikhonov 初始化逻辑（这一步使用 *新生成的* $F_{obs}$ 作为输入）。

------



#### **第三阶段：数据流与设备管理 (注意事项)**



在实施过程中需注意数据类型和设备（CPU/GPU）的转换，避免运行时错误。

1. **设备 (Device) 转换**
   - `f_true` 生成时通常在 GPU 上（如果配置了 CUDA）。
   - `RadonTransformSimulator` 内部处理了 CPU/GPU 转换（ODL 主要运行在 CPU，但 `forward_radon_transform` 包含了 `.cpu()` 和 `.to(device)` 的逻辑）。
   - **检查点**: 确保传入 `compute_F_from_radon` 的数据在正确的设备上。
2. **数据类型 (Dtype) 一致性**
   - `compute_F_from_radon` 输出的是复数张量 (`torch.complex64` 或 `128`)。
   - 确保后续加噪步骤支持复数运算（现有代码已支持：`torch.randn_like(F_clean)` 对复数输入会自动生成复数噪声）。

------



#### **第四阶段：验证修改**



修改完成后，建议运行一次简单的验证以确保形状和梯度流正常。

1. **形状检查**
   - 打印新生成的 `F_observed` 的 shape。应为 `(441,)` (单样本) 或 `(B, 1, 441)` (批次)。
   - 对比修改前后的 shape，必须完全一致，否则后续的模型输入会报错。
2. **模型兼容性验证**
   - 运行 `train.py` 一个 epoch。
   - 观察 `data_fidelity_error`。
   - **预期现象**: 初始误差可能会比修改前（理论路径）稍大，因为现在引入了“模型误差”（Model Error/Mismatch），这正是消除“逆罪”后的正常且预期的结果。



### **总结代码片段预览**



修改后的 `generate_training_sample` 核心区域应如下所示：

Python



```
# ... (生成 coeff_true 和 f_true 后)

# --- 物理路径生成 F ---
# 1. 调整维度适配 Radon 模拟器
f_input = f_true.unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 128)

# 2. 物理 Radon 变换 (模拟 CT 扫描)
radon_data = self.radon_simulator.forward_radon_transform(f_input)

# 3. FFT 与相位校正 (计算 F_obs)
F_physical = self.fourier_calculator.compute_F_from_radon(radon_data) # (1, 441)
F_clean = F_physical.squeeze(0) # (441,)

# --- 后续保持不变 ---
# 4. 加噪
# ...
```