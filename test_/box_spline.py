"""
箱样条（Box Spline）和B样条（B-spline）函数模块，用于CT图像重建。

本模块实现了多种类型的箱样条和B样条函数，这些函数可以用于生成CT重建的训练数据。
箱样条函数是一类分段多项式函数，能够创建各种平滑的测试模式，特别适合CT重建算法的测试。

主要功能：
1. BoxSpline2D：二维箱样条实现，生成平滑测试模式
2. TensorProductBSpline：张量积B样条曲面生成器
3. MultiScaleBoxSpline：多尺度箱样条模式生成器
4. 各种可视化工具和辅助函数

数学背景：
箱样条函数具有局部支撑性、平滑性、可分性和尺度不变性等优点，
使其成为CT重建测试数据的理想选择。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple, List, Optional
import odl

class BoxSpline2D:
    """
    二维箱样条实现，用于生成平滑的测试模式。

    箱样条是一类分段多项式函数，能够创建各种平滑的模式，这些模式
    对于CT重建测试非常有用。箱样条具有以下优点：
    - 局部支撑性：便于计算和优化
    - 平滑性：提供连续的导数
    - 可分性：支持多变量分离
    - 尺度不变性：支持多尺度分析

    应用场景：
    1. 生成CT重建的测试幻影数据
    2. 创建具有不同频率成分的训练模式
    3. 模拟医学图像中的各种结构特征
    """

    def __init__(self, degree: int = 3):
        """
        初始化箱样条生成器。

        参数:
            degree: B样条的阶数（默认值：3，表示三次样条）
                   - 0: 最近邻插值
                   - 1: 线性插值
                   - 2: 二次样条
                   - 3: 三次样条（推荐，平衡平滑度和计算效率）
                   - 更高阶: 通过卷积低阶核获得

        数学原理：
        B样条核是通过重复卷积基本核函数[0.5, 0.5]来构造的。
        阶数越高，函数越平滑，但计算复杂度也相应增加。
        """
        self.degree = degree
        self.kernel = self._create_bspline_kernel()

    def _create_bspline_kernel(self) -> np.ndarray:
        """
        创建指定阶数的一维B样条核。

        返回:
            一维B样条核数组

        核函数说明：
        - 0阶: [1.0] - 最近邻插值，最简单但不够平滑
        - 1阶: [0.5, 0.5] - 线性插值，计算效率高
        - 2阶: [0.25, 0.5, 0.25] - 二次样条，适度平滑
        - 3阶: [0.125, 0.375, 0.375, 0.125] - 三次样条，推荐使用
        - 更高阶: 通过卷积构造，平滑度递增

        注意：核函数系数之和始终为1，保证数值稳定性。
        """
        if self.degree == 0:  # 最近邻插值
            return np.array([1.0])
        elif self.degree == 1:  # 线性插值
            return np.array([0.5, 0.5])
        elif self.degree == 2:  # 二次样条
            return np.array([0.25, 0.5, 0.25])
        elif self.degree == 3:  # 三次样条
            return np.array([0.125, 0.375, 0.375, 0.125])
        else:
            # 对于更高阶数，通过卷积低阶核来构造
            kernel = np.array([0.5, 0.5])
            for _ in range(self.degree - 1):
                kernel = np.convolve(kernel, [0.5, 0.5])
            return kernel

    def generate_2d_pattern(self, shape: Tuple[int, int],
                          control_points: Optional[List[Tuple[float, float, float]]] = None,
                          frequency: float = 0.1) -> np.ndarray:
        """
        生成二维箱样条模式。

        这是箱样条生成的核心方法，通过控制点和高斯权重函数创建平滑的二维模式，
        然后应用B样条核进行平滑处理。

        参数:
            shape: 输出形状 (高度, 宽度)
            control_points: 控制点列表，每个控制点格式为 (x, y, 振幅)
                           如果为None，将自动生成3-8个随机控制点
            frequency: 模式的基础频率，控制模式的平滑程度
                      较小值产生更平滑的模式，较大值产生更细节的模式

        返回:
            包含箱样条模式的二维numpy数组，值范围归一化到[0, 1]

        生成过程：
        1. 设置控制点（随机或指定）
        2. 创建坐标网格
        3. 基于控制点和高斯权重函数计算基础模式
        4. 应用二维B样条卷积进行平滑
        5. 归一化到[0, 1]范围

        数学原理：
        模式 = 归一化(卷积2D(基础模式, 外积(B样条核, B样条核)))
        其中基础模式由控制点的高斯加权组合构成。
        """
        if control_points is None:
            # 如果没有提供控制点，生成随机控制点
            # 控制点数量随机选择，确保模式的多样性
            n_points = np.random.randint(3, 8)
            control_points = []
            for _ in range(n_points):
                # 控制点位置允许超出[0,1]范围，产生边界效应
                x = np.random.uniform(-0.5, 1.5)
                y = np.random.uniform(-0.5, 1.5)
                # 振幅范围选择确保合理的对比度
                amplitude = np.random.uniform(0.3, 1.0)
                control_points.append((x, y, amplitude))

        # 创建坐标网格，用于计算每个像素到控制点的距离
        x = np.linspace(0, 1, shape[1])  # x轴坐标
        y = np.linspace(0, 1, shape[0])  # y轴坐标
        X, Y = np.meshgrid(x, y)  # 二维坐标网格

        # 初始化模式数组
        pattern = np.zeros(shape)

        # 每个控制点对模式的贡献
        for cx, cy, amp in control_points:
            # 计算当前像素到控制点的欧几里得距离
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

            # 应用高斯权重函数：权重 = exp(-距离²/(2*频率²))
            # 这确保了控制点附近的影响更强，远处影响逐渐衰减
            weight = np.exp(-dist**2 / (2 * frequency**2))
            pattern += amp * weight  # 叠加所有控制点的贡献

        # 应用二维卷积与B样条核进行平滑处理
        # 外积操作将一维核扩展为二维可分离核
        kernel_2d = np.outer(self.kernel, self.kernel)
        pattern_2d = signal.convolve2d(pattern, kernel_2d,
                                      mode='same', boundary='symm')
        # mode='same' 保持输出尺寸不变
        # boundary='symm' 使用对称边界条件，减少边界伪影

        # 归一化到[0, 1]范围
        if pattern_2d.max() > pattern_2d.min():
            pattern_2d = (pattern_2d - pattern_2d.min()) / (pattern_2d.max() - pattern_2d.min())

        return pattern_2d


class TensorProductBSpline:
    """
    张量积B样条曲面生成器，用于创建平滑的图像。

    张量积B样条是两个一维B样条函数的张量积，可以生成非常平滑的二维曲面。
    相比于简单的箱样条，张量积B样条提供了更精确的局部控制能力。

    数学定义：
    S(u,v) = Σᵢⱼ Nᵢ,p(u) * Nⱼ,q(v) * Pᵢⱼ
    其中 Nᵢ,p(u) 和 Nⱼ,q(v) 是B样条基函数，Pᵢⱼ 是控制点网格。

    应用场景：
    1. 生成具有精确局部控制的平滑测试图像
    2. 模拟医学图像中的平滑解剖结构
    3. 创建具有特定频率特征的训练数据
    """

    def __init__(self, degree_u: int = 3, degree_v: int = 3):
        """
        初始化张量积B样条生成器。

        参数:
            degree_u: u方向的B样条阶数（通常为3，表示三次样条）
            degree_v: v方向的B样条阶数（通常为3，表示三次样条）
                     不同方向的阶数可以不同，用于创建非对称的模式

        初始化说明：
        - 更高的阶数产生更平滑的曲面，但计算复杂度增加
        - 三次样条(C²连续性)通常是最佳选择，平衡了平滑度和效率
        - 控制点数量和阶数共同决定曲面的灵活性
        """
        self.degree_u = degree_u
        self.degree_v = degree_v
        self.knot_vector_u = None  # u方向的节点向量
        self.knot_vector_v = None  # v方向的节点向量
        self.control_points = None  # 控制点网格

    def create_uniform_knot_vector(self, n_control_points: int, degree: int) -> np.ndarray:
        """
        为B样条创建均匀节点向量。

        节点向量是B样条定义的关键组成部分，它决定了基函数在参数域上的分布。

        参数:
            n_control_points: 控制点数量
            degree: B样条阶数

        返回:
            均匀分布的节点向量

        节点向量结构：
        - 前'degree'个节点为0（钳制边界）
        - 中间部分均匀分布在[0,1]区间
        - 后'degree'个节点为1（钳制边界）

        钳制节点向量的优点：
        1. 确保曲面通过边界控制点
        2. 提供更好的边界行为
        3. 减少边界伪影
        """
        n_knots = n_control_points + degree + 1  # 节点总数 = 控制点数 + 阶数 + 1
        knot_vector = np.zeros(n_knots)

        # 创建钳制节点向量（clamped knot vector）
        knot_vector[:degree] = 0  # 起始钳制
        knot_vector[degree:-degree] = np.linspace(0, 1, n_knots - 2*degree)  # 内部节点均匀分布
        knot_vector[-degree:] = 1  # 结束钳制

        return knot_vector

    def generate_surface(self, shape: Tuple[int, int],
                        n_control_u: int = 5, n_control_v: int = 5,
                        random_seed: Optional[int] = None) -> np.ndarray:
        """
        生成张量积B样条曲面

        Args:
            shape: 输出形状 (高度, 宽度)
            n_control_u: u方向的控制点数量
            n_control_v: v方向的控制点数量
            random_seed: 可重现性的随机种子

        Returns:
            包含B样条曲面的2D numpy数组
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # 生成随机控制点
        self.control_points = np.random.uniform(0.2, 0.8, (n_control_u, n_control_v))

        # 创建节点向量
        self.knot_vector_u = self.create_uniform_knot_vector(n_control_u, self.degree_u)
        self.knot_vector_v = self.create_uniform_knot_vector(n_control_v, self.degree_v)

        # 创建参数空间
        u = np.linspace(0, 1, shape[1])
        v = np.linspace(0, 1, shape[0])
        U, V = np.meshgrid(u, v)

        # 计算B样条曲面
        surface = self._evaluate_surface(U, V)

        # 归一化到[0, 1]
        if surface.max() > surface.min():
            surface = (surface - surface.min()) / (surface.max() - surface.min())

        return surface

    def _evaluate_surface(self, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        """在参数值处计算张量积B样条曲面"""
        surface = np.zeros_like(U)

        for i in range(self.control_points.shape[0]):
            for j in range(self.control_points.shape[1]):
                basis_u = self._basis_function(i, self.degree_u, self.knot_vector_u, U)
                basis_v = self._basis_function(j, self.degree_v, self.knot_vector_v, V)
                surface += self.control_points[i, j] * basis_u * basis_v

        return surface

    def _basis_function(self, i: int, degree: int, knot_vector: np.ndarray,
                       param: np.ndarray) -> np.ndarray:
        """使用Cox-de Boor递归计算B样条基函数"""
        if degree == 0:
            return ((param >= knot_vector[i]) &
                   (param < knot_vector[i + 1])).astype(float)

        # Cox-de Boor递归
        denom1 = knot_vector[i + degree] - knot_vector[i]
        term1 = 0.0
        if denom1 != 0:
            term1 = ((param - knot_vector[i]) / denom1 *
                    self._basis_function(i, degree - 1, knot_vector, param))

        denom2 = knot_vector[i + degree + 1] - knot_vector[i + 1]
        term2 = 0.0
        if denom2 != 0:
            term2 = ((knot_vector[i + degree + 1] - param) / denom2 *
                    self._basis_function(i + 1, degree - 1, knot_vector, param))

        return term1 + term2


class MultiScaleBoxSpline:
    """
    多尺度箱样条模式，用于在不同频带测试CT重建
    """

    def __init__(self, scales: List[float] = [0.05, 0.1, 0.2]):
        """
        初始化多尺度箱样条生成器

        Args:
            scales: 要生成的频率尺度列表
        """
        self.scales = scales
        self.generators = [BoxSpline2D(degree=3) for _ in scales]

    def generate_multiscale_pattern(self, shape: Tuple[int, int],
                                  random_seed: Optional[int] = None) -> np.ndarray:
        """
        生成多尺度箱样条模式

        Args:
            shape: 输出形状 (高度, 宽度)
            random_seed: 可重现性的随机种子

        Returns:
            组合的多尺度模式
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        combined_pattern = np.zeros(shape)

        for i, (scale, generator) in enumerate(zip(self.scales, self.generators)):
            # 在此尺度生成模式
            pattern = generator.generate_2d_pattern(shape, frequency=scale)

            # 权重贡献（更高频率获得更低权重）
            weight = 1.0 / (i + 1)
            combined_pattern += weight * pattern

        # 归一化最终模式
        if combined_pattern.max() > combined_pattern.min():
            combined_pattern = (combined_pattern - combined_pattern.min()) / \
                            (combined_pattern.max() - combined_pattern.min())

        return combined_pattern


def create_box_spline_function(space: odl.DiscretizedSpace,
                              pattern_type: str = 'multiscale',
                              random_seed: Optional[int] = None) -> odl.DiscretizedSpaceElement:
    """
    为CT重建测试创建箱样条函数

    Args:
        space: ODL函数空间
        pattern_type: 模式类型 ('box_spline', 'tensor_product', 'multiscale')
        random_seed: 可重现性的随机种子

    Returns:
        包含箱样条函数的ODL函数
    """
    shape = space.shape

    if random_seed is not None:
        np.random.seed(random_seed)

    if pattern_type == 'box_spline':
        generator = BoxSpline2D(degree=3)
        pattern = generator.generate_2d_pattern(shape)
    elif pattern_type == 'tensor_product':
        generator = TensorProductBSpline(degree_u=3, degree_v=3)
        pattern = generator.generate_surface(shape)
    elif pattern_type == 'multiscale':
        generator = MultiScaleBoxSpline(scales=[0.05, 0.1, 0.2])
        pattern = generator.generate_multiscale_pattern(shape)
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    # 转换为ODL函数
    return space.element(pattern)


def visualize_box_splines(shape: Tuple[int, int] = (128, 128),
                         save_path: Optional[str] = None):
    """
    可视化不同类型的箱样条模式

    Args:
        shape: 要生成的模式形状
        save_path: 保存可视化的路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 箱样条
    box_spline = BoxSpline2D(degree=3)
    pattern1 = box_spline.generate_2d_pattern(shape)
    axes[0, 0].imshow(pattern1, cmap='gray')
    axes[0, 0].set_title('Box Spline Pattern')
    axes[0, 0].axis('off')

    # 张量积B样条
    tensor_spline = TensorProductBSpline(degree_u=3, degree_v=3)
    pattern2 = tensor_spline.generate_surface(shape)
    axes[0, 1].imshow(pattern2, cmap='gray')
    axes[0, 1].set_title('Tensor Product B-spline')
    axes[0, 1].axis('off')

    # 多尺度
    multi_spline = MultiScaleBoxSpline(scales=[0.05, 0.1, 0.2])
    pattern3 = multi_spline.generate_multiscale_pattern(shape)
    axes[1, 0].imshow(pattern3, cmap='gray')
    axes[1, 0].set_title('Multi-scale Box Spline')
    axes[1, 0].axis('off')

    # 组合可视化
    axes[1, 1].imshow(pattern1 + pattern2 + pattern3, cmap='gray')
    axes[1, 1].set_title('Combined Patterns')
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


class CardinalBSpline2D:
    """
    基数B样条（Cardinal B-spline）实现，基于论文第五节的定义。

    对于n=2的情况，定义为 φ(x,y) = B₂(x)B₁(y)
    其中：
    - B₁是一阶基数B样条（线性B样条）
    - B₂是二阶基数B样条（二次B样条）
    - 支撑集为 (0,2] × (0,1]

    这用于生成论文中描述的441个基函数的线性组合。
    """

    def __init__(self):
        """
        初始化基数B样条生成器
        """
        self.support_x = (0, 2)  # B₂的支撑区间，根据论文 supp(φ) = (0,2] × (0,1]
        self.support_y = (0, 1)  # B₁的支撑区间，根据论文

    def B1(self, x: np.ndarray) -> np.ndarray:
        """
        一阶基数B样条 B₁(x) (零阶B样条)

        根据论文定义：B₁(x) = χ_{(0,1]} (特征函数)
        结果：B₁(x) =
            1, 当 0 < x ≤ 1
            0, 其他

        注意：这里B₁实际上是零阶B样条（矩形脉冲），支撑区间为(0,1]

        参数:
            x: 输入数组

        返回:
            B₁(x) 的值
        """
        result = np.zeros_like(x)

        # 0 < x ≤ 1 区间 - 确保边界条件正确
        # 使用更宽松的条件来确保数值稳定性
        mask1 = (x > 0) & (x <= 1)
        result[mask1] = 1.0

        # 添加极小值的边界处理以确保数值稳定性
        # 这可以帮助处理由于浮点数精度导致的边界问题
        eps = 1e-10
        mask_upper = (x > 1) & (x <= 1 + eps)
        result[mask_upper] = 1.0  # 确保x=1处包含在内

        return result

    def B2(self, x: np.ndarray) -> np.ndarray:
        """
        二阶基数B样条 B₂(x) 根据论文定义

        根据论文：B₂(x) = χ_{(0,1]} ⋆ χ_{(0,1]}
        即两个特征函数的卷积，结果是三角形脉冲

        B₂(x) =
            x, 当 0 < x ≤ 1
            2-x, 当 1 < x ≤ 2
            0, 其他

        这产生一个支撑区间为(0,2]的三角形脉冲

        参数:
            x: 输入数组

        返回:
            B₂(x) 的值
        """
        result = np.zeros_like(x)

        # 0 < x ≤ 1 区间: 上升的斜线
        mask1 = (x > 0) & (x <= 1)
        result[mask1] = x[mask1]

        # 1 < x ≤ 2 区间: 下降的斜线
        mask2 = (x > 1) & (x <= 2)
        result[mask2] = 2 - x[mask2]

        return result

    def phi(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        基数B样条函数 φ(x,y) = B₂(x)B₁(y)

        参数:
            x: x坐标数组
            y: y坐标数组

        返回:
            φ(x,y) 的值
        """
        return self.B2(x) * self.B1(y)

    def generate_cardinal_pattern(self, shape: Tuple[int, int],
                                coefficients: Optional[np.ndarray] = None,
                                region: Tuple[Tuple[float, float], Tuple[float, float]] = ((2, 20), (1, 20)),
                                enforce_region_constraint: bool = True,
                                random_seed: Optional[int] = None) -> np.ndarray:
        """
        生成基数B样条模式，基于441个基函数的线性组合

        参数:
            shape: 输出形状 (高度, 宽度)
            coefficients: 441个系数 c_{k_j}，如果为None则随机生成
            region: 定义区域E = [2,20] × [1,20]
            enforce_region_constraint: 是否强制在区域E外置零（论文要求）
            random_seed: 随机种子

        返回:
            基数B样条模式，在区域E外完全为0
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # 定义网格点范围 E⁺ = {[0,20] × [0,20]} ∩ ℤ²
        # 这对应于论文中的441个点 (21×21=441)
        x_range = range(0, 21)  # x: 0到20
        y_range = range(0, 21)  # y: 0到20

        # 生成或使用提供的系数，使用正态分布（均值0，标准差1）
        if coefficients is None:
            coefficients = np.random.normal(0, 1, 441)
            # 裁剪到[-3, 3]范围内
            coefficients = np.clip(coefficients, -3, 3)

        # 确保系数数量正确
        if len(coefficients) != 441:
            raise ValueError(f"需要441个系数，但提供了{len(coefficients)}个")

        # 创建坐标网格 - 基函数的[0,20]×[0,20]坐标空间
        height, width = shape
        # 基函数坐标空间：[0,20] × [0,20]
        # B样条基函数在这个坐标范围内定义
        x_physical = np.linspace(0, 20, width)  # x范围：0到20
        y_physical = np.linspace(0, 20, height)  # y范围：0到20
        X, Y = np.meshgrid(x_physical, y_physical)

        # 初始化结果
        pattern = np.zeros(shape)

        # 遍历所有441个基函数
        coeff_idx = 0
        for kx in x_range:
            for ky in y_range:
                if coeff_idx >= len(coefficients):
                    break

                # 计算平移后的基函数贡献
                # φ(x - kx, y - ky) = B₂(x - kx) * B₁(y - ky)
                contribution = coefficients[coeff_idx] * self.phi(X - kx, Y - ky)
                pattern += contribution

                coeff_idx += 1

        # 根据论文要求，强制在区域E=[2,20]×[1,20]外置零，但在完整空间[0,20]×[0,20]内计算
        if enforce_region_constraint:
            # 创建物理坐标到像素坐标的映射
            # x_physical ∈ [0, 20], y_physical ∈ [0, 20]
            # 我们需要找出在区域E=[2,20]×[1,20]之外的像素位置

            # 创建掩码：在区域E内的位置为True，E外为False
            # 使用与上面相同的坐标网格：x_physical ∈ [0,20], y_physical ∈ [0,20]
            x_coords = x_physical
            y_coords = y_physical

            # 生成2D坐标网格
            X_grid, Y_grid = np.meshgrid(x_coords, y_coords)

            # 区域E的掩码：[2,20] × [1,20] 在完整空间[0,20]×[0,20]内
            E_mask = (X_grid >= 2) & (X_grid <= 20) & (Y_grid >= 1) & (Y_grid <= 20)

            # 应用区域约束：在E外置零
            pattern = pattern * E_mask

        # 不归一化，保持原始系数范围[-3,3]以显示正确的密度值
        return pattern

    def visualize_basis_functions(self, save_path: Optional[str] = None):
        """
        可视化B₁和B₂基函数

        参数:
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        x = np.linspace(-1, 4, 1000)

        # B₁函数
        B1_vals = self.B1(x)
        axes[0].plot(x, B1_vals, 'b-', linewidth=2)
        axes[0].set_title('B₁(x) - 一阶基数B样条')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('B₁(x)')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(-0.1, 1.1)

        # B₂函数
        B2_vals = self.B2(x)
        axes[1].plot(x, B2_vals, 'r-', linewidth=2)
        axes[1].set_title('B₂(x) - 二阶基数B样条')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('B₂(x)')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-0.1, 1.1)

        # φ函数的二维可视化（选择kx=0, ky=0的情况）
        x_2d = np.linspace(-1, 4, 200)
        y_2d = np.linspace(-1, 3, 150)
        X_2d, Y_2d = np.meshgrid(x_2d, y_2d)
        phi_vals = self.phi(X_2d, Y_2d)

        im = axes[2].imshow(phi_vals, extent=[-1, 4, -1, 3], origin='lower', cmap='viridis')
        axes[2].set_title('φ(x,y) = B₂(x)B₁(y)')
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('y')
        plt.colorbar(im, ax=axes[2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


def create_cardinal_bspline_function(space: odl.DiscretizedSpace,
                                    coefficients: Optional[np.ndarray] = None,
                                    enforce_region_constraint: bool = True,
                                    random_seed: Optional[int] = None) -> odl.DiscretizedSpaceElement:
    """
    创建基数B样条函数，基于论文第五节的定义

    参数:
        space: ODL函数空间
        coefficients: 441个系数数组，如果为None则随机生成
        enforce_region_constraint: 是否强制在区域E=[2,20]×[1,20]外置零
        random_seed: 随机种子

    返回:
        包含基数B样条函数的ODL函数，在区域E外完全为0
    """
    generator = CardinalBSpline2D()
    pattern = generator.generate_cardinal_pattern(
        space.shape,
        coefficients,
        enforce_region_constraint=enforce_region_constraint,
        random_seed=random_seed
    )
    return space.element(pattern)


if __name__ == "__main__":
    # 示例用法和可视化
    visualize_box_splines(save_path='box_spline_examples.png')

    # 演示基数B样条
    cardinal = CardinalBSpline2D()
    cardinal.visualize_basis_functions(save_path='cardinal_bspline_basis.png')

    # 生成基数B样条函数示例
    import odl
    space = odl.uniform_discr([0, 0], [22, 22], [128, 128], dtype='float32')
    bspline_func = create_cardinal_bspline_function(space, random_seed=42)

    plt.figure(figsize=(8, 6))
    plt.imshow(bspline_func.asarray(), cmap='gray')
    plt.title('Cardinal B-spline Function (441 basis functions)')
    plt.axis('off')
    plt.colorbar()
    plt.savefig('cardinal_bspline_function.png', dpi=150, bbox_inches='tight')
    plt.close()