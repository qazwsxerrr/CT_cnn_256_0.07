import os
import torch
import numpy as np
from tqdm import tqdm
from config import DATA_DIR, THEORETICAL_CONFIG, DATA_CONFIG, IMAGE_SIZE
# [修改] 导入 SheppLoganGenerator
from radon_transform import TheoreticalDataGenerator, SheppLoganGenerator


def generate_and_save_dataset(filename, num_samples, generator, desc="Generating"):
    """
    生成数据并保存为 .pt 文件
    保存内容: (coeff_true, F_observed, coeff_initial)
    """
    print(f"\nStarted {desc}...")
    print(f"Target Samples: {num_samples}")

    coeff_true_list = []
    F_observed_list = []
    coeff_initial_list = []

    # 估算显存/内存占用 (每个样本约 0.26 MB)
    est_mem_mb = num_samples * 0.26
    print(f"Estimated RAM usage for this dataset: ~{est_mem_mb:.2f} MB (~{est_mem_mb / 1024:.2f} GB)")
    if est_mem_mb > 12000:
        print("WARNING: Dataset size is very large (>12GB). Ensure you have >32GB RAM!")

    for i in tqdm(range(num_samples), desc=desc):
        # 使用随机种子确保可复现性，但每个样本都不一样
        seed = i if desc == "Train" else i + 1000000

        c_true, _, F_obs, c_init = generator.generate_training_sample(random_seed=seed)

        # 移至 CPU 以节省显存并准备保存
        coeff_true_list.append(c_true.detach().cpu())
        F_observed_list.append(F_obs.detach().cpu())
        coeff_initial_list.append(c_init.detach().cpu())

    # 堆叠为大 Tensor
    data_dict = {
        "coeff_true": torch.stack(coeff_true_list).unsqueeze(1),  # [N, 1, IMAGE_SIZE, IMAGE_SIZE]
        "F_observed": torch.stack(F_observed_list),  # [N, IMAGE_SIZE*IMAGE_SIZE] complex
        "coeff_initial": torch.stack(coeff_initial_list).unsqueeze(1)  # [N, 1, IMAGE_SIZE, IMAGE_SIZE]
    }

    save_path = os.path.join(DATA_DIR, filename)
    torch.save(data_dict, save_path)
    print(f"Saved {num_samples} samples to {save_path}")
    print(f"Final Tensor shape (GT): {data_dict['coeff_true'].shape}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing Generator on {device}...")

    # 初始化生成器 (这一步会把物理算子 Phi 放进 GPU)
    generator = TheoreticalDataGenerator()

    # 1. 生成训练集: 20,000 个 (保持默认的随机椭圆)
    # 训练时我们需要多样性，所以保持随机生成
    generate_and_save_dataset(
        "train_dataset.pt",
        20000,
        generator,
        desc="Generating Train Set (Random Ellipses)"
    )

    # [关键修改] 2. 生成验证集: 2,000 个 (使用 Shepp-Logan)
    # 验证时我们需要标准答案，所以替换生成器
    print("\n[Switching to Shepp-Logan Generator for Validation Data]")
    generator.phantom_gen = SheppLoganGenerator(size=IMAGE_SIZE)

    generate_and_save_dataset(
        "val_dataset.pt",
        2000,
        generator,
        desc="Generating Val Set (Shepp-Logan)"
    )

    print("\nAll data generated successfully!")
    print("Now run: python train_offline.py")


if __name__ == "__main__":
    main()
