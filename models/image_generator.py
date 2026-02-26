import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from config import IMAGE_SIZE


class DifferentiableImageGenerator(nn.Module):
    """
    If coefficient grid matches image size, coefficients are pixels; otherwise bilinear resize.
    """

    def __init__(self, image_size: int = IMAGE_SIZE, coeff_grid: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)):
        super().__init__()
        self.H = image_size
        self.W = image_size
        if isinstance(coeff_grid, int):
            self.grid_H = self.grid_W = coeff_grid
        else:
            self.grid_H, self.grid_W = coeff_grid
        self.is_identity = (self.H == self.grid_H) and (self.W == self.grid_W)

    def forward(self, coeff_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coeff_batch: (B,1,H,W), (B,H,W) or (B, H*W)
        Returns:
            images: (B,1,H,W)
        """
        if coeff_batch.dim() == 2:
            b = coeff_batch.shape[0]
            coeff_batch = coeff_batch.view(b, 1, self.grid_H, self.grid_W)
        elif coeff_batch.dim() == 3:
            coeff_batch = coeff_batch.unsqueeze(1)

        if self.is_identity:
            return coeff_batch

        return F.interpolate(
            coeff_batch, size=(self.H, self.W), mode="bilinear", align_corners=True
        )
