"""Minimal Cardinal B-spline utilities (only what train/test need)."""

import numpy as np


class CardinalBSpline2D:
    """Cardinal B-spline utilities for pixel basis ? = B1(x) * B1(y)."""

    def __init__(self):
        self.support_x = (0, 1)
        self.support_y = (0, 1)

    def B1(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x, dtype=float)
        mask1 = (x > 0) & (x <= 1)
        result[mask1] = 1.0
        return result

    def B1_hat_complex(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=np.float64)
        return np.exp(-1j * xi / 2.0) * np.sinc(xi / (2.0 * np.pi))

    def phi(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.B1(x) * self.B1(y)
