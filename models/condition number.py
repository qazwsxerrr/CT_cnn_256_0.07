"""
Script to calculate the condition numbers of matrices Phi, G, and Phi*G
based on the provided theoretical framework.
"""

import torch
import numpy as np
import sys
import os

# Ensure we can import modules from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from radon_transform import FourierOperatorCalculator
from config import THEORETICAL_CONFIG, IMAGE_SIZE
from box_spline import CardinalBSpline2D

def compute_full_scale_condition_number():
    """
    Computes the condition number for the actual problem size (e.g., 256x256).
    Note: We cannot construct the full G matrix (256^2 x 256^2) in memory,
    so we rely on the theoretical property that k(Phi G) = k(Phi).
    """
    print(f"\n{'='*60}")
    print(f"FULL SCALE CALCULATION (Image Size: {IMAGE_SIZE}x{IMAGE_SIZE})")
    print(f"{'='*60}")

    N = IMAGE_SIZE * IMAGE_SIZE

    # Initialize the calculator to get the Phi diagonal
    print("Initializing Fourier Operator...")
    calc = FourierOperatorCalculator(
        beta=THEORETICAL_CONFIG["beta_vector"],
        n_coefficients=N,
        m=2
    )

    # Phi is stored as a 1D diagonal vector in the code
    phi_diag = calc.Phi_diagonal

    # 1. Singular Values of Phi
    # Since Phi is diagonal, its singular values are simply the magnitudes of its diagonal entries.
    singular_values = torch.abs(phi_diag)

    sigma_max = torch.max(singular_values).item()
    sigma_min = torch.min(singular_values).item()

    print(f"\n[Matrix Phi]")
    print(f"  Max Singular Value (sigma_max): {sigma_max:.6e}")
    print(f"  Min Singular Value (sigma_min): {sigma_min:.6e}")

    if sigma_min == 0:
        k_phi = float('inf')
        print(f"  Condition Number k(Phi): Infinity (singular)")
    else:
        k_phi = sigma_max / sigma_min
        print(f"  Condition Number k(Phi): {k_phi:.6e}")

    # 2. Condition Number of G (FFT Matrix)
    print(f"\n[Matrix G (FFT Matrix)]")
    print("  Theoretical Analysis:")
    print("  The matrix G is the unnormalized DFT matrix.")
    print("  G_{jk} = exp(-i * 2pi * j * k / N)")
    print("  Properties: G * G^H = N * I")
    print("  Eigenvalues of G * G^H are all equal to N.")
    print("  Singular values of G are sqrt(N).")
    print(f"  Max Singular Value: {np.sqrt(N):.6e}")
    print(f"  Min Singular Value: {np.sqrt(N):.6e}")
    print(f"  Condition Number k(G): 1.000000e+00")

    # 3. Condition Number of Phi * G
    print(f"\n[Matrix Phi * G]")
    print("  Theoretical Analysis:")
    print("  Let A = Phi * G.")
    print("  Since G is a scaled unitary matrix (up to factor sqrt(N)),")
    print("  multiplication by G does not change the condition number of the diagonal matrix Phi.")
    print("  Alternatively: A * A^H = Phi * G * G^H * Phi^H = N * Phi * Phi^H.")
    print("  The singular values of A are sqrt(N) * |Phi_ii|.")
    print("  The ratio (condition number) cancels out the sqrt(N) factor.")
    print(f"  Therefore, k(Phi G) = k(Phi) = {k_phi:.6e}")


def verify_on_small_scale():
    """
    Performs a numerical verification on a small matrix where we CAN
    construct the full G matrix and compute SVD explicitly.
    """
    print(f"\n\n{'='*60}")
    print(f"SMALL SCALE VERIFICATION (N=100)")
    print(f"{'='*60}")

    N_small = 100
    # Use arbitrary beta for verification
    beta_small = (1, 10)

    print(f"Constructing matrices for N={N_small}...")

    # --- Reconstruct Phi diagonal manually for this small size ---
    # Logic copied from FourierOperatorCalculator
    freq = torch.fft.fftfreq(N_small, d=1.0).to(torch.float64)
    arg_x = (2.0 * np.pi * freq) * beta_small[0]
    arg_y = (2.0 * np.pi * freq) * beta_small[1]

    bspline = CardinalBSpline2D()
    B1_x = torch.from_numpy(np.asarray(bspline.B1_hat_complex(arg_x.numpy()), dtype=np.complex128))
    B1_y = torch.from_numpy(np.asarray(bspline.B1_hat_complex(arg_y.numpy()), dtype=np.complex128))
    phi_diag_small = B1_x * B1_y

    # Create the diagonal matrix Phi
    Phi_mat = torch.diag(phi_diag_small)

    # --- Construct G matrix explicitly ---
    # G_jk = exp(-i * 2pi * j * k / N)
    # Note: Theoretical.md definition implies unnormalized DFT
    indices = torch.arange(N_small, dtype=torch.float64)
    j_grid, k_grid = torch.meshgrid(indices, indices, indexing='ij')
    G_mat = torch.exp(-1j * 2 * np.pi * j_grid * k_grid / N_small).to(torch.complex128)

    # --- Compute Product A = Phi * G ---
    A_mat = torch.matmul(Phi_mat, G_mat)

    # --- Compute Condition Numbers via SVD ---
    try:
        # SVD of Phi
        # We know SVs are abs(diag), but let's use linalg.svd to be sure
        _, S_phi, _ = torch.linalg.svd(Phi_mat)
        k_phi_computed = S_phi.max() / S_phi.min()

        # SVD of G
        _, S_G, _ = torch.linalg.svd(G_mat)
        k_G_computed = S_G.max() / S_G.min()

        # SVD of Phi * G
        _, S_A, _ = torch.linalg.svd(A_mat)
        k_A_computed = S_A.max() / S_A.min()

        print(f"\nResults:")
        print(f"  Computed k(Phi):   {k_phi_computed.item():.6f}")
        print(f"  Computed k(G):     {k_G_computed.item():.6f}")
        print(f"  Computed k(Phi G): {k_A_computed.item():.6f}")

        # Verify k(G) is approx 1
        if abs(k_G_computed.item() - 1.0) < 1e-5:
             print("  >> VERIFICATION SUCCESSFUL: k(G) approx 1.0")
        else:
             print("  >> VERIFICATION FAILED: k(G) not 1.0")

        # Verify k(Phi) approx k(Phi G)
        diff = abs(k_phi_computed - k_A_computed).item()
        print(f"  Diff(Phi, Phi G):  {diff:.6e}")

        if diff < 1e-5:
            print("  >> VERIFICATION SUCCESSFUL: k(Phi) approx k(Phi G)")
        else:
            print("  >> VERIFICATION FAILED: k(Phi) != k(Phi G)")

    except Exception as e:
        print(f"  SVD Calculation failed: {e}")

if __name__ == "__main__":
    compute_full_scale_condition_number()
    verify_on_small_scale()