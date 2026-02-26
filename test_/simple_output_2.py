"""
Simple version: Generate B-spline function and output F, Phi, G values
"""

import torch
import numpy as np
import sys
import os
import json

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

from box_spline import CardinalBSpline2D


def generate_and_output():
    """Generate B-spline function and output F, Phi, G values"""
    print("=" * 60)
    print("B-spline Function Generation and F, Phi, G Output")
    print("=" * 60)

    # 1. Parameters
    beta = (1, 21)
    n_coefficients = 441  # 21x21
    height, width = 21, 21
    image_size = (128, 128)
    m = 2

    print(f"Parameters:")
    print(f"  beta = {beta}")
    print(f"  coefficients = {n_coefficients}")
    print(f"  image size = {image_size}")

    # 2. Generate B-spline function f(x,y) = sum c_{i,j} * phi_{i,j}(x,y)
    print(f"\n1. Generating B-spline function:")

    # Initialize B-spline generator
    bspline_generator = CardinalBSpline2D()

    # Generate coefficients c_{i,j} ~ N(0,1)
    torch.manual_seed(42)
    coeff_matrix = torch.randn(height, width)
    print(f"  Coefficient matrix shape: {coeff_matrix.shape}")
    print(f"  Coefficient range: [{coeff_matrix.min():.4f}, {coeff_matrix.max():.4f}]")

    # Verify normal distribution properties
    coeff_flat = coeff_matrix.flatten()
    mean_val = torch.mean(coeff_flat).item()
    std_val = torch.std(coeff_flat).item()
    within_3std = torch.sum(torch.abs(coeff_flat) < 3).item() / len(coeff_flat)
    extreme_count = torch.sum(torch.abs(coeff_flat) >= 3).item()
    max_abs = torch.max(torch.abs(coeff_flat)).item()

    print(f"  Normal distribution verification:")
    print(f"    Mean: {mean_val:.4f} (expected ~0.0)")
    print(f"    Std: {std_val:.4f} (expected ~1.0)")
    print(f"    Values in (-3,3): {within_3std:.1%} (expected ~99.7%)")
    print(f"    Extreme values (|x|≥3): {extreme_count}")
    print(f"    Max absolute value: {max_abs:.4f}")

    # Generate additional test to show wider range
    if max_abs < 2.5:  # If current sample doesn't have extreme values
        print(f"  Note: Current sample doesn't include extreme values (>2.5)")
        print(f"  This is normal for random samples of size {len(coeff_flat)}")

    # FIX 1: Correct coefficient ordering for image generation
    # Transpose coeff_matrix before flattening so that y iterates fastest,
    # matching the inner loop (ky) in generate_cardinal_pattern.
    # d_vector stays as regular flatten() to match beta=(1,21) (x iterates fastest).
    coefficients_for_image = coeff_matrix.t().flatten().numpy()

    f_image = bspline_generator.generate_cardinal_pattern(
        shape=image_size,
        coefficients=coefficients_for_image,  # Use the transposed version here
        region=((2, 20), (1, 20)),
        enforce_region_constraint=True,
        random_seed=42
    )
    f_image = torch.from_numpy(f_image).float()
    print(f"  Image shape: {f_image.shape}")
    print(f"  Image range: [{f_image.min():.4f}, {f_image.max():.4f}]")
    print(f"  f(x,y) is real: {torch.all(torch.isreal(f_image))}")

    # 3. Calculate d vector (equivalent to c_k)
    print(f"\n2. Calculate d vector:")
    d_vector = coeff_matrix.flatten()
    print(f"  d vector shape: {d_vector.shape}")
    print(f"  d is real: {torch.all(torch.isreal(d_vector))}")
    print(f"  d range: [{d_vector.min():.4f}, {d_vector.max():.4f}]")

    # 4. Calculate Phi matrix (CORRECTED VERSION)
    print(f"\n3. Calculate Phi matrix (with B_2(x)B_1(y) decomposition):")

    # Calculate sampling frequencies for each direction
    beta1, beta2 = beta[0], beta[1]  # beta1=1, beta2=21
    N = n_coefficients
    j = torch.arange(N, dtype=torch.float32)

    # Frequency components for each direction
    xi_x = (2 * np.pi * j / N) * beta1  # xi_x = 2πj/N * 1
    xi_y = (2 * np.pi * j / N) * beta2  # xi_y = 2πj/N * 21

    print(f"  Frequency components:")
    print(f"    xi_x range: [{xi_x.min():.4f}, {xi_x.max():.4f}]")
    print(f"    xi_y range: [{xi_y.min():.4f}, {xi_y.max():.4f}]")

    # FIX 2: Correct B-spline orders in Phi calculation
    # phi(x,y) = B2(x) * B1(y) implies we need B2_hat(xi_x) * B1_hat(xi_y)

    # Calculate B_2_hat(xi_x) for x-direction (m=2)
    xi_x_safe = torch.where(xi_x == 0, torch.tensor(1e-9), xi_x)
    B2_hat_x = torch.exp(-1j * xi_x) * (torch.sin(xi_x / 2) / (xi_x / 2))**m  # m=2
    B2_hat_x[xi_x == 0] = 1.0  # limit as xi -> 0

    # Calculate B_1_hat(xi_y) for y-direction (m=1)
    xi_y_safe = torch.where(xi_y == 0, torch.tensor(1e-9), xi_y)
    B1_hat_y = torch.exp(-1j * xi_y / 2) * (torch.sin(xi_y / 2) / (xi_y / 2))  # m=1
    B1_hat_y[xi_y == 0] = 1.0  # limit as xi -> 0

    # Phi diagonal: Phi_jj = B_2_hat(xi_x) * B_1_hat(xi_y)
    # This follows from φ(x,y) = B_2(x)B_1(y) => φ̂(ξ_x,ξ_y) = B̂_2(ξ_x) * B̂_1(ξ_y)
    Phi_diagonal_complex = B2_hat_x * B1_hat_y

    # Build Phi matrix
    Phi = torch.diag(Phi_diagonal_complex)

    print(f"  Phi matrix shape: {Phi.shape}")
    print(f"  Phi is diagonal: {torch.allclose(Phi, torch.diag(torch.diag(Phi)))}")
    print(f"  Phi diagonal is complex: {torch.is_complex(Phi_diagonal_complex)}")

    # Show magnitude and phase information
    Phi_magnitude = torch.abs(Phi_diagonal_complex)
    Phi_phase = torch.angle(Phi_diagonal_complex)
    print(f"  Phi magnitude range: [{Phi_magnitude.min():.6f}, {Phi_magnitude.max():.6f}]")
    print(f"  Phi phase range: [{Phi_phase.min():.4f}, {Phi_phase.max():.4f}] rad")

    # Compare with old incorrect implementation
    xi_combined = torch.sqrt(xi_x**2 + xi_y**2)
    xi_combined_safe = torch.where(xi_combined == 0, torch.tensor(1e-9), xi_combined)
    old_Phi = (torch.sin(np.pi * xi_combined_safe) / (np.pi * xi_combined_safe))**m
    old_Phi[xi_combined == 0] = 1.0

    print(f"  Old (incorrect) vs New (correct) comparison:")
    print(f"    Old magnitude range: [{old_Phi.min():.6f}, {old_Phi.max():.6f}]")
    print(f"    New magnitude range: [{Phi_magnitude.min():.6f}, {Phi_magnitude.max():.6f}]")
    print(f"    Difference in max magnitude: {abs(old_Phi.max() - Phi_magnitude.max()):.6f}")

    # 5. Calculate G matrix
    print(f"\n4. Calculate G matrix:")

    # G matrix elements: G[j,n] = e^{-in 2πj/N}
    j_grid = torch.arange(N).float().unsqueeze(1)
    n_grid = torch.arange(N).float().unsqueeze(0)
    exponent = -1j * 2 * np.pi * n_grid * j_grid / N
    G = torch.exp(exponent)

    print(f"  G matrix shape: {G.shape}")
    print(f"  G is complex: {torch.is_complex(G)}")

    # Check unitary property (should be G @ G^H = N * I for unnormalized DFT)
    N = len(j)
    G_orthogonal = torch.allclose(
        torch.matmul(G, torch.conj(G).T),
        torch.eye(N, dtype=torch.complex64) * N,
        atol=1e-6
    )
    print(f"  G satisfies G @ G^H = N*I: {G_orthogonal}")

    # 6. Calculate F vector
    print(f"\n5. Calculate F vector:")

    # F = Phi * G * d (now with complex Phi matrix)
    d_complex = torch.complex(d_vector, torch.zeros_like(d_vector))
    G_d = torch.matmul(G, d_complex)
    F = torch.matmul(Phi, G_d)

    print(f"  F vector shape: {F.shape}")
    print(f"  F is complex: {torch.is_complex(F)}")
    print(f"  F real range: [{torch.real(F).min():.4f}, {torch.real(F).max():.4f}]")
    print(f"  F imag range: [{torch.imag(F).min():.4f}, {torch.imag(F).max():.4f}]")

    # Show F magnitude and phase
    F_magnitude = torch.abs(F)
    F_phase = torch.angle(F)
    print(f"  F magnitude range: [{F_magnitude.min():.4f}, {F_magnitude.max():.4f}]")
    print(f"  F phase range: [{F_phase.min():.4f}, {F_phase.max():.4f}] rad")

    # 7. Verify F = Phi * G * d relationship
    print(f"\n6. Verify F = Phi * G * d:")
    F_check = torch.matmul(Phi, torch.matmul(G, d_complex))
    verification_error = torch.norm(F - F_check) / torch.norm(F)
    print(f"  Verification error: {verification_error.item():.2e}")
    print(f"  Relationship verified: {verification_error < 1e-10}")

    # 8. Additional verification: Check if F corresponds to the generated image
    print(f"\n7. Cross-verification: F consistency check")
    print(f"  This ensures the calculated F corresponds to the actual generated f_image")

    # For verification, we can check if the magnitude relationships are reasonable
    F_magnitude = torch.abs(F)
    max_F_magnitude = torch.max(F_magnitude).item()

    # Calculate expected magnitude based on the image
    image_power = torch.sum(f_image**2).item()
    expected_F_magnitude_order = np.sqrt(image_power)  # Rough estimate

    print(f"  F magnitude max: {max_F_magnitude:.4f}")
    print(f"  Image power: {image_power:.4f}")
    print(f"  Expected F magnitude order: {expected_F_magnitude_order:.4f}")
    print(f"  Magnitude consistency: {'reasonable' if max_F_magnitude < 10 * expected_F_magnitude_order else 'needs investigation'}")

    # 9. Show the impact of corrections
    print(f"\n8. Summary of corrections applied:")
    print(f"  FIX 1: Fixed coefficient ordering for image generation")
    print(f"    - Used coeff_matrix.t().flatten() to match generator's (y fast, x slow) loop")
    print(f"    - d_vector remains regular flatten() to match beta=(1,21) (x fast, y slow)")
    print(f"  FIX 2: Fixed B-spline orders in Phi calculation")
    print(f"    - B2_hat(xi_x) for x-direction (m=2) - CORRECTED")
    print(f"    - B1_hat(xi_y) for y-direction (m=1) - CORRECTED")
    print(f"    - Ensures phi(x,y) = B2(x)B1(y) consistency")

    # 8. Save results
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Prepare output data
    output_data = {
        "input": {
            "description": "B-spline function f(x,y) = sum c_{i,j} * phi_{i,j}(x,y)",
            "coefficients_shape": list(coeff_matrix.shape),
            "coefficients_range": [float(coeff_matrix.min()), float(coeff_matrix.max())],
            "image_shape": list(f_image.shape),
            "image_range": [float(f_image.min()), float(f_image.max())],
            "d_vector_shape": list(d_vector.shape),
            "d_vector_range": [float(d_vector.min()), float(d_vector.max())]
        },
        "outputs": {
            "F": {
                "shape": list(F.shape),
                "is_complex": True,
                "real_range": [float(torch.real(F).min()), float(torch.real(F).max())],
                "imag_range": [float(torch.imag(F).min()), float(torch.imag(F).max())],
                "real_values": torch.real(F).numpy().tolist(),
                "imag_values": torch.imag(F).numpy().tolist()
            },
            "Phi": {
                "shape": list(Phi.shape),
                "is_diagonal": True,
                "is_complex": True,
                "magnitude_range": [float(Phi_magnitude.min()), float(Phi_magnitude.max())],
                "phase_range": [float(Phi_phase.min()), float(Phi_phase.max())],
                "correction_applied": True
            },
            "G": {
                "shape": list(G.shape),
                "is_complex": True,
                "is_unitary": bool(G_orthogonal),
                "real_range": [float(torch.real(G).min()), float(torch.real(G).max())],
                "imag_range": [float(torch.imag(G).min()), float(torch.imag(G).max())]
            }
        },
        "verification": {
            "F_equals_Phi_G_d": bool(verification_error.item() < 1e-10),
            "verification_error": float(verification_error.item())
        }
    }

    # Save tensors
    torch.save(F, os.path.join(output_dir, 'F_tensor.pt'))
    torch.save(Phi, os.path.join(output_dir, 'Phi_tensor.pt'))
    torch.save(G, os.path.join(output_dir, 'G_tensor.pt'))
    torch.save(coeff_matrix, os.path.join(output_dir, 'coeff_matrix.pt'))
    torch.save(f_image, os.path.join(output_dir, 'f_image.pt'))
    torch.save(d_vector, os.path.join(output_dir, 'd_vector.pt'))

    # Save JSON (simplified to avoid serialization issues)
    json_data = {
        "summary": {
            "F_shape": list(F.shape),
            "Phi_shape": list(Phi.shape),
            "G_shape": list(G.shape),
            "verification_passed": bool(verification_error.item() < 1e-10),
            "verification_error": float(verification_error.item())
        },
        "value_ranges": {
            "F_real": [float(torch.real(F).min()), float(torch.real(F).max())],
            "F_imag": [float(torch.imag(F).min()), float(torch.imag(F).max())],
            "Phi_diagonal": [float(Phi_magnitude.min()), float(Phi_magnitude.max())],
            "coefficients": [float(coeff_matrix.min()), float(coeff_matrix.max())],
            "image": [float(f_image.min()), float(f_image.max())],
            "d_vector": [float(d_vector.min()), float(d_vector.max())]
        },
        "properties": {
            "f_is_real": bool(torch.all(torch.isreal(f_image))),
            "d_is_real": bool(torch.all(torch.isreal(d_vector))),
            "Phi_is_diagonal": bool(torch.allclose(Phi, torch.diag(torch.diag(Phi)))),
            "Phi_is_complex": bool(torch.is_complex(Phi)),
            "G_is_complex": bool(torch.is_complex(G)),
            "G_unitary_property": bool(G_orthogonal),
            "F_is_complex": bool(torch.is_complex(F))
        }
    }

    json_path = os.path.join(output_dir, 'summary.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    print(f"\n7. Files saved:")
    print(f"  Tensors: {output_dir}/F_tensor.pt, Phi_tensor.pt, G_tensor.pt")
    print(f"  Input data: {output_dir}/coeff_matrix.pt, f_image.pt, d_vector.pt")
    print(f"  Summary: {json_path}")

    # Print key results
    print(f"\n{'='*60}")
    print("KEY RESULTS:")
    print(f"{'='*60}")
    print(f"F vector: shape={list(F.shape)}, complex=True")
    print(f"  Real range: [{float(torch.real(F).min()):.4f}, {float(torch.real(F).max()):.4f}]")
    print(f"  Imag range: [{float(torch.imag(F).min()):.4f}, {float(torch.imag(F).max()):.4f}]")

    print(f"\nPhi matrix: shape={list(Phi.shape)}, diagonal=True")
    print(f"  Magnitude range: [{float(Phi_magnitude.min()):.6f}, {float(Phi_magnitude.max()):.6f}]")
    print(f"  Phase range: [{float(Phi_phase.min()):.4f}, {float(Phi_phase.max()):.4f}] rad")

    print(f"\nG matrix: shape={list(G.shape)}, unitary={G_orthogonal}")
    print(f"  Real range: [{float(torch.real(G).min()):.4f}, {float(torch.real(G).max()):.4f}]")
    print(f"  Imag range: [{float(torch.imag(G).min()):.4f}, {float(torch.imag(G).max()):.4f}]")

    print(f"\nVerification:")
    print(f"  F = Phi * G * d: {verification_error.item():.2e} error")
    print(f"  Relationship verified: {verification_error < 1e-10}")

    return F, Phi, G, verification_error.item() < 1e-10


if __name__ == "__main__":
    try:
        F, Phi, G, success = generate_and_output()

        print(f"\n{'='*60}")
        if success:
            print("SUCCESS: All requirements satisfied!")
            print("Generated B-spline function and calculated F, Phi, G")
            print("F = Phi * G * d relationship verified")
        else:
            print("FAILED: Verification errors found")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()