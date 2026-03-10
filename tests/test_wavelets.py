import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.wavelets import haar_forward, haar_inverse, wavelet_denoise, make_scale_weight_vector, sigmoid

# =============================================================
# Unit tests for Haar wavelet transforms
#
# These tests encode mathematical properties that must hold
# regardless of the input signal. If Session 4 changes the
# wavelet implementation, these tests will catch regressions.
# =============================================================

def test_energy_preservation():
    """
    ||haar_forward(signal)||² == ||signal||²
    The transform is orthonormal — it preserves L2 energy.
    """
    np.random.seed(42)
    for length in [2, 4, 8, 16, 32]:
        signal = np.random.randn(length)
        coeffs = haar_forward(signal)
        assert np.isclose(np.sum(signal**2), np.sum(coeffs**2)), \
            f"Energy not preserved for length={length}"

    print("test_energy_preservation passed")


def test_perfect_reconstruction():
    """
    haar_inverse(haar_forward(signal)) == signal exactly.
    Forward then inverse is the identity operation.
    """
    np.random.seed(0)
    for length in [2, 4, 8, 16]:
        signal = np.random.randn(length)
        reconstructed = haar_inverse(haar_forward(signal))
        assert np.allclose(signal, reconstructed), \
            f"Reconstruction failed for length={length}"

    print("test_perfect_reconstruction passed")


def test_constant_signal_has_zero_details():
    """
    A constant signal has zero detail coefficients at all scales.
    All information lives in coeffs[0] — the approximation.
    """
    for length in [4, 8, 16]:
        signal = np.ones(length) * 5.0
        coeffs = haar_forward(signal)

        # All detail coefficients should be zero
        assert np.allclose(coeffs[1:], 0.0), \
            f"Non-zero details for constant signal, length={length}"

        # Approximation encodes the mean: coeffs[0] = mean * sqrt(N)
        expected_c0 = 5.0 * np.sqrt(length)
        assert np.isclose(coeffs[0], expected_c0), \
            f"Wrong approximation for constant signal: {coeffs[0]} != {expected_c0}"

    print("test_constant_signal_has_zero_details passed")


def test_coarsest_coefficient_is_scaled_mean():
    """
    coeffs[0] == mean(signal) * sqrt(N)
    This is always true for the Haar transform.
    """
    np.random.seed(7)
    for length in [4, 8, 16]:
        signal = np.random.randn(length)
        coeffs = haar_forward(signal)
        expected = np.mean(signal) * np.sqrt(length)
        assert np.isclose(coeffs[0], expected), \
            f"coeffs[0]={coeffs[0]:.4f} != mean*sqrt(N)={expected:.4f}"

    print("test_coarsest_coefficient_is_scaled_mean passed")


def test_invalid_length_raises():
    """Non-power-of-2 inputs should raise ValueError."""
    for bad_length in [3, 5, 6, 7, 9, 12]:
        signal = np.ones(bad_length)
        try:
            haar_forward(signal)
            assert False, f"Should have raised for length={bad_length}"
        except ValueError:
            pass

    print("test_invalid_length_raises passed")


def test_denoise_preserves_constant_signal():
    """
    Denoising a constant signal should return the same value
    regardless of threshold — constant signals have no detail
    coefficients to threshold.
    """
    signal = np.ones(8) * 3.0
    for threshold in [0.1, 1.0, 5.0]:
        denoised, _, _ = wavelet_denoise(signal, threshold)
        assert np.isclose(denoised, 3.0), \
            f"Constant signal denoised incorrectly: {denoised} != 3.0"

    print("test_denoise_preserves_constant_signal passed")


def test_denoise_last_value_is_current_step():
    """
    wavelet_denoise returns the last value of the reconstructed signal.
    For a signal with no detail (constant), that's the constant value.
    For a ramp signal, it's the last value of the ramp.
    """
    # Ramp signal: [0, 1, 2, 3, 4, 5, 6, 7]
    signal = np.arange(8, dtype=float)
    denoised, _, _ = wavelet_denoise(signal, threshold=0.0)  # threshold=0 keeps everything
    # With threshold=0, all coefficients survive → perfect reconstruction
    # Last value should be 7.0
    assert np.isclose(denoised, 7.0), \
        f"Last value should be 7.0, got {denoised}"

    print("test_denoise_last_value_is_current_step passed")
    
def test_make_scale_weight_vector_shape():
    """Output shape matches signal length."""
    for n in [4, 8, 16]:
        import math
        J = int(math.log2(n))
        raw_params = np.zeros(J)
        weights = make_scale_weight_vector(raw_params, n)
        assert weights.shape == (n,), f"Wrong shape for n={n}"
        assert weights[0] == 1.0, "Approximation weight must be 1.0"

    print("test_make_scale_weight_vector_shape passed")


def test_zero_raw_params_gives_half_weights():
    """sigmoid(0) = 0.5 for all detail scales."""
    raw_params = np.zeros(3)
    weights = make_scale_weight_vector(raw_params, 8)
    assert weights[0] == 1.0
    assert np.allclose(weights[1:], 0.5), \
        f"Expected all 0.5, got {weights[1:]}"

    print("test_zero_raw_params_gives_half_weights passed")

def test_idwt_last_row_cache():
    """Cache returns same result as direct computation."""
    from core.wavelets import get_idwt_last_row
    import math
    for n in [4, 8, 16]:
        # Call twice — second should use cache
        row1 = get_idwt_last_row(n)
        row2 = get_idwt_last_row(n)
        assert np.allclose(row1, row2), f"Cache inconsistent for n={n}"
        assert len(row1) == n, f"Wrong length for n={n}"
    print("test_idwt_last_row_cache passed")


def test_dgx_clean_dweightvec_fast_matches_slow():
    """Fast vectorized version matches original loop version."""
    from core.wavelets import dgx_clean_dweightvec, dgx_clean_dweightvec_fast
    np.random.seed(0)
    for n in [4, 8, 16]:
        sig = np.random.randn(n)
        weight_vec = np.random.rand(n)
        slow = dgx_clean_dweightvec(sig, weight_vec)
        fast = dgx_clean_dweightvec_fast(sig)
        assert np.allclose(slow, fast), \
            f"Mismatch for n={n}: max diff={np.max(np.abs(slow-fast)):.2e}"
    print("test_dgx_clean_dweightvec_fast_matches_slow passed")

if __name__ == "__main__":
    test_energy_preservation()
    test_perfect_reconstruction()
    test_constant_signal_has_zero_details()
    test_coarsest_coefficient_is_scaled_mean()
    test_invalid_length_raises()
    test_denoise_preserves_constant_signal()
    test_denoise_last_value_is_current_step()
    test_make_scale_weight_vector_shape()
    test_zero_raw_params_gives_half_weights()
    test_idwt_last_row_cache()
    test_dgx_clean_dweightvec_fast_matches_slow()
    print("\nAll wavelet tests passed.")
    
