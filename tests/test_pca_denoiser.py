import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.pca_denoiser import PCAGradientDenoiser


def test_output_shape_matches_input():
    """Denoised gradient has same shape as input."""
    d = PCAGradientDenoiser(n_components=4, buffer_size=8)
    rng = np.random.default_rng(0)
    for _ in range(8):
        d.update(rng.standard_normal(13))
    out = d.denoise(rng.standard_normal(13))
    assert out.shape == (13,), f"Expected (13,), got {out.shape}"
    print("test_output_shape_matches_input passed")


def test_returns_input_when_buffer_not_full():
    """Before buffer fills, denoise returns input unchanged."""
    d = PCAGradientDenoiser(n_components=4, buffer_size=8)
    rng = np.random.default_rng(0)
    for _ in range(5):
        d.update(rng.standard_normal(13))
    g = rng.standard_normal(13)
    out = d.denoise(g)
    assert np.allclose(out, g), "Should return input unchanged before buffer full"
    print("test_returns_input_when_buffer_not_full passed")


def test_is_full_triggers_correctly():
    """is_full is False before N updates, True after."""
    d = PCAGradientDenoiser(n_components=4, buffer_size=8)
    rng = np.random.default_rng(0)
    for i in range(8):
        assert not d.is_full or i == 8
        d.update(rng.standard_normal(13))
    assert d.is_full, "Should be full after 8 updates"
    print("test_is_full_triggers_correctly passed")


def test_zero_scale_weights_returns_mean():
    """With all scale weights zero, output equals mean gradient."""
    d = PCAGradientDenoiser(n_components=4, buffer_size=8,
                             scale_weights=np.zeros(4))
    rng = np.random.default_rng(42)
    grads = [rng.standard_normal(13) for _ in range(8)]
    for g in grads:
        d.update(g)
    mean = np.mean(grads, axis=0)
    out = d.denoise(rng.standard_normal(13))
    assert np.allclose(out, mean, atol=1e-6), \
        f"Expected mean gradient, max diff={np.max(np.abs(out-mean)):.2e}"
    print("test_zero_scale_weights_returns_mean passed")


def test_denoising_reduces_noise():
    """PCA denoising reduces error vs true signal on noisy inputs."""
    rng = np.random.default_rng(42)
    true_dir = np.ones(13) / np.sqrt(13)
    sigma = 0.5

    d = PCAGradientDenoiser(n_components=2, buffer_size=8,
                             scale_weights=np.array([1.0, 0.0]))
    for i in range(8):
        g = true_dir * (1.0 - i*0.02) + rng.standard_normal(13) * sigma
        d.update(g)

    noisy = true_dir + rng.standard_normal(13) * sigma
    clean = d.denoise(noisy)

    err_noisy = np.linalg.norm(noisy - true_dir)
    err_clean = np.linalg.norm(clean - true_dir)
    assert err_clean < err_noisy, \
        f"Denoising made things worse: {err_noisy:.3f} -> {err_clean:.3f}"
    print("test_denoising_reduces_noise passed")


def test_gram_matrix_size_independent_of_D():
    """Buffer size bounds computation — works for large D."""
    d = PCAGradientDenoiser(n_components=4, buffer_size=8)
    rng = np.random.default_rng(0)
    D = 50000
    for _ in range(8):
        d.update(rng.standard_normal(D))
    out = d.denoise(rng.standard_normal(D))
    assert out.shape == (D,), f"Expected ({D},), got {out.shape}"
    print("test_gram_matrix_size_independent_of_D passed")


def test_n_components_must_be_leq_buffer_size():
    """n_components > buffer_size raises AssertionError."""
    try:
        d = PCAGradientDenoiser(n_components=9, buffer_size=8)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass
    print("test_n_components_must_be_leq_buffer_size passed")


def test_set_scale_weights_updates_correctly():
    """set_scale_weights changes the weights used in denoising."""
    d = PCAGradientDenoiser(n_components=4, buffer_size=8)
    new_weights = np.array([1.0, 0.5, 0.0, 0.0])
    d.set_scale_weights(new_weights)
    assert np.allclose(d.scale_weights, new_weights), \
        f"Weights not updated: {d.scale_weights}"
    print("test_set_scale_weights_updates_correctly passed")


def test_denoise_with_temporal_reduces_noise():
    """Two-stage denoising works and returns correct shape."""
    rng = np.random.default_rng(42)
    true_dir = np.ones(13) / np.sqrt(13)
    sigma = 0.3

    d = PCAGradientDenoiser(n_components=4, buffer_size=8,
                             scale_weights=np.array([1.0, 0.5, 0.1, 0.0]))
    for i in range(8):
        g = true_dir + rng.standard_normal(13) * sigma
        d.update(g)

    noisy = true_dir + rng.standard_normal(13) * sigma
    clean = d.denoise_with_temporal(noisy)

    assert clean.shape == (13,), f"Expected (13,), got {clean.shape}"
    err_noisy = np.linalg.norm(noisy - true_dir)
    err_clean = np.linalg.norm(clean - true_dir)
    assert err_clean < err_noisy, \
        f"Temporal denoising made things worse: {err_noisy:.3f} -> {err_clean:.3f}"
    print("test_denoise_with_temporal_reduces_noise passed")
    
def test_fd_runs_on_mnist():
    """FD sketch initialises, updates, and denoises without error."""
    from core.frequent_directions import FrequentDirections
    import numpy as np
    rng = np.random.default_rng(0)
    fd = FrequentDirections(sketch_size=8, D=100)
    for _ in range(20):
        fd.update(rng.standard_normal(100))
    assert fd.is_ready
    clean = fd.denoise(rng.standard_normal(100))
    assert clean.shape == (100,)
    print("test_fd_runs_on_mnist passed")


if __name__ == "__main__":
    test_output_shape_matches_input()
    test_returns_input_when_buffer_not_full()
    test_is_full_triggers_correctly()
    test_zero_scale_weights_returns_mean()
    test_denoising_reduces_noise()
    test_gram_matrix_size_independent_of_D()
    test_n_components_must_be_leq_buffer_size()
    test_set_scale_weights_updates_correctly()
    test_denoise_with_temporal_reduces_noise()
    test_fd_runs_on_mnist()
    print("\nAll PCA denoiser tests passed.")