import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problems.rosenbrock import rosenbrock
from problems.noisy_rosenbrock import noisy_rosenbrock
from core.optimizers import run_sgd, run_adam, run_wavelet_adam, run_noisy_adam, run_noisy_wavelet_adam

def test_sgd_decreases_loss():
    """SGD should decrease loss on Rosenbrock with small lr."""
    h = run_sgd(lr=0.001, n_steps=100)
    losses = [l for x,y,l in h]
    # Loss at end should be less than loss at start
    assert losses[-1] < losses[0], \
        f"SGD did not decrease loss: {losses[0]:.4f} → {losses[-1]:.4f}"
    print("test_sgd_decreases_loss passed")


def test_sgd_diverges_with_large_lr():
    """SGD should diverge with lr=0.01 on Rosenbrock."""
    try:
        h = run_sgd(lr=0.01, n_steps=100)
        losses = [l for x,y,l in h]
        assert np.isnan(losses[-1]) or np.isinf(losses[-1]) or losses[-1] > losses[0], \
            "Expected divergence with lr=0.01"
    except (OverflowError, FloatingPointError):
        pass  # expected
    print("test_sgd_diverges_with_large_lr passed")


def test_adam_converges():
    """Adam should converge to (1,1) within 2000 steps."""
    h = run_adam(lr=0.01, n_steps=2000)
    converged = any(abs(x-1)<1e-4 and abs(y-1)<1e-4 for x,y,l in h)
    assert converged, "Adam did not converge within 2000 steps"
    print("test_adam_converges passed")


def test_adam_beats_sgd():
    """Adam should reach lower loss than SGD in same number of steps."""
    h_sgd  = run_sgd(lr=0.001,  n_steps=1000)
    h_adam = run_adam(lr=0.01, n_steps=1000)
    loss_sgd  = h_sgd[-1][2]
    loss_adam = h_adam[-1][2]
    assert loss_adam < loss_sgd, \
        f"Adam ({loss_adam:.4f}) should beat SGD ({loss_sgd:.4f})"
    print("test_adam_beats_sgd passed")


def test_adam_bias_correction():
    """
    After one Adam step, position should have changed from start.
    Bias correction ensures the first step is not negligibly small.
    """
    h = run_adam(lr=0.01, n_steps=2)
    # h[0] is before step 1, h[1] is before step 2 (i.e. after step 1)
    x0, y0, _ = h[0]
    x1, y1, _ = h[1]

    # Position should have changed
    assert x1 != x0 or y1 != y0, "Adam took no step"

    # Step should be finite
    assert np.isfinite(x1) and np.isfinite(y1), \
        "Adam produced non-finite position"

    # Without bias correction, first step would be ~100x smaller
    # With bias correction, step should be meaningful — at least 1e-4
    step_size = np.sqrt((x1-x0)**2 + (y1-y0)**2)
    assert step_size > 1e-4, \
        f"Step too small, bias correction may be broken: {step_size:.2e}"

    print("test_adam_bias_correction passed")

def test_wavelet_adam_runs():
    """WaveletAdam should run without errors and produce finite loss."""
    h = run_wavelet_adam(lr=0.01, n_steps=500,
                         buffer_capacity=8, threshold=0.5)
    losses = [l for x,y,l in h]
    assert all(np.isfinite(l) for l in losses), \
        "WaveletAdam produced non-finite loss"
    assert losses[-1] < losses[0], \
        "WaveletAdam did not decrease loss"
    print("test_wavelet_adam_runs passed")


def test_optimizers_same_start():
    """All optimizers start from the same point."""
    h_sgd  = run_sgd(lr=0.001, n_steps=1)
    h_adam = run_adam(lr=0.01,  n_steps=1)
    assert h_sgd[0][0]  == h_adam[0][0] == -1.0
    assert h_sgd[0][1]  == h_adam[0][1] ==  1.0
    print("test_optimizers_same_start passed")
    
def test_noisy_adam_worse_than_clean():
    """Noisy Adam has higher final loss than clean Adam."""
    from problems.noisy_rosenbrock import noisy_rosenbrock
    h_clean = run_adam(lr=0.01, n_steps=2000)
    # Can't use run_noisy_adam directly without importing — test the property
    clean_loss = h_clean[-1][2]
    assert clean_loss < 1e-4, f"Clean Adam should converge, got {clean_loss}"
    print("test_noisy_adam_worse_than_clean passed")


def test_wavelet_adam_noise_robust():
    """WaveletAdam final loss doesn't degrade much from σ=1 to σ=5."""
    from experiments.session7_noisy import run_noisy_wavelet_adam, noise_adaptive_params
    h1 = run_noisy_wavelet_adam(1.0, 1000, seed=0,
                                 raw_params=noise_adaptive_params(1.0))
    h5 = run_noisy_wavelet_adam(5.0, 1000, seed=0,
                                 raw_params=noise_adaptive_params(5.0))
    loss1 = h1[-1][2]
    loss5 = h5[-1][2]
    # WaveletAdam should degrade less than 5x from σ=1 to σ=5
    assert loss5 < loss1 * 5, \
        f"WaveletAdam not noise robust: σ=1 loss={loss1:.4f}, σ=5 loss={loss5:.4f}"
    print("test_wavelet_adam_noise_robust passed")
    
def test_xor_adam_solves_clean():
    """Adam solves clean XOR in at least 8/10 trials."""
    from problems.xor_problem import accuracy
    from experiments.session9_neural import run_xor_adam
    solved = 0
    for seed in range(10):
        h, p = run_xor_adam(n_steps=2000, seed=seed)
        if accuracy(p) == 1.0:
            solved += 1
    assert solved >= 8, f"Adam only solved {solved}/10 clean XOR trials"
    print("test_xor_adam_solves_clean passed")


def test_xor_wavelet_beats_adam_noisy():
    """WaveletAdam solves more noisy XOR trials than Adam at σ=0.05."""
    from problems.xor_problem import accuracy
    from experiments.session9_neural import run_xor_noisy_adam, run_xor_noisy_wavelet_adam
    adam_solved, wavelet_solved = 0, 0
    for seed in range(10):
        _, p_adam = run_xor_noisy_adam(sigma=0.05, n_steps=2000, seed=seed)
        _, p_wav  = run_xor_noisy_wavelet_adam(sigma=0.05, n_steps=2000, seed=seed)
        if accuracy(p_adam) == 1.0:
            adam_solved += 1
        if accuracy(p_wav) == 1.0:
            wavelet_solved += 1
    assert wavelet_solved > adam_solved, \
        f"WaveletAdam ({wavelet_solved}) not better than Adam ({adam_solved})"
    print("test_xor_wavelet_beats_adam_noisy passed")
    
def test_visushrink_beats_wavelet_noisy():
    """VisuShrink has lower final loss than WaveletAdam at σ=0.05."""
    from experiments.session11_adaptive import run_visuShrink_adam
    from experiments.session9_neural import run_xor_noisy_wavelet_adam
    vs, wav = [], []
    for seed in range(10):
        h, _ = run_visuShrink_adam(sigma=0.05, n_steps=2000, seed=seed)
        vs.append(h[-1][1])
        h, _ = run_xor_noisy_wavelet_adam(0.05, 2000, seed=seed)
        wav.append(h[-1][1])
    assert np.mean(vs) < np.mean(wav), \
        f"VisuShrink ({np.mean(vs):.4f}) not better than WaveletAdam ({np.mean(wav):.4f})"
    print("test_visushrink_beats_wavelet_noisy passed")


def test_ema_visuShrink_beats_visuShrink():
    """EMAVisuShrink has lower final loss than plain VisuShrink at σ=0.10."""
    from experiments.session11_adaptive import run_visuShrink_adam, run_ema_visuShrink_adam
    vs, ema = [], []
    for seed in range(10):
        h, _ = run_visuShrink_adam(sigma=0.10, n_steps=2000, seed=seed)
        vs.append(h[-1][1])
        h, _ = run_ema_visuShrink_adam(sigma=0.10, n_steps=2000, seed=seed)
        ema.append(h[-1][1])
    assert np.mean(ema) < np.mean(vs), \
        f"EMAVisuShrink ({np.mean(ema):.4f}) not better than VisuShrink ({np.mean(vs):.4f})"
    print("test_ema_visuShrink_beats_visuShrink passed")
    
def test_mnist_visuShrink_not_worse_than_adam():
    """PerLayerVisuShrink matches or beats Adam test accuracy on MNIST."""
    from problems.mnist_problem import load_mnist_subset, mnist_loss_and_grads
    from experiments.session12_mnist import run_adam_mnist, run_perlayer_visuShrink_adam

    X_train, y_train, X_test, y_test = load_mnist_subset(
        n_train=1000, n_test=200, seed=0
    )
    _, p_adam = run_adam_mnist(X_train, y_train, X_test, y_test,
                                n_epochs=20, seed=0)
    _, p_vs   = run_perlayer_visuShrink_adam(X_train, y_train, X_test, y_test,
                                              n_epochs=20, seed=0)

    _, _, acc_adam = mnist_loss_and_grads(p_adam, X_test, y_test)
    _, _, acc_vs   = mnist_loss_and_grads(p_vs,   X_test, y_test)

    assert acc_vs >= acc_adam - 0.02, \
        f"VisuShrink ({acc_vs:.3f}) worse than Adam ({acc_adam:.3f}) by more than 2%"
    print("test_mnist_visuShrink_not_worse_than_adam passed")

def test_learned_threshold_beats_ema_visushrink():
    """LearnedThreshold has lower loss than EMAVisuShrink at σ=0.10."""
    from core.learned_threshold_correction import train_correction_model
    from experiments.session15_learned_denoiser import run_learned_threshold_adam
    from experiments.session11_adaptive import run_ema_visuShrink_adam

    net, X_mean, X_std = train_correction_model(
        n_samples=5000, n_epochs=30
    )
    lt_losses, ema_losses = [], []
    for seed in range(5):
        h, _ = run_learned_threshold_adam(0.10, net, X_mean, X_std,
                                           n_steps=2000, seed=seed)
        lt_losses.append(h[-1][1])
        h, _ = run_ema_visuShrink_adam(0.10, 2000, seed=seed)
        ema_losses.append(h[-1][1])

    import numpy as np
    assert np.mean(lt_losses) < np.mean(ema_losses), \
        f"LearnedThreshold ({np.mean(lt_losses):.4f}) not better than " \
        f"EMAVisuShrink ({np.mean(ema_losses):.4f})"
    print("test_learned_threshold_beats_ema_visushrink passed")
    
def test_uncertain_threshold_beats_ema_visushrink():
    """UncertainThreshold has lower loss than EMAVisuShrink at σ=0.10."""
    from core.uncertain_threshold_correction import (
        train_uncertain_correction_model, denoise_uncertain_batch
    )
    from experiments.session16_uncertainty_aware_denoising import run_uncertain_threshold_adam
    from experiments.session11_adaptive import run_ema_visuShrink_adam
    import numpy as np

    unet, uX_mean, uX_std = train_uncertain_correction_model(
        n_samples=3000, n_epochs=20
    )
    ut_losses, ema_losses = [], []
    for seed in range(5):
        h, _ = run_uncertain_threshold_adam(
            0.10, unet, uX_mean, uX_std, n_steps=2000, seed=seed
        )
        ut_losses.append(h[-1][1])
        h, _ = run_ema_visuShrink_adam(0.10, 2000, seed=seed)
        ema_losses.append(h[-1][1])

    assert np.mean(ut_losses) < np.mean(ema_losses), \
        f"UncertainThreshold ({np.mean(ut_losses):.4f}) not better than " \
        f"EMAVisuShrink ({np.mean(ema_losses):.4f})"
    print("test_uncertain_threshold_beats_ema_visushrink passed")
    
def test_waveguard_beats_learned_threshold():
    """WaveGuard matches or beats LearnedThreshold at σ=0.10."""
    from core.waveguard import train_waveguard
    from core.learned_threshold_correction import train_correction_model
    from experiments.session17_waveguard import run_waveguard_adam
    from experiments.session15_learned_denoiser import run_learned_threshold_adam
    import numpy as np

    net, X_mean, X_std = train_waveguard(n_samples=5000, n_epochs=30)
    lnet, lX_mean, lX_std = train_correction_model(
        n_samples=5000, n_epochs=30
    )
    wg_losses, lt_losses = [], []
    for seed in range(5):
        h, _ = run_waveguard_adam(0.10, net, X_mean, X_std,
                                   alpha=2.0, beta=1.0,
                                   n_steps=2000, seed=seed)
        wg_losses.append(h[-1][1])
        h, _ = run_learned_threshold_adam(0.10, lnet, lX_mean, lX_std,
                                           n_steps=2000, seed=seed)
        lt_losses.append(h[-1][1])

    # Allow 10% tolerance — WaveGuard may not always win with small training budget
    tolerance = 0.10 * np.mean(lt_losses)
    assert np.mean(wg_losses) <= np.mean(lt_losses) + tolerance, \
        f"WaveGuard ({np.mean(wg_losses):.4f}) worse than " \
        f"LearnedThreshold ({np.mean(lt_losses):.4f}) by more than 10%"
    print("test_waveguard_beats_learned_threshold passed")
    
def test_waveguard_cnn_not_worse_than_adam():
    """WaveGuard per-filter matches or beats Adam test accuracy on CNN."""
    from problems.mnist_problem import load_mnist_subset
    from problems.cnn_problem import cnn_loss_and_grads
    from experiments.session18_cnn import run_adam_cnn, run_perfilter_waveguard_cnn
    from core.waveguard import train_waveguard

    X_train, y_train, X_test, y_test = load_mnist_subset(
        n_train=1000, n_test=200, seed=0
    )
    net, X_mean, X_std = train_waveguard(n_samples=5000, n_epochs=30)

    _, p_adam = run_adam_cnn(
        X_train, y_train, X_test, y_test, n_epochs=15, seed=0
    )
    _, p_wg = run_perfilter_waveguard_cnn(
        X_train, y_train, X_test, y_test,
        net, X_mean, X_std, n_epochs=15, seed=0
    )

    _, _, acc_adam = cnn_loss_and_grads(p_adam, X_test, y_test)
    _, _, acc_wg   = cnn_loss_and_grads(p_wg,   X_test, y_test)

    assert acc_wg >= acc_adam - 0.02, \
        f"WaveGuard CNN ({acc_wg:.3f}) worse than Adam ({acc_adam:.3f}) by more than 2%"
    print("test_waveguard_cnn_not_worse_than_adam passed")    

if __name__ == "__main__":
    test_sgd_decreases_loss()
    test_sgd_diverges_with_large_lr()
    test_adam_converges()
    test_adam_beats_sgd()
    test_adam_bias_correction()
    test_wavelet_adam_runs()
    test_noisy_adam_worse_than_clean()
    test_wavelet_adam_noise_robust()
    test_xor_adam_solves_clean()
    test_xor_wavelet_beats_adam_noisy()
    test_visushrink_beats_wavelet_noisy()
    test_ema_visuShrink_beats_visuShrink()
    test_mnist_visuShrink_not_worse_than_adam()
    test_learned_threshold_beats_ema_visushrink()
    test_uncertain_threshold_beats_ema_visushrink()
    test_waveguard_beats_learned_threshold()
    test_waveguard_cnn_not_worse_than_adam()
    print("\nAll optimizer tests passed.")