import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problems.mnist_problem import (
    load_mnist_subset, init_params, mnist_loss_and_grads, get_batches
)
from experiments.session11_adaptive import get_haar_matrix, denoise_visuShrink_batch

# =============================================================
# SESSION 13 — ADAPTIVE SNR DENOISING
#
# VisuShrink uses a fixed threshold formula:
#   T = σ_hat * sqrt(2 * log(N))
#
# Problem: early in training signal is strong (high SNR),
# universal threshold is too aggressive — removes real signal.
# Late in training signal is weak (low SNR), threshold is right.
#
# Fix: scale threshold by SNR estimate computed from coefficient
# energy ratio. High SNR → scale down threshold. Low SNR → keep.
#
# SNR_hat = energy(coarse coefficients) / energy(fine coefficients)
# threshold_scale = 1 / (1 + SNR_hat)
#
# This gives a threshold that:
#   - Approaches 0 when signal dominates (high SNR, early training)
#   - Approaches full VisuShrink when noise dominates (low SNR, late)
# =============================================================

BUFFER_SIZE  = 8
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
N_EPOCHS     = 50
BATCH_SIZE   = 64
LR           = 0.001

decay = 0.85
EMA_WEIGHTS = np.array([decay**(BUFFER_SIZE-1-i) for i in range(BUFFER_SIZE)])
EMA_WEIGHTS /= EMA_WEIGHTS.sum()


def estimate_snr_batch(coeffs):
    """
    Estimate SNR per parameter from Haar coefficients.

    Args:
        coeffs: (N, D) — Haar coefficients, each column = one parameter

    Returns:
        snr: (D,) — SNR estimate per parameter
    """
    N, D = coeffs.shape
    # Coarse: first N//2 coefficients (low frequency)
    # Fine:   last N//2 coefficients (high frequency)
    coarse_energy = np.sum(coeffs[:N//2]**2, axis=0) + 1e-10  # (D,)
    fine_energy   = np.sum(coeffs[N//2:]**2, axis=0) + 1e-10  # (D,)
    return coarse_energy / fine_energy                          # (D,)


def denoise_adaptive_snr_batch(buf, ema_weights):
    """
    Adaptive SNR denoising — vectorised across all parameters.

    Same as VisuShrink but threshold scaled by SNR estimate:
        threshold_scale = 1 / (1 + SNR_hat)

    High SNR (early training): scale → 0, minimal denoising
    Low SNR (late training):   scale → 1, full VisuShrink

    Args:
        buf:         (N, D) — gradient buffer
        ema_weights: (N,)   — EMA weights

    Returns:
        clean: (D,) — denoised gradient
    """
    N, D = buf.shape
    H = get_haar_matrix(N)

    # Haar transform all parameters simultaneously
    coeffs = H.T @ buf                                     # (N, D)

    # Estimate noise σ from finest-scale coefficients
    finest    = coeffs[N//2:]                              # (N//2, D)
    sigma_hat = np.median(np.abs(finest), axis=0) / 0.6745  # (D,)

    # Universal threshold
    base_threshold = sigma_hat * np.sqrt(2 * np.log(max(N, 2)))  # (D,)

    # SNR estimate — scale threshold down when signal dominates
    snr   = estimate_snr_batch(coeffs)                    # (D,)
    scale = 1.0 / (1.0 + snr)                             # (D,) in (0, 1)

    # Adaptive threshold — low when SNR high, full when SNR low
    threshold = base_threshold * scale                     # (D,)

    # Soft-threshold detail coefficients
    denoised_coeffs = coeffs.copy()
    detail = coeffs[1:]                                    # (N-1, D)
    denoised_coeffs[1:] = np.sign(detail) * np.maximum(
        np.abs(detail) - threshold[np.newaxis, :], 0
    )

    # Inverse transform + EMA aggregation
    denoised_buf = H @ denoised_coeffs                    # (N, D)
    return ema_weights @ denoised_buf                      # (D,)

def denoise_learned_correction_batch(buf, ema_weights,
                                      net=None, X_mean=None, X_std=None):
    """
    Vectorised denoising with learned threshold correction.
    Falls back to AdaptiveSNR if no model provided.
    """
    if net is None:
        return denoise_adaptive_snr_batch(buf, ema_weights)

    from core.learned_threshold_correction import (
        compute_mad_threshold, soft_threshold, apply_learned_threshold
    )
    from core.wavelets import haar_forward, haar_inverse

    N, D = buf.shape
    H    = get_haar_matrix(N)

    # Haar transform all parameters
    coeffs = H.T @ buf                           # (N, D)

    # Per-parameter MAD threshold
    finest    = coeffs[N//2:]                    # (N//2, D)
    sigma_mad = np.median(np.abs(finest), axis=0) / 0.6745  # (D,)
    T_mad     = sigma_mad * np.sqrt(2 * np.log(max(N, 2)))   # (D,)

    # Learned correction — apply per-parameter (vectorised over D)
    from core.learned_noise_model import extract_coefficient_stats
    corrections = np.ones(D)
    for j in range(D):
        feats      = extract_coefficient_stats(coeffs[:, j], N)
        feats_norm = (feats - X_mean) / (X_std + 1e-8)
        log_corr   = net.forward(feats_norm)
        corrections[j] = np.exp(np.clip(log_corr, -3.0, 3.0))

    T_final = T_mad * corrections                # (D,)

    # Soft-threshold with corrected threshold
    denoised_coeffs = coeffs.copy()
    detail = coeffs[1:]
    denoised_coeffs[1:] = np.sign(detail) * np.maximum(
        np.abs(detail) - T_final[np.newaxis, :], 0
    )

    denoised_buf = H @ denoised_coeffs
    return ema_weights @ denoised_buf

def run_adaptive_snr_adam(X_train, y_train, X_test, y_test,
                           n_epochs=N_EPOCHS, lr=LR,
                           batch_size=BATCH_SIZE, seed=0):
    """
    Adam with adaptive SNR-scaled VisuShrink denoising.
    Per-layer buffers, vectorised, no manual tuning.
    """
    rng = np.random.default_rng(seed)
    params = init_params(seed=seed)
    history = []

    param_shapes = {k: v.shape for k, v in params.items()}
    param_sizes  = {k: v.size  for k, v in params.items()}
    grad_history = {k: np.zeros((BUFFER_SIZE, param_sizes[k])) for k in params}
    buf_count = 0

    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    t = 0

    for epoch in range(n_epochs):
        for X_b, y_b in get_batches(X_train, y_train, batch_size, rng):
            t += 1
            loss, grads, acc = mnist_loss_and_grads(params, X_b, y_b)

            idx = (t - 1) % BUFFER_SIZE
            for k in params:
                grad_history[k][idx] = grads[k].flatten()
            buf_count += 1
            buffer_full = buf_count >= BUFFER_SIZE

            clean_grads = {}
            for k in params:
                if buffer_full:
                    order = [(idx + 1 + i) % BUFFER_SIZE
                             for i in range(BUFFER_SIZE)]
                    buf = grad_history[k][order]
                    clean_grads[k] = denoise_adaptive_snr_batch(
                        buf, EMA_WEIGHTS
                    ).reshape(param_shapes[k])
                else:
                    clean_grads[k] = grads[k]

            for k in params:
                g = clean_grads[k]
                m[k] = BETA1*m[k] + (1-BETA1)*g
                v[k] = BETA2*v[k] + (1-BETA2)*g**2
                mh = m[k] / (1-BETA1**t)
                vh = v[k] / (1-BETA2**t)
                params[k] -= lr * mh / (np.sqrt(vh) + EPS)

        if (epoch + 1) % 10 == 0:
            _, _, test_acc   = mnist_loss_and_grads(params, X_test, y_test)
            tl, _, train_acc = mnist_loss_and_grads(
                params, X_train[:500], y_train[:500])
            history.append((epoch+1, tl, train_acc, test_acc))
            print(f"  epoch {epoch+1:3d}  loss={tl:.4f}  "
                  f"train={train_acc:.3f}  test={test_acc:.3f}")

    return history, params


def benchmark():
    print("Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist_subset(
        n_train=5000, n_test=1000, seed=0
    )
    print(f"Train: {X_train.shape}  Test: {X_test.shape}\n")

    from experiments.session12_mnist import (
        run_adam_mnist, run_perlayer_visuShrink_adam
    )

    print("--- Adam ---")
    h_adam, _ = run_adam_mnist(X_train, y_train, X_test, y_test)

    print("\n--- PerLayerVisuShrink ---")
    h_vs, _ = run_perlayer_visuShrink_adam(X_train, y_train, X_test, y_test)

    print("\n--- AdaptiveSNR ---")
    h_snr, _ = run_adaptive_snr_adam(X_train, y_train, X_test, y_test)

    print("\n--- Final Comparison ---")
    print(f"{'optimizer':>25} {'train_loss':>12} {'train_acc':>10} {'test_acc':>10}")
    print("-" * 60)
    for name, h in [
        ("Adam",             h_adam),
        ("PerLayerVisuShrink", h_vs),
        ("AdaptiveSNR",      h_snr),
    ]:
        if h:
            e, tl, ta, tea = h[-1]
            print(f"{name:>25} {tl:>12.4f} {ta:>10.3f} {tea:>10.3f}")


benchmark()