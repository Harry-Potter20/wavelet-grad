import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problems.xor_problem import xor_loss_and_grads_noisy, init_params, accuracy
from core.learned_threshold_correction import (
    train_correction_model, apply_learned_threshold,
    compute_mad_threshold, soft_threshold
)
from core.wavelets import haar_forward, haar_inverse
from experiments.session11_adaptive import get_haar_matrix
from core.learned_noise_model import extract_coefficient_stats
from viz.plot_utils import plot_convergence, plot_noise_comparison, plot_gradient_signal, plot_scale_weight_evolution

# =============================================================
# SESSION 15 — LEARNED THRESHOLD CORRECTION ON XOR
#
# Replaces MAD threshold in AdaptiveSNR with learned correction.
# Tests whether the 4.5% reconstruction improvement translates
# to better optimizer performance.
# =============================================================

BUFFER_SIZE  = 8
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
N_STEPS      = 2000
N_TRIALS     = 10
LR           = 0.05
ALPHA       = 1.0

decay = 1.0
EMA_WEIGHTS = np.array([decay**(BUFFER_SIZE-1-i)
                         for i in range(BUFFER_SIZE)])
EMA_WEIGHTS /= EMA_WEIGHTS.sum()


def denoise_learned_batch(buf, ema_weights, net, X_mean, X_std):
    """
    Stable learned threshold denoiser with:
      - MAD baseline (robust)
      - Learned residual correction
      - Monotonicity constraint
      - Alpha blending
    """

    N, D = buf.shape
    H = get_haar_matrix(N)

    # ---------------------------------------------------------
    # 1. Haar transform
    # ---------------------------------------------------------
    coeffs = H.T @ buf   # (N, D)
    detail = coeffs[1:]  # (N-1, D)

    # ---------------------------------------------------------
    # 2. Robust MAD baseline (vectorised)
    # ---------------------------------------------------------
    median = np.median(np.abs(detail), axis=0)
    sigma_hat = median / 0.6745

    T_mad = sigma_hat * np.sqrt(2 * np.log(N))   # (D,)

    # ---------------------------------------------------------
    # 3. Feature extraction
    # ---------------------------------------------------------
    feats = np.stack(
        [extract_coefficient_stats(coeffs[:, j], N) for j in range(D)],
        axis=0
    )

    assert feats.shape[1] == X_mean.shape[0], \
        f"Feature mismatch: {feats.shape} vs {X_mean.shape}"

    feats_norm = (feats - X_mean) / X_std

    # ---------------------------------------------------------
    # 4. Vectorised NN forward
    # ---------------------------------------------------------
    z1 = feats_norm @ net.W1.T + net.b1
    h1 = np.maximum(0, z1)

    z2 = h1 @ net.W2.T + net.b2
    h2 = np.maximum(0, z2)

    z3 = h2 @ net.W3.T + net.b3
    log_corr = z3.squeeze()

    # ---------------------------------------------------------
    # 5. STABLE CORRECTION
    # ---------------------------------------------------------

    # Clamp extreme predictions
    log_corr = np.clip(log_corr, -1.0, 1.0)
    raw_correction = np.exp(log_corr)

    # --- NEW: noise-level awareness ---
    # If noise is low → trust MAD
    # If noise is high → allow correction

    noise_level = sigma_hat / (np.mean(np.abs(buf), axis=0) + 1e-8)

    # Smooth gating function (0 → low noise, 1 → high noise)
    gate = np.clip((noise_level - 0.1) / 0.5, 0.0, 1.0)

    # --- Controlled correction ---
    correction = 1 + ALPHA * gate * (raw_correction - 1)

    # Flexibility
    correction = np.clip(correction, 0.8, 2.0)

    # Final thresholds
    thresholds = T_mad * correction

    # ---------------------------------------------------------
    # 6. Soft threshold
    # ---------------------------------------------------------
    denoised_coeffs = coeffs.copy()

    denoised_coeffs[1:] = np.sign(detail) * np.maximum(
        np.abs(detail) - thresholds[None, :],
        0
    )

    # ---------------------------------------------------------
    # 7. Inverse + EMA + ALPHA BLENDING
    # ---------------------------------------------------------
    denoised_buf = H @ denoised_coeffs

    g_denoised = ema_weights @ denoised_buf
    g_raw      = ema_weights @ buf

    return ALPHA * g_denoised + (1 - ALPHA) * g_raw



def run_learned_threshold_adam(
    sigma, net, X_mean, X_std,
    n_steps=N_STEPS, lr=LR, seed=0,
    beta1=BETA1, beta2=BETA2, eps=EPS
):
    rng = np.random.default_rng(seed)
    params = init_params(seed=seed)
    history = []

    param_shapes = {k: v.shape for k, v in params.items()}
    param_sizes  = {k: v.size  for k, v in params.items()}

    grad_history = {
        k: np.zeros((BUFFER_SIZE, param_sizes[k]))
        for k in params
    }

    buf_count = 0

    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}

    for t in range(1, n_steps + 1):
        loss, grads = xor_loss_and_grads_noisy(params, sigma=sigma, rng=rng)
        history.append((t, loss, accuracy(params)))

        idx = (t - 1) % BUFFER_SIZE

        for k in params:
            grad_history[k][idx] = grads[k].flatten()

        buf_count += 1
        buffer_full = buf_count >= BUFFER_SIZE

        clean_grads = {}

        for k in params:
            if buffer_full:
                order = [(idx + 1 + i) % BUFFER_SIZE for i in range(BUFFER_SIZE)]
                buf = grad_history[k][order]

                clean = denoise_learned_batch(
                    buf, EMA_WEIGHTS, net, X_mean, X_std
                )

                clean_grads[k] = clean.reshape(param_shapes[k])
            else:
                clean_grads[k] = grads[k]

        # Adam update
        for k in params:
            g = clean_grads[k]

            m[k] = beta1*m[k] + (1-beta1)*g
            v[k] = beta2*v[k] + (1-beta2)*g**2

            mh = m[k] / (1-beta1**t)
            vh = v[k] / (1-beta2**t)

            params[k] -= lr * mh / (np.sqrt(vh) + eps)

    return history, params



def benchmark():
    from experiments.session9_neural import run_xor_noisy_adam
    from experiments.session11_adaptive import run_ema_visuShrink_adam

    print("Training correction model...")
    net, X_mean, X_std = train_correction_model(
        n_samples=10000,
        n_epochs=50
    )

    sigma_levels = [0.0, 0.05, 0.1, 0.2]

    print(f"\n{'sigma':>8} {'optimizer':>25} {'final_loss':>12} {'solved':>8}")
    print("-" * 60)

    for sigma in sigma_levels:
        configs = [
            ("Adam",
             lambda s, sg=sigma: run_xor_noisy_adam(sg, N_STEPS, seed=s)),

            ("EMAVisuShrink",
             lambda s, sg=sigma: run_ema_visuShrink_adam(sg, N_STEPS, seed=s)),

            ("LearnedThreshold+Stable",
             lambda s, sg=sigma: run_learned_threshold_adam(
                 sg, net, X_mean, X_std, seed=s)),
        ]

        for name, fn in configs:
            losses, solved = [], []

            for seed in range(N_TRIALS):
                h, p = fn(seed)
                losses.append(h[-1][1])
                solved.append(accuracy(p) == 1.0)

            print(f"{sigma:>8.2f} {name:>25} "
                  f"{np.mean(losses):>12.6f} "
                  f"{sum(solved)}/{N_TRIALS}")
        print()


if __name__ == "__main__":
    benchmark()