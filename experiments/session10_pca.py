import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problems.xor_problem import xor_loss_and_grads, xor_loss_and_grads_noisy, \
                                   init_params, accuracy
from core.pca_denoiser import PCAGradientDenoiser
from viz.plot_utils import plot_convergence, plot_noise_comparison

# =============================================================
# SESSION 10: PCA JOINT GRADIENT DENOISING
#
# Replace per-element wavelet denoising with joint PCA denoising.
#
# Key difference:
#   Per-element: denoise each of 13 weights independently
#   PCA joint:   denoise the full 13-dim gradient vector together
#
# The PCA approach captures correlations between parameters —
# when W1 and W2 gradients oscillate together, PCA identifies
# that shared direction and can suppress it jointly.
# =============================================================

BUFFER_SIZE  = 8
N_STEPS      = 2000
N_TRIALS     = 10
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
LR           = 0.01


def flatten_grads(grads):
    """Flatten all gradient arrays into a single vector."""
    return np.concatenate([grads[k].flatten()
                           for k in sorted(grads.keys())])


def unflatten_grads(flat, grads_template):
    """Restore flat gradient vector to dict matching grads_template."""
    result = {}
    idx = 0
    for k in sorted(grads_template.keys()):
        size = grads_template[k].size
        result[k] = flat[idx:idx+size].reshape(grads_template[k].shape)
        idx += size
    return result


def run_pca_wavelet_adam(sigma, n_steps=N_STEPS, lr=LR, seed=0,
                          n_components=4, scale_weights=None,
                          beta1=BETA1, beta2=BETA2, eps=EPS):
    """
    Adam with PCA joint gradient denoising.

    Denoises the full 13-dim gradient vector jointly using
    Gram matrix PCA — captures inter-parameter correlations.
    """
    rng = np.random.default_rng(seed)
    params = init_params(seed=seed)
    history = []

    # Default scale weights — suppress last components (noisiest)
    if scale_weights is None:
        scale_weights = np.array([1.0, 0.8, 0.5, 0.2])[:n_components]

    denoiser = PCAGradientDenoiser(
        n_components=n_components,
        buffer_size=BUFFER_SIZE,
        scale_weights=scale_weights
    )

    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}

    for t in range(1, n_steps + 1):
        loss, grads = xor_loss_and_grads_noisy(params, sigma=sigma, rng=rng)
        history.append((t, loss, accuracy(params)))

        # Flatten all gradients into one vector
        g_flat = flatten_grads(grads)

        # Update PCA buffer
        denoiser.update(g_flat)

        # Denoise jointly
        g_clean_flat = denoiser.denoise(g_flat)

        # Restore dict structure
        clean_grads = unflatten_grads(g_clean_flat, grads)

        # Adam update
        for k in params:
            g = clean_grads[k]
            m[k] = beta1*m[k] + (1-beta1)*g
            v[k] = beta2*v[k] + (1-beta2)*g**2
            mh = m[k] / (1-beta1**t)
            vh = v[k] / (1-beta2**t)
            params[k] -= lr * mh / (np.sqrt(vh) + eps)

    return history, params

def run_pca_temporal_adam(sigma, n_steps=N_STEPS, lr=LR, seed=0,
                           n_components=4, haar_weight=0.7,
                           beta1=BETA1, beta2=BETA2, eps=EPS):
    """PCA + temporal wavelet denoising combined."""
    rng = np.random.default_rng(seed)
    params = init_params(seed=seed)
    history = []

    denoiser = PCAGradientDenoiser(
        n_components=n_components,
        buffer_size=BUFFER_SIZE,
        scale_weights=np.array([1.0, 0.5, 0.1, 0.0])
    )

    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}

    for t in range(1, n_steps + 1):
        loss, grads = xor_loss_and_grads_noisy(params, sigma=sigma, rng=rng)
        history.append((t, loss, accuracy(params)))

        g_flat = flatten_grads(grads)
        denoiser.update(g_flat)
        g_clean_flat = denoiser.denoise_with_temporal(g_flat,
                                                       haar_weight=haar_weight)
        clean_grads = unflatten_grads(g_clean_flat, grads)

        for k in params:
            g = clean_grads[k]
            m[k] = beta1*m[k] + (1-beta1)*g
            v[k] = beta2*v[k] + (1-beta2)*g**2
            mh = m[k] / (1-beta1**t)
            vh = v[k] / (1-beta2**t)
            params[k] -= lr * mh / (np.sqrt(vh) + eps)

    return history, params


import numpy as np

# =============================================================
# PCA JOINT GRADIENT DENOISER

def benchmark(n_trials=N_TRIALS, n_steps=N_STEPS):
    from experiments.session9_neural import (
        run_xor_noisy_adam,
        run_xor_noisy_wavelet_adam,
    )

    sigma_levels = [0.0, 0.05, 0.1, 0.2, 0.5]

    print(f"\n{'sigma':>8} {'optimizer':>25} {'final_loss':>12} {'solved':>8}")
    print("-" * 60)

    adam_losses    = []
    wavelet_losses = []
    pca_losses     = []

    for sigma in sigma_levels:
        al, wl, pl = [], [], []

        configs = [
            ("Adam",         lambda s, sg=sigma:
                             run_xor_noisy_adam(sg, n_steps, seed=s)),
            ("WaveletAdam",  lambda s, sg=sigma:
                             run_xor_noisy_wavelet_adam(sg, n_steps, seed=s)),
            ("PCAWavelet",   lambda s, sg=sigma:
                             run_pca_wavelet_adam(sg, n_steps, seed=s)),
            ("PCAWavelet-tuned", lambda s, sg=sigma:
                             run_pca_wavelet_adam(sg, n_steps, seed=s, scale_weights=np.array([1.0, 0.1, 0.0, 0.0]))),
            ("PCATemporal", lambda s, sg=sigma:
                             run_pca_temporal_adam(sg, n_steps, seed=s))
        ]

        for name, fn in configs:
            losses, solved = [], []
            for seed in range(n_trials):
                h, p = fn(seed)
                losses.append(h[-1][1])
                solved.append(accuracy(p) == 1.0)
            print(f"{sigma:>8.2f} {name:>25} "
                  f"{np.mean(losses):>12.6f} "
                  f"{sum(solved)}/{n_trials}")
            if name == "Adam":         al = losses
            elif name == "WaveletAdam": wl = losses
            else:                       pl = losses

        adam_losses.append(np.mean(al))
        wavelet_losses.append(np.mean(wl))
        pca_losses.append(np.mean(pl))
        print()

    # Visualization
    plot_noise_comparison(
        sigma_levels, adam_losses, wavelet_losses, pca_losses,
        filename="pca_noise_comparison.png"
    )

    # Convergence at σ=0.05
    h_adam, _    = run_xor_noisy_adam(0.05, n_steps, seed=0)
    h_wav,  _    = run_xor_noisy_wavelet_adam(0.05, n_steps, seed=0)
    h_pca,  _    = run_pca_wavelet_adam(0.05, n_steps, seed=0)

    plot_convergence(
        [[(t, 0, l) for t, l, a in h_adam],
         [(t, 0, l) for t, l, a in h_wav],
         [(t, 0, l) for t, l, a in h_pca]],
        ['Adam', 'WaveletAdam', 'PCAWavelet'],
        title="XOR σ=0.05 — Adam vs WaveletAdam vs PCAWavelet",
        filename="pca_convergence.png"
    )


benchmark()