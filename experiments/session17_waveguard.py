import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problems.xor_problem import xor_loss_and_grads_noisy, init_params, accuracy
from core.waveguard import train_waveguard, denoise_waveguard_batch

# =============================================================
# SESSION 17 — WAVEGUARD BENCHMARK
#
# Tests WaveGuard across:
#   1. Default (α=1, β=1) — equal weight
#   2. SNR-strict (α=2, β=1) — only correct when noise clear
#   3. Model-trust (α=1, β=2) — only correct when confident
#   4. Ablations: α=1 β=0 (AdaptiveSNR), α=0 β=1 (confidence only)
# =============================================================

BUFFER_SIZE  = 8
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
N_STEPS      = 2000
N_TRIALS     = 10
LR           = 0.05

decay = 1.0
EMA_WEIGHTS = np.array([decay**(BUFFER_SIZE-1-i)
                         for i in range(BUFFER_SIZE)])
EMA_WEIGHTS /= EMA_WEIGHTS.sum()


def run_waveguard_adam(sigma, net, X_mean, X_std,
                        alpha=1.0, beta=1.0,
                        n_steps=N_STEPS, lr=LR, seed=0,
                        beta1=BETA1, beta2=BETA2, eps=EPS):
    """Adam with WaveGuard denoising."""
    rng = np.random.default_rng(seed)
    params = init_params(seed=seed)
    history = []

    param_shapes = {k: v.shape for k, v in params.items()}
    param_sizes  = {k: v.size  for k, v in params.items()}
    grad_history = {k: np.zeros((BUFFER_SIZE, param_sizes[k]))
                    for k in params}
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
                order = [(idx + 1 + i) % BUFFER_SIZE
                         for i in range(BUFFER_SIZE)]
                buf = grad_history[k][order]
                clean_grads[k] = denoise_waveguard_batch(
                    buf, EMA_WEIGHTS, net, X_mean, X_std,
                    alpha=alpha, beta=beta
                ).reshape(param_shapes[k])
            else:
                clean_grads[k] = grads[k]

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
    from experiments.session15_learned_denoiser import run_learned_threshold_adam
    from core.learned_threshold_correction import train_correction_model

    print("Training WaveGuard model...")
    net, X_mean, X_std = train_waveguard(n_samples=10000, n_epochs=50)

    print("Training LearnedThreshold model for comparison...")
    lnet, lX_mean, lX_std = train_correction_model(
        n_samples=10000, n_epochs=50
    )

    sigma_levels = [0.0, 0.05, 0.1, 0.2]
    print(f"\n{'sigma':>8} {'optimizer':>30} {'final_loss':>12} {'solved':>8}")
    print("-" * 65)

    for sigma in sigma_levels:
        configs = [
            ("Adam",
             lambda s, sg=sigma:
             run_xor_noisy_adam(sg, N_STEPS, seed=s)),

            ("EMAVisuShrink",
             lambda s, sg=sigma:
             run_ema_visuShrink_adam(sg, N_STEPS, seed=s)),

            ("LearnedThreshold",
             lambda s, sg=sigma:
             run_learned_threshold_adam(sg, lnet, lX_mean, lX_std, seed=s)),

            ("WaveGuard(α=1,β=1)",
             lambda s, sg=sigma:
             run_waveguard_adam(sg, net, X_mean, X_std,
                                alpha=1.0, beta=1.0, seed=s)),

            ("WaveGuard(α=2,β=1)",
             lambda s, sg=sigma:
             run_waveguard_adam(sg, net, X_mean, X_std,
                                alpha=2.0, beta=1.0, seed=s)),

            ("WaveGuard(α=1,β=2)",
             lambda s, sg=sigma:
             run_waveguard_adam(sg, net, X_mean, X_std,
                                alpha=1.0, beta=2.0, seed=s)),
        ]

        for name, fn in configs:
            losses, solved = [], []
            for seed in range(N_TRIALS):
                h, p = fn(seed)
                losses.append(h[-1][1])
                solved.append(accuracy(p) == 1.0)
            print(f"{sigma:>8.2f} {name:>30} "
                  f"{np.mean(losses):>12.6f} "
                  f"{sum(solved)}/{N_TRIALS}")
        print()


if __name__ == "__main__":
    benchmark()