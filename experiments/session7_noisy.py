import numpy as np
import math
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problems.rosenbrock import rosenbrock
from problems.noisy_rosenbrock import noisy_rosenbrock
from core.buffers import CircularGradientBuffer
from core.wavelets import haar_forward, haar_inverse, make_scale_weight_vector
from core.optimizers import (
    run_adam,
    analytical_meta_gradient,
    compute_state_features,
    AdamOptimizer,
)
from core.meta_net import MetaNet

# =============================================================
# SESSION 7: NOISY GRADIENTS
#
# This is where wavelet denoising should show real advantage.
# Adam's 1st moment smooths at one fixed timescale (β₁=0.9).
# The wavelet transform smooths at multiple scales simultaneously.
#
# Hypothesis: under sufficient gradient noise, WaveletAdam
# converges faster than plain Adam.
# =============================================================

SIGMA_LEVELS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
N_STEPS = 3000
N_TRIALS = 5    # average over multiple random seeds for fairness


# --- Noisy Adam baseline ---

def run_noisy_adam(sigma, n_steps, seed=0, start=(-1.0, 1.0),
                   lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam with noisy gradient observations."""
    rng = np.random.default_rng(seed)
    x, y = start
    history = []
    mx, my = 0.0, 0.0
    vx, vy = 0.0, 0.0

    for t in range(1, n_steps + 1):
        loss, gx, gy = noisy_rosenbrock(x, y, sigma=sigma, rng=rng)
        history.append((x, y, loss))

        mx = beta1*mx + (1-beta1)*gx
        my = beta1*my + (1-beta1)*gy
        vx = beta2*vx + (1-beta2)*gx**2
        vy = beta2*vy + (1-beta2)*gy**2

        mx_hat = mx/(1-beta1**t); my_hat = my/(1-beta1**t)
        vx_hat = vx/(1-beta2**t); vy_hat = vy/(1-beta2**t)

        x = x - lr*mx_hat/(np.sqrt(vx_hat)+eps)
        y = y - lr*my_hat/(np.sqrt(vy_hat)+eps)

    return history


# --- Noisy WaveletAdam ---

def run_noisy_wavelet_adam(sigma, n_steps, seed=0,
                            buffer_capacity=8, start=(-1.0, 1.0),
                            lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
                            raw_params=None):
    rng = np.random.default_rng(seed)
    x, y = start
    history = []

    buf_x = CircularGradientBuffer(capacity=buffer_capacity, grad_shape=(1,))
    buf_y = CircularGradientBuffer(capacity=buffer_capacity, grad_shape=(1,))

    mx, my = 0.0, 0.0
    vx, vy = 0.0, 0.0

    # Default: suppress fine scales (highest noise), keep coarse
    if raw_params is None:
        J = int(math.log2(buffer_capacity))
        raw_params = np.zeros(J)  # coarse→keep, fine→suppress

    for t in range(1, n_steps + 1):
        loss, gx, gy = noisy_rosenbrock(x, y, sigma=sigma, rng=rng)
        history.append((x, y, loss))

        buf_x.write(np.array([gx]))
        buf_y.write(np.array([gy]))

        # Only update once buffer is full — use denoised gradient from step 1
        if buf_x.is_full():
            sig_x = buf_x.read().flatten()
            sig_y = buf_y.read().flatten()

            weight_vec = make_scale_weight_vector(raw_params, buffer_capacity)
            gx = haar_inverse(haar_forward(sig_x) * weight_vec)[-1]
            gy = haar_inverse(haar_forward(sig_y) * weight_vec)[-1]

        mx = beta1*mx + (1-beta1)*gx
        my = beta1*my + (1-beta1)*gy
        vx = beta2*vx + (1-beta2)*gx**2
        vy = beta2*vy + (1-beta2)*gy**2

        mx_hat = mx/(1-beta1**t); my_hat = my/(1-beta1**t)
        vx_hat = vx/(1-beta2**t); vy_hat = vy/(1-beta2**t)

        x = x - lr*mx_hat/(np.sqrt(vx_hat)+eps)
        y = y - lr*my_hat/(np.sqrt(vy_hat)+eps)

    return history


def noise_adaptive_params(sigma):
    """
    Choose raw_params based on noise level.
    Higher sigma → more suppression of fine scales.
    """
    if sigma == 0.0:
        return np.array([0.0, 0.0, 0.0])    # all 0.5 — barely denoise
    elif sigma <= 0.1:
        return np.array([0.5, 0.0, -0.5])   # mild suppression
    elif sigma <= 0.5:
        return np.array([1.0, 0.0, -1.0])   # moderate suppression
    elif sigma <= 1.0:
        return np.array([1.0, 0.5, -2.0])   # heavy suppression
    else:
        return np.array([2.0, 1.0, -3.0]) # extreme suppression


# --- Benchmark across noise levels ---

def benchmark(n_trials=N_TRIALS, n_steps=N_STEPS):
    print(f"{'sigma':>8} {'optimizer':>20} {'final_loss':>12} {'converged':>10}")
    print("-" * 55)

    for sigma in SIGMA_LEVELS:
        for name, use_wavelet in [("Adam", False), ("WaveletAdam", True)]:
            losses = []
            converged_steps = []

            for seed in range(n_trials):
                if not use_wavelet or sigma == 0.0:
                    h = run_noisy_adam(sigma, n_steps, seed)
                else:
                    h = run_noisy_wavelet_adam(
                        sigma, n_steps, seed,
                        buffer_capacity=8,
                        raw_params=noise_adaptive_params(sigma)
                    )

                final_loss = h[-1][2]
                losses.append(final_loss)
                conv = next((i for i,(x,y,l) in enumerate(h)
                             if abs(x-1)<1e-2 and abs(y-1)<1e-2), None)
                converged_steps.append(conv)

            mean_loss = np.mean(losses)
            n_converged = sum(c is not None for c in converged_steps)
            print(f"{sigma:>8.1f} {name:>20} "
                  f"{mean_loss:>12.6f} "
                  f"{n_converged}/{n_trials} trials")

        print()
benchmark()
