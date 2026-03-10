import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problems.xor_problem import xor_loss_and_grads, xor_loss_and_grads_noisy, init_params, accuracy
from core.buffers import CircularGradientBuffer
from core.wavelets import haar_forward, haar_inverse, make_scale_weight_vector
from core.optimizers import AdamOptimizer
from viz.plot_utils import plot_convergence, plot_noise_comparison

# =============================================================
# SESSION 9: XOR NEURAL NETWORK
#
# First multi-parameter test of wavelet denoising.
# Network: 2 → 4 → 1, binary cross-entropy loss
# Parameters: W1(4x2), b1(4), W2(1x4), b2(1) = 13 total
#
# Key difference from Rosenbrock:
#   - Multiple parameter groups with different gradient scales
#   - Non-convex loss surface with saddle points
#   - Gradient noise from saddle point oscillations
# =============================================================

BUFFER_CAPACITY = 8
N_STEPS = 2000
N_TRIALS = 10


# =============================================================
# ADAM FOR XOR
# =============================================================

def run_xor_adam(n_steps=N_STEPS, lr=0.01, seed=0,
                 beta1=0.9, beta2=0.999, eps=1e-8):
    """Standard Adam on XOR network."""
    params = init_params(seed=seed)
    history = []

    # One moment estimate per parameter array
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}

    for t in range(1, n_steps + 1):
        loss, grads = xor_loss_and_grads(params)
        history.append((t, loss, accuracy(params)))

        for k in params:
            m[k] = beta1*m[k] + (1-beta1)*grads[k]
            v[k] = beta2*v[k] + (1-beta2)*grads[k]**2
            mh = m[k] / (1-beta1**t)
            vh = v[k] / (1-beta2**t)
            params[k] -= lr * mh / (np.sqrt(vh) + eps)

    return history, params


# =============================================================
# WAVELET ADAM FOR XOR
#
# Key design decision: one buffer per parameter GROUP.
# Each scalar in each parameter array gets its own slot
# in a flattened buffer. We denoise per-parameter.
# =============================================================

def run_xor_wavelet_adam(n_steps=N_STEPS, lr=0.01, seed=0,
                          buffer_capacity=BUFFER_CAPACITY,
                          raw_params_scale=None,
                          beta1=0.9, beta2=0.999, eps=1e-8):
    """
    WaveletAdam on XOR network.

    For each parameter group (W1, b1, W2, b2):
      - Maintain a circular buffer of the last gradient values
      - Denoise using wavelet transform before Adam update
      - Use shared scale weights across all parameter groups
    """
    import math
    J = int(math.log2(buffer_capacity))

    if raw_params_scale is None:
        raw_params_scale = np.zeros(J)  # neutral start

    params = init_params(seed=seed)
    history = []

    # One buffer per scalar parameter — track each weight separately
    # For simplicity: buffer tracks the most recent gradients of each param
    # We use a dict of buffers keyed by parameter name
    # Each buffer stores the last `capacity` gradient VALUES for that param array
    # We flatten each param array and denoise element-wise

    param_shapes = {k: v.shape for k, v in params.items()}
    param_sizes  = {k: v.size  for k, v in params.items()}

    # Circular buffers — one per parameter element
    # Store as 2D array: (capacity, param_size)
    grad_history = {k: np.zeros((buffer_capacity, param_sizes[k]))
                    for k in params}
    buf_count = 0

    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}

    weight_vec = make_scale_weight_vector(raw_params_scale, buffer_capacity)

    for t in range(1, n_steps + 1):
        loss, grads = xor_loss_and_grads(params)
        history.append((t, loss, accuracy(params)))

        # Write gradients to history
        idx = (t - 1) % buffer_capacity
        for k in params:
            grad_history[k][idx] = grads[k].flatten()

        buf_count += 1
        buffer_full = buf_count >= buffer_capacity

        # Denoise each parameter's gradient using wavelet transform
        clean_grads = {}
        for k in params:
            if buffer_full:
                # Read in chronological order
                order = [(idx + 1 + i) % buffer_capacity
                         for i in range(buffer_capacity)]
                buf = grad_history[k][order]  # (capacity, param_size)

                # Denoise each parameter element independently
                clean = np.zeros(param_sizes[k])
                for j in range(param_sizes[k]):
                    sig = buf[:, j]
                    coeffs = haar_forward(sig)
                    denoised_coeffs = coeffs * weight_vec
                    clean[j] = haar_inverse(denoised_coeffs)[-1]

                clean_grads[k] = clean.reshape(param_shapes[k])
            else:
                clean_grads[k] = grads[k]

        # Adam update with denoised gradients
        for k in params:
            g = clean_grads[k]
            m[k] = beta1*m[k] + (1-beta1)*g
            v[k] = beta2*v[k] + (1-beta2)*g**2
            mh = m[k] / (1-beta1**t)
            vh = v[k] / (1-beta2**t)
            params[k] -= lr * mh / (np.sqrt(vh) + eps)

    return history, params

def run_xor_noisy_adam(sigma, n_steps=N_STEPS, lr=0.01, seed=0,
                        beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam on XOR with noisy gradients."""
    rng = np.random.default_rng(seed)
    params = init_params(seed=seed)
    history = []
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}

    for t in range(1, n_steps + 1):
        loss, grads = xor_loss_and_grads_noisy(params, sigma=sigma, rng=rng)
        history.append((t, loss, accuracy(params)))

        for k in params:
            m[k] = beta1*m[k] + (1-beta1)*grads[k]
            v[k] = beta2*v[k] + (1-beta2)*grads[k]**2
            mh = m[k] / (1-beta1**t)
            vh = v[k] / (1-beta2**t)
            params[k] -= lr * mh / (np.sqrt(vh) + eps)

    return history, params


def run_xor_noisy_wavelet_adam(sigma, n_steps=N_STEPS, lr=0.01, seed=0,
                                buffer_capacity=BUFFER_CAPACITY,
                                raw_params_scale=None,
                                beta1=0.9, beta2=0.999, eps=1e-8):
    """WaveletAdam on XOR with noisy gradients."""
    import math
    J = int(math.log2(buffer_capacity))
    if raw_params_scale is None:
        raw_params_scale = np.array([1.0, 0.0, -1.0])

    rng = np.random.default_rng(seed)
    params = init_params(seed=seed)
    history = []

    param_shapes = {k: v.shape for k, v in params.items()}
    param_sizes  = {k: v.size  for k, v in params.items()}
    grad_history = {k: np.zeros((buffer_capacity, param_sizes[k]))
                    for k in params}
    buf_count = 0

    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    weight_vec = make_scale_weight_vector(raw_params_scale, buffer_capacity)

    for t in range(1, n_steps + 1):
        loss, grads = xor_loss_and_grads_noisy(params, sigma=sigma, rng=rng)
        history.append((t, loss, accuracy(params)))

        idx = (t - 1) % buffer_capacity
        for k in params:
            grad_history[k][idx] = grads[k].flatten()
        buf_count += 1
        buffer_full = buf_count >= buffer_capacity

        clean_grads = {}
        for k in params:
            if buffer_full:
                order = [(idx + 1 + i) % buffer_capacity
                         for i in range(buffer_capacity)]
                buf = grad_history[k][order]
                clean = np.zeros(param_sizes[k])
                for j in range(param_sizes[k]):
                    sig = buf[:, j]
                    coeffs = haar_forward(sig)
                    clean[j] = haar_inverse(coeffs * weight_vec)[-1]
                clean_grads[k] = clean.reshape(param_shapes[k])
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



# =============================================================
# BENCHMARK
# =============================================================

def benchmark(n_trials=N_TRIALS, n_steps=N_STEPS):
    print(f"{'optimizer':>20} {'final_loss':>12} {'final_acc':>10} {'solved':>8}")
    print("-" * 55)

    results = {}
    all_histories = {}

    configs = [
        ("Adam",               lambda s: run_xor_adam(n_steps, seed=s)),
        ("WaveletAdam-neutral", lambda s: run_xor_wavelet_adam(
            n_steps, seed=s, raw_params_scale=np.zeros(3))),
        ("WaveletAdam-denoise", lambda s: run_xor_wavelet_adam(
            n_steps, seed=s, raw_params_scale=np.array([1.0, 0.0, -1.0]))),
    ]

    for name, fn in configs:
        losses, accs, solved = [], [], []
        histories = []
        for seed in range(n_trials):
            h, p = fn(seed)
            final_loss = h[-1][1]
            final_acc  = h[-1][2]
            losses.append(final_loss)
            accs.append(final_acc)
            solved.append(final_acc == 1.0)
            histories.append([(t, l) for t, l, a in h])

        results[name] = {
            'loss': np.mean(losses),
            'acc':  np.mean(accs),
            'solved': sum(solved),
        }
        all_histories[name] = histories

        print(f"{name:>20} {np.mean(losses):>12.6f} "
              f"{np.mean(accs):>10.3f} "
              f"{sum(solved)}/{n_trials} trials")

    # Plot convergence for one representative seed
    rep_histories = []
    rep_labels = []
    for name, fn in configs:
        h, _ = fn(0)
        rep_histories.append([(x, 0, l) for x, l, a in h])
        rep_labels.append(name)

    plot_convergence(rep_histories, rep_labels,
                     title="XOR Network — Convergence Comparison",
                     filename="xor_convergence.png")

    return results


benchmark()

def noisy_benchmark(n_trials=N_TRIALS, n_steps=N_STEPS):
    sigma_levels = [0.0, 0.05, 0.1, 0.2, 0.5]

    print(f"\n{'sigma':>8} {'optimizer':>25} {'final_loss':>12} {'solved':>8}")
    print("-" * 58)

    adam_losses    = []
    wavelet_losses = []

    for sigma in sigma_levels:
        al, wl = [], []
        for name, fn in [
            ("Adam",        lambda s, sg=sigma: run_xor_noisy_adam(sg, n_steps, seed=s)),
            ("WaveletAdam", lambda s, sg=sigma: run_xor_noisy_wavelet_adam(
                                sg, n_steps, seed=s)),
        ]:
            losses, solved = [], []
            for seed in range(n_trials):
                h, p = fn(seed)
                losses.append(h[-1][1])
                solved.append(accuracy(p) == 1.0)
            print(f"{sigma:>8.2f} {name:>25} "
                  f"{np.mean(losses):>12.6f} "
                  f"{sum(solved)}/{n_trials}")
            if name == "Adam":
                al = losses
            else:
                wl = losses
        adam_losses.append(np.mean(al))
        wavelet_losses.append(np.mean(wl))
        print()

    # Noise comparison bar chart
    plot_noise_comparison(
        sigma_levels, adam_losses, wavelet_losses, wavelet_losses,
        filename="xor_noise_comparison.png"
    )

    # Convergence curves at σ=0.05
    h_adam, _    = run_xor_noisy_adam(0.05, n_steps=N_STEPS, seed=0)
    h_wavelet, _ = run_xor_noisy_wavelet_adam(0.05, n_steps=N_STEPS, seed=0)

    plot_convergence(
        [[(t, 0, l) for t, l, a in h_adam],
         [(t, 0, l) for t, l, a in h_wavelet]],
        ['Adam', 'WaveletAdam'],
        title="XOR Network σ=0.05 — Adam vs WaveletAdam",
        filename="xor_sigma005.png"
    )
noisy_benchmark()