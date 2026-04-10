import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problems.xor_problem import xor_loss_and_grads, xor_loss_and_grads_noisy, \
                                   init_params, accuracy
from core.wavelets import haar_forward, haar_inverse, make_scale_weight_vector
from core.buffers import CircularGradientBuffer
from viz.plot_utils import plot_convergence, plot_noise_comparison

# =============================================================
# SESSION 11 — IMPROVEMENT 1: ADAPTIVE DENOISING (VISUSHRINK)
#
# Instead of hand-tuned scale weights, estimate noise level
# directly from the finest-scale wavelet coefficients.
#
# σ_hat = median(|finest coefficients|) / 0.6745
# threshold = σ_hat * sqrt(2 * log(N))
#
# Then soft-threshold all coefficients by this threshold.
# No tuning required. Works differently per parameter,
# per step, adapting to local gradient statistics.
#
# Limitation: assumes finest coefficients are purely noise.
# Breaks when true gradient has high-frequency components.
# =============================================================

BUFFER_SIZE = 8
N_STEPS     = 2000
N_TRIALS    = 10
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
LR          = 0.01


def soft_threshold(coeffs, threshold):
    """
    Soft thresholding: shrink toward zero by threshold amount.
    Preserves sign, zeros small coefficients, attenuates large ones.

    Different from hard threshold (zero/keep) — smoother, preserves
    partial signal in ambiguous coefficients.
    """
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)


def estimate_sigma(finest_coeffs):
    """
    Estimate noise std from finest-scale Haar coefficients.

    Uses the MAD (median absolute deviation) estimator:
      σ_hat = median(|coeffs|) / 0.6745

    0.6745 = Φ⁻¹(0.75) — the 75th percentile of N(0,1).
    Makes median(|X|) an unbiased estimator of σ for Gaussian noise.

    Robust to outliers — median ignores large signal spikes.
    """
    return np.median(np.abs(finest_coeffs)) / 0.6745


def visushrink_threshold(sigma_hat, N):
    """
    Universal threshold (Donoho & Johnstone 1994):
      T = σ_hat * sqrt(2 * log(N))

    Provably removes all noise coefficients with high probability
    as N → ∞. Conservative — may over-smooth at small N.
    """
    return sigma_hat * np.sqrt(2 * np.log(max(N, 2)))


def denoise_visuShrink(signal):
    """
    Denoise a 1D signal using VisuShrink soft thresholding.

    Steps:
    1. Haar forward transform
    2. Estimate σ from finest-scale coefficients (last N//2 entries)
    3. Compute universal threshold
    4. Soft-threshold all detail coefficients (keep coarsest)
    5. Haar inverse transform
    """
    N = len(signal)
    coeffs = haar_forward(signal)

    # Finest-scale coefficients — last N//2 entries in Haar layout
    finest = coeffs[N//2:]
    sigma_hat = estimate_sigma(finest)
    threshold = visushrink_threshold(sigma_hat, N)

    # Soft-threshold detail coefficients — preserve coarsest (index 0)
    denoised_coeffs = coeffs.copy()
    denoised_coeffs[1:] = soft_threshold(coeffs[1:], threshold)

    return haar_inverse(denoised_coeffs)


def flatten_grads(grads):
    return np.concatenate([grads[k].flatten() for k in sorted(grads.keys())])


def unflatten_grads(flat, grads_template):
    result = {}
    idx = 0
    for k in sorted(grads_template.keys()):
        size = grads_template[k].size
        result[k] = flat[idx:idx+size].reshape(grads_template[k].shape)
        idx += size
    return result


def run_visuShrink_adam(sigma, n_steps=N_STEPS, lr=LR, seed=0,
                         beta1=BETA1, beta2=BETA2, eps=EPS):
    """
    Adam with per-parameter VisuShrink adaptive denoising.
    No tuning required — threshold estimated from data each step.
    """
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

                clean = np.zeros(param_sizes[k])
                for j in range(param_sizes[k]):
                    denoised = denoise_visuShrink(buf[:, j])
                    clean[j] = denoised[-1]
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

def run_ema_visuShrink_adam(sigma, n_steps=N_STEPS, lr=LR, seed=0,
                             decay=0.85, beta1=BETA1, beta2=BETA2, eps=EPS):
    """
    VisuShrink denoising + EMA aggregation.

    Correct order:
    1. Denoise the full buffer with VisuShrink
    2. Take EMA-weighted average of denoised values
       (instead of using only the last reconstructed value)

    EMA applied after denoising — doesn't distort the threshold estimate.
    """
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

    # EMA weights — oldest to newest, sum to 1
    ema_weights = np.array([decay**(BUFFER_SIZE-1-i)
                            for i in range(BUFFER_SIZE)])
    ema_weights /= ema_weights.sum()

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

                clean = np.zeros(param_sizes[k])
                for j in range(param_sizes[k]):
                    # Step 1: denoise raw buffer
                    denoised_signal = denoise_visuShrink(buf[:, j])
                    # Step 2: EMA-weighted average of denoised values
                    clean[j] = np.dot(ema_weights, denoised_signal)
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

def make_haar_matrix(N):
    """Build N×N Haar transform matrix — H @ signal = haar_forward(signal)."""
    assert (N & (N-1)) == 0, "N must be power of 2"
    identity = np.eye(N)
    return np.array([haar_forward(identity[i]) for i in range(N)]).T


_HAAR_MATRIX_CACHE = {}

def get_haar_matrix(N):
    if N not in _HAAR_MATRIX_CACHE:
        _HAAR_MATRIX_CACHE[N] = make_haar_matrix(N)
    return _HAAR_MATRIX_CACHE[N]


def denoise_visuShrink_batch(buf, ema_weights):
    """
    Vectorised VisuShrink denoising for all parameters at once.

    Args:
        buf:         (N, D) — N gradient snapshots, D parameters
        ema_weights: (N,)   — EMA weights summing to 1

    Returns:
        clean: (D,) — denoised gradient estimate per parameter
    """
    N, D = buf.shape
    H = get_haar_matrix(N)

    # Haar transform all D signals simultaneously
    coeffs = H.T @ buf                          # (N, D)

    # Estimate σ per parameter from finest-scale coefficients
    finest    = coeffs[N//2:]                   # (N//2, D)
    sigma_hat = np.median(np.abs(finest), axis=0) / 0.6745  # (D,)

    # Universal threshold per parameter
    threshold = sigma_hat * np.sqrt(2 * np.log(max(N, 2)))  # (D,)

    # Soft-threshold all detail coefficients
    denoised_coeffs = coeffs.copy()
    detail = coeffs[1:]                         # (N-1, D)
    denoised_coeffs[1:] = np.sign(detail) * np.maximum(
        np.abs(detail) - threshold[np.newaxis, :], 0
    )

    # Inverse Haar transform
    denoised_buf = H @ denoised_coeffs          # (N, D)

    # EMA-weighted average across time
    clean = ema_weights @ denoised_buf          # (D,)

    return clean

def run_fd_adam(sigma, n_steps=2000, lr=0.01, seed=0,
                sketch_size=8, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Adam with Frequent Directions gradient denoising.
    Per-parameter-group FD sketch — one sketch per layer.
    """
    from problems.xor_problem import xor_loss_and_grads_noisy, \
                                      init_params, accuracy
    from core.frequent_directions import FrequentDirections

    rng = np.random.default_rng(seed)
    params = init_params(seed=seed)
    history = []

    param_sizes  = {k: v.size  for k, v in params.items()}
    param_shapes = {k: v.shape for k, v in params.items()}

    # One FD sketch per parameter group
    sketches = {k: FrequentDirections(sketch_size=sketch_size,
                                       D=param_sizes[k])
                for k in params}

    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}

    for t in range(1, n_steps + 1):
        loss, grads = xor_loss_and_grads_noisy(params, sigma=sigma, rng=rng)
        history.append((t, loss, accuracy(params)))

        clean_grads = {}
        for k in params:
            g_flat = grads[k].flatten()
            sketches[k].update(g_flat)
            if sketches[k].is_ready:
                clean_grads[k] = sketches[k].denoise(g_flat).reshape(param_shapes[k])
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

if __name__ == "__main__":
    from experiments.session9_neural import run_xor_noisy_adam, run_xor_noisy_wavelet_adam
    import numpy as np

    sigma_levels = [0.0, 0.05, 0.1, 0.2]
    n_trials = 10
    n_steps  = 2000

    print(f"\n{'sigma':>8} {'optimizer':>20} {'final_loss':>12} {'solved':>8}")
    print("-" * 55)

    for sigma in sigma_levels:
        configs = [
            ("Adam",          lambda s, sg=sigma: run_xor_noisy_adam(sg, n_steps, seed=s)),
            ("EMAVisuShrink", lambda s, sg=sigma: run_ema_visuShrink_adam(sg, n_steps, seed=s)),
            ("FD",            lambda s, sg=sigma: run_fd_adam(sg, n_steps, seed=s)),
        ]
        for name, fn in configs:
            losses, solved = [], []
            for seed in range(n_trials):
                h, p = fn(seed)
                from problems.xor_problem import accuracy
                losses.append(h[-1][1])
                solved.append(accuracy(p) == 1.0)
            print(f"{sigma:>8.2f} {name:>20} "
                  f"{np.mean(losses):>12.6f} "
                  f"{sum(solved)}/{n_trials}")
        print()