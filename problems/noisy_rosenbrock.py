import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# =============================================================
# NOISY ROSENBROCK
#
# Same loss landscape as Rosenbrock, but gradient observations
# are corrupted by additive Gaussian noise.
#
# f(x, y) = (1-x)² + 100(y-x²)²  ← true loss, unchanged
#
# Observed gradients:
#   gx_obs = gx_true + σ * ε_x,   ε_x ~ N(0, 1)
#   gy_obs = gy_true + σ * ε_y,   ε_y ~ N(0, 1)
#
# The optimizer sees only the noisy gradient.
# The true loss is used only for evaluation.
#
# This models real DL training where:
#   - Mini-batch gradients are noisy estimates of true gradient
#   - σ corresponds to batch size: small batch → large σ
# =============================================================

from problems.rosenbrock import rosenbrock


def noisy_rosenbrock(x, y, sigma, rng=None):
    """
    Returns true loss and noisy gradient at (x, y).

    Args:
        x, y:   current parameters
        sigma:  noise standard deviation
        rng:    numpy random Generator (for reproducibility)
                if None, uses global numpy random state

    Returns:
        loss:   true f(x, y) — no noise on loss, only on gradients
        gx_obs: noisy ∂f/∂x
        gy_obs: noisy ∂f/∂y
    """
    loss, gx_true, gy_true = rosenbrock(x, y)

    if rng is None:
        noise_x = np.random.randn()
        noise_y = np.random.randn()
    else:
        noise_x = rng.standard_normal()
        noise_y = rng.standard_normal()

    gx_obs = gx_true + sigma * noise_x
    gy_obs = gy_true + sigma * noise_y

    return loss, gx_obs, gy_obs


def gradient_stats(n_steps=2000, lr=0.01, seed=42):
    """
    Run clean Adam and report gradient magnitude statistics.
    Used to calibrate sigma relative to signal strength.
    """
    from core.optimizers import run_adam_instrumented
    _, grad_log, _ = run_adam_instrumented(lr=lr, n_steps=n_steps)
    norms = [np.sqrt(gx**2 + gy**2) for gx, gy in grad_log]
    return {
        'mean':   np.mean(norms),
        'median': np.median(norms),
        'max':    np.max(norms),
        'min':    np.min(norms),
        'std':    np.std(norms),
    }


if __name__ == "__main__":
    # Calibrate noise levels
    stats = gradient_stats()
    print("Gradient norm statistics (clean Adam, 2000 steps):")
    for k, v in stats.items():
        print(f"  {k:8s}: {v:.4f}")

    mean_norm = stats['mean']
    print(f"\nRecommended sigma values:")
    for frac, label in [(0.1, 'mild'), (0.5, 'moderate'),
                        (1.0, 'heavy'), (2.0, 'severe')]:
        print(f"  σ={frac*mean_norm:.3f}  ({label})")