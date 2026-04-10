import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problems.isic_problem import (
    load_isic_dataset, init_params, isic_loss_and_grads,
    get_batches, compute_auc, N_FEATURES
)
from experiments.session11_adaptive import denoise_visuShrink_batch
from experiments.session13_adaptive_snr import denoise_adaptive_snr_batch

# =============================================================
# SESSION 14 — MEDICAL BENCHMARK: ISIC MELANOMA
# + Context-Aware Gradient Denoising
# =============================================================

BUFFER_SIZE  = 8
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
N_EPOCHS     = 30
BATCH_SIZE   = 8
LR           = 0.0005

decay = 0.85
EMA_WEIGHTS = np.array([decay**(BUFFER_SIZE-1-i) for i in range(BUFFER_SIZE)])
EMA_WEIGHTS /= EMA_WEIGHTS.sum()

MELANOMA_DIR = "data/isic/melanoma"
BENIGN_DIR   = "data/isic/benign"


# =============================================================
# BASELINE: ADAM
# =============================================================
def run_adam_isic(X_train, y_train, X_test, y_test, class_weights,
                  n_epochs=N_EPOCHS, lr=LR, batch_size=BATCH_SIZE, seed=0):

    rng = np.random.default_rng(seed)
    params = init_params(seed=seed)

    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    t = 0

    for epoch in range(n_epochs):
        for X_b, y_b in get_batches(X_train, y_train, batch_size, rng):
            t += 1
            _, grads, _, _ = isic_loss_and_grads(
                params, X_b, y_b, class_weights)

            for k in params:
                m[k] = BETA1*m[k] + (1-BETA1)*grads[k]
                v[k] = BETA2*v[k] + (1-BETA2)*grads[k]**2
                mh = m[k] / (1-BETA1**t)
                vh = v[k] / (1-BETA2**t)
                params[k] -= lr * mh / (np.sqrt(vh) + EPS)

    return params


# =============================================================
# DENOISED OPTIMIZER (WITH CONTEXT-AWARE BLENDING)
# =============================================================
def run_denoised_isic(X_train, y_train, X_test, y_test, class_weights,
                     denoise_fn,
                     n_epochs=N_EPOCHS, lr=LR, batch_size=BATCH_SIZE, seed=0):

    rng = np.random.default_rng(seed)
    params = init_params(seed=seed)

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
            _, grads, _, _ = isic_loss_and_grads(
                params, X_b, y_b, class_weights)

            idx = (t - 1) % BUFFER_SIZE

            # --- store gradients ---
            for k in params:
                grad_history[k][idx] = grads[k].flatten()

            buf_count += 1
            buffer_full = buf_count >= BUFFER_SIZE

            clean_grads = {}

            for k in params:
                if buffer_full:
                    # correct temporal ordering
                    order = [(idx + 1 + i) % BUFFER_SIZE for i in range(BUFFER_SIZE)]
                    buf = grad_history[k][order]

                    clean_grads[k] = denoise_fn(
                        buf, EMA_WEIGHTS
                    ).reshape(param_shapes[k])
                else:
                    clean_grads[k] = grads[k]

            # =====================================================
            # CONTEXT-AWARE BLENDING (CORE ADDITION)
            # =====================================================
            for k in params:
                g_raw = grads[k]
                g_d   = clean_grads[k]

                if buffer_full:
                    order = [(idx + 1 + i) % BUFFER_SIZE for i in range(BUFFER_SIZE)]
                    buf = grad_history[k][order]

                    # --- SNR estimation ---
                    signal = np.mean(g_raw**2)
                    noise  = np.var(buf - buf.mean(axis=0))
                    snr = signal / (noise + 1e-8)

                    # --- blending coefficient ---
                    alpha = snr / (snr + 1.0)

                    # blended gradient
                    g = alpha * g_raw + (1 - alpha) * g_d
                else:
                    g = g_raw

                # --- Adam update ---
                m[k] = BETA1*m[k] + (1-BETA1)*g
                v[k] = BETA2*v[k] + (1-BETA2)*g**2
                mh = m[k] / (1-BETA1**t)
                vh = v[k] / (1-BETA2**t)
                params[k] -= lr * mh / (np.sqrt(vh) + EPS)

    return params


# =============================================================
# STATS
# =============================================================
def bootstrap_ci(aucs_a, aucs_b, n_bootstrap=10000, alpha=0.05):
    rng = np.random.default_rng(42)
    diffs = []

    for _ in range(n_bootstrap):
        a = rng.choice(aucs_a, len(aucs_a), replace=True)
        b = rng.choice(aucs_b, len(aucs_b), replace=True)
        diffs.append(a.mean() - b.mean())

    diffs = np.array(diffs)
    ci_low  = np.percentile(diffs, 100*alpha/2)
    ci_high = np.percentile(diffs, 100*(1-alpha/2))

    mean_diff = aucs_a.mean() - aucs_b.mean()
    significant = ci_low > 0 or ci_high < 0

    return mean_diff, ci_low, ci_high, significant


# =============================================================
# BENCHMARK
# =============================================================
def benchmark(n_seeds=5, noise_rate=0.0):

    X_train, y_train, X_test, y_test, cw, _, _ = load_isic_dataset(
        MELANOMA_DIR, BENIGN_DIR,
        n_melanoma=150, n_benign=150,
        test_fraction=0.2, seed=0
    )

    # --- label noise ---
    def corrupt_labels(y, noise_rate=0.0, seed=0):
        if noise_rate == 0:
            return y
        rng = np.random.default_rng(seed)
        noisy_y = y.copy()
        idx = rng.choice(len(y), int(len(y)*noise_rate), replace=False)
        noisy_y[idx] = 1 - noisy_y[idx]
        return noisy_y

    y_train = corrupt_labels(y_train, noise_rate)

    configs = [
        ("Adam",
         lambda seed: run_adam_isic(X_train, y_train, X_test, y_test, cw, seed=seed)),

        ("VisuShrink",
         lambda seed: run_denoised_isic(X_train, y_train, X_test, y_test, cw,
                                       denoise_visuShrink_batch, seed=seed)),

        ("AdaptiveSNR",
         lambda seed: run_denoised_isic(X_train, y_train, X_test, y_test, cw,
                                       denoise_adaptive_snr_batch, seed=seed)),
    ]

    results = {}

    for name, fn in configs:
        aucs = []
        print(f"\n{name}")

        for seed in range(n_seeds):
            params = fn(seed)
            auc = compute_auc(params, X_test, y_test)
            aucs.append(auc)
            print(f"  seed {seed}: {auc:.4f}")

        results[name] = np.array(aucs)

    print("\nFINAL RESULTS")
    for name, aucs in results.items():
        print(f"{name:>15}: {aucs.mean():.4f} ± {aucs.std():.4f}")

    return results


# =============================================================
# RUN
# =============================================================
if __name__ == "__main__":
    for noise in [0.0, 0.1, 0.2, 0.3]:
        print(f"\n{'='*50}")
        print(f"NOISE RATE = {noise:.0%}")
        print(f"{'='*50}")
        results = benchmark(n_seeds=10, noise_rate=noise)

        for name_a, name_b in [("AdaptiveSNR", "Adam"), ("VisuShrink", "Adam")]:
            diff, lo, hi, sig = bootstrap_ci(results[name_a], results[name_b])
            print(f"{name_a} vs {name_b}: diff={diff:+.4f} "
                  f"CI=[{lo:.4f},{hi:.4f}] sig={sig}")