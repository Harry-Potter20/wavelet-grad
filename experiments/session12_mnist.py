import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problems.mnist_problem import (
    load_mnist_subset, init_params, mnist_loss_and_grads, get_batches
)
from experiments.session11_adaptive import (
    denoise_visuShrink, denoise_visuShrink_batch
)



# Low Rank Projection 
from core.frequent_directions import FrequentDirections

# =============================================================
# SESSION 12 — PER-LAYER DENOISING ON MNIST
#
# Network: 784 → 128 → 10
# Parameters: W1(128,784), b1(128), W2(10,128), b2(10)
#
# Key insight: per-layer denoising estimates σ independently
# per parameter group. W1 and W2 have gradients differing
# by ~1000×. A single global threshold calibrated to W1
# under-denoises W2 and vice versa.
#
# Vectorised: all D parameters denoised in one matrix multiply.
# =============================================================

BUFFER_SIZE  = 8
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
N_EPOCHS     = 50
BATCH_SIZE   = 64
LR           = 0.001

decay = 0.85
EMA_WEIGHTS = np.array([decay**(BUFFER_SIZE-1-i) for i in range(BUFFER_SIZE)])
EMA_WEIGHTS /= EMA_WEIGHTS.sum()


def run_adam_mnist(X_train, y_train, X_test, y_test,
                   n_epochs=N_EPOCHS, lr=LR, batch_size=BATCH_SIZE, seed=0):
    """Standard Adam on MNIST."""
    rng = np.random.default_rng(seed)
    params = init_params(seed=seed)
    history = []

    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    t = 0

    for epoch in range(n_epochs):
        for X_b, y_b in get_batches(X_train, y_train, batch_size, rng):
            t += 1
            loss, grads, acc = mnist_loss_and_grads(params, X_b, y_b)
            for k in params:
                m[k] = BETA1*m[k] + (1-BETA1)*grads[k]
                v[k] = BETA2*v[k] + (1-BETA2)*grads[k]**2
                mh = m[k] / (1-BETA1**t)
                vh = v[k] / (1-BETA2**t)
                params[k] -= lr * mh / (np.sqrt(vh) + EPS)

        if (epoch + 1) % 10 == 0:
            _, _, test_acc  = mnist_loss_and_grads(params, X_test, y_test)
            tl, _, train_acc = mnist_loss_and_grads(params, X_train[:500], y_train[:500])
            history.append((epoch+1, tl, train_acc, test_acc))
            print(f"  epoch {epoch+1:3d}  loss={tl:.4f}  "
                  f"train={train_acc:.3f}  test={test_acc:.3f}")

    return history, params


def run_perlayer_visuShrink_adam(X_train, y_train, X_test, y_test,
                                  n_epochs=N_EPOCHS, lr=LR,
                                  batch_size=BATCH_SIZE, seed=0):
    """
    Adam with per-layer vectorised VisuShrink denoising.

    Each parameter group has its own buffer — threshold calibrated
    to that layer's gradient scale independently.
    Vectorised: all D parameters processed in one matrix multiply.
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
                    order = [(idx + 1 + i) % BUFFER_SIZE for i in range(BUFFER_SIZE)]
                    buf = grad_history[k][order]             # (N, D)
                    clean_grads[k] = denoise_visuShrink_batch(
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
            tl, _, train_acc = mnist_loss_and_grads(params, X_train[:500], y_train[:500])
            history.append((epoch+1, tl, train_acc, test_acc))
            print(f"  epoch {epoch+1:3d}  loss={tl:.4f}  "
                  f"train={train_acc:.3f}  test={test_acc:.3f}")

    return history, params

def run_fd_adam_mnist(X_train, y_train, X_test, y_test,
                       n_epochs=N_EPOCHS, lr=LR,
                       batch_size=BATCH_SIZE, sketch_size=8, seed=0):
    """
    Adam with Frequent Directions denoising on MNIST.
    Per-layer FD sketch — one per parameter group.
    FD shines here: D=100K >> k=8.
    """
    from core.frequent_directions import FrequentDirections

    rng = np.random.default_rng(seed)
    params = init_params(seed=seed)
    history = []

    param_shapes = {k: v.shape for k, v in params.items()}
    param_sizes  = {k: v.size  for k, v in params.items()}

    sketches = {k: FrequentDirections(sketch_size=sketch_size,
                                       D=param_sizes[k])
                for k in params}

    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    t = 0

    for epoch in range(n_epochs):
        for X_b, y_b in get_batches(X_train, y_train, batch_size, rng):
            t += 1
            loss, grads, acc = mnist_loss_and_grads(params, X_b, y_b)

            clean_grads = {}
            for k in params:
                g_flat = grads[k].flatten()
                sketches[k].update(g_flat)
                if sketches[k].is_ready:
                    clean_grads[k] = sketches[k].denoise(
                        g_flat).reshape(param_shapes[k])
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

def run_learned_correction_adam(X_train, y_train, X_test, y_test,
                                 n_epochs=N_EPOCHS, lr=LR,
                                 batch_size=BATCH_SIZE, seed=0):
    """Adam + learned threshold correction denoising on MNIST."""
    from core.learned_threshold_correction import train_correction_model
    from experiments.session13_adaptive_snr import denoise_learned_correction_batch

    print("  Training correction model...")
    net, X_mean, X_std = train_correction_model(
        n_samples=5000, n_epochs=30, seed=seed
    )

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
                    clean_grads[k] = denoise_learned_correction_batch(
                        buf, EMA_WEIGHTS, net, X_mean, X_std
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

    print("--- Adam ---")
    h_adam, _ = run_adam_mnist(X_train, y_train, X_test, y_test)

    print("\n--- PerLayerVisuShrink ---")
    h_vs, _ = run_perlayer_visuShrink_adam(X_train, y_train, X_test, y_test)
    
    print("\n--- FD (sketch_size=8) ---")
    h_fd, _ = run_fd_adam_mnist(X_train, y_train, X_test, y_test)

    print("\n--- Final Comparison ---")
    print(f"{'optimizer':>25} {'train_loss':>12} {'train_acc':>10} {'test_acc':>10}")
    print("-" * 60)
    for name, h in [("Adam", h_adam), ("PerLayerVisuShrink", h_vs), ("FD k=8", h_fd)]:
        if h:
            e, tl, ta, tea = h[-1]
            print(f"{name:>25} {tl:>12.4f} {ta:>10.3f} {tea:>10.3f}")


benchmark()