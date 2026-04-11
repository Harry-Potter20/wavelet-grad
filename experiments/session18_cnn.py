import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problems.mnist_problem import load_mnist_subset
from problems.cnn_problem import init_params, cnn_loss_and_grads, get_batches
from core.waveguard import train_waveguard, denoise_waveguard_batch

# =============================================================
# SESSION 18 — PER-FILTER WAVEGUARD ON CNN (FIXED)
# =============================================================

BUFFER_SIZE  = 8
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
N_EPOCHS     = 30
BATCH_SIZE   = 32
LR           = 0.001

decay = 0.85
EMA_WEIGHTS = np.array([decay**(BUFFER_SIZE-1-i) for i in range(BUFFER_SIZE)])
EMA_WEIGHTS /= EMA_WEIGHTS.sum()


# =============================================================
# BASELINE — ADAM
# =============================================================

def run_adam_cnn(X_train, y_train, X_test, y_test,
                n_epochs=N_EPOCHS, lr=LR, batch_size=BATCH_SIZE, seed=0):

    rng = np.random.default_rng(seed)
    params = init_params(seed=seed)
    history = []

    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    t = 0

    for epoch in range(n_epochs):
        for X_b, y_b in get_batches(X_train, y_train, batch_size, rng):
            t += 1
            loss, grads, acc = cnn_loss_and_grads(params, X_b, y_b)

            for k in params:
                m[k] = BETA1*m[k] + (1-BETA1)*grads[k]
                v[k] = BETA2*v[k] + (1-BETA2)*grads[k]**2

                mh = m[k] / (1-BETA1**t)
                vh = v[k] / (1-BETA2**t)

                params[k] -= lr * mh / (np.sqrt(vh) + EPS)

        if (epoch + 1) % 5 == 0:
            loss, _, test_acc = cnn_loss_and_grads(params, X_test, y_test)
            tl, _, tr_acc = cnn_loss_and_grads(params, X_train[:200], y_train[:200])

            history.append((epoch+1, tl, tr_acc, test_acc))
            print(f"  epoch {epoch+1:3d}  loss={tl:.4f}  train={tr_acc:.3f}  test={test_acc:.3f}")

    return history, params


# =============================================================
# PER-FILTER WAVEGUARD (FIXED FC HANDLING)
# =============================================================

def run_perfilter_waveguard_cnn(X_train, y_train, X_test, y_test,
                               net, X_mean, X_std,
                               alpha=2.0, beta=1.0,
                               n_epochs=N_EPOCHS, lr=LR,
                               batch_size=BATCH_SIZE, seed=0):

    rng = np.random.default_rng(seed)
    params = init_params(seed=seed)
    history = []

    # =========================
    # CONV BUFFERS (PER FILTER)
    # =========================
    n_filters = {'W1': 8, 'W2': 16}

    filter_dim = {
        'W1': params['W1'][0].size,   # 9
        'W2': params['W2'][0].size,   # 72
    }

    conv_buffers = {
        k: np.zeros((n_filters[k], BUFFER_SIZE, filter_dim[k]))
        for k in n_filters
    }

    # =========================
    # FC BUFFERS (PER OUTPUT)
    # =========================
    n_fc_units = params['Wfc'].shape[0]   # 10
    fc_dim     = params['Wfc'].shape[1]   # 576

    fc_buffers = np.zeros((n_fc_units, BUFFER_SIZE, fc_dim))

    # =========================
    # BIAS BUFFERS
    # =========================
    bias_buffers = {
        'b1':  np.zeros((BUFFER_SIZE, params['b1'].size)),
        'b2':  np.zeros((BUFFER_SIZE, params['b2'].size)),
        'bfc': np.zeros((BUFFER_SIZE, params['bfc'].size)),
    }

    buf_count = 0
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    t = 0

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(n_epochs):
        for X_b, y_b in get_batches(X_train, y_train, batch_size, rng):
            t += 1
            loss, grads, acc = cnn_loss_and_grads(params, X_b, y_b)

            idx = (t - 1) % BUFFER_SIZE
            buf_count += 1
            buffer_full = buf_count >= BUFFER_SIZE

            # -------------------------
            # STORE GRADIENTS
            # -------------------------
            for k in ['W1', 'W2']:
                for f in range(n_filters[k]):
                    conv_buffers[k][f, idx] = grads[k][f].flatten()

            for i in range(n_fc_units):
                fc_buffers[i, idx] = grads['Wfc'][i]

            for k in bias_buffers:
                bias_buffers[k][idx] = grads[k].flatten()

            clean_grads = {}

            # -------------------------
            # DENOISE CONV FILTERS
            # -------------------------
            for k in ['W1', 'W2']:
                clean_W = np.zeros_like(params[k])

                for f in range(n_filters[k]):
                    if buffer_full:
                        order = [(idx + 1 + i) % BUFFER_SIZE for i in range(BUFFER_SIZE)]
                        buf = conv_buffers[k][f][order]

                        clean_W[f] = denoise_waveguard_batch(
                            buf, EMA_WEIGHTS, net, X_mean, X_std,
                            alpha=alpha, beta=beta
                        ).reshape(params[k][f].shape)
                    else:
                        clean_W[f] = grads[k][f]

                clean_grads[k] = clean_W

            # -------------------------
            # DENOISE FC (PER OUTPUT)
            # -------------------------
            clean_Wfc = np.zeros_like(params['Wfc'])

            for i in range(n_fc_units):
                if buffer_full:
                    order = [(idx + 1 + j) % BUFFER_SIZE for j in range(BUFFER_SIZE)]
                    buf = fc_buffers[i][order]

                    clean_Wfc[i] = denoise_waveguard_batch(
                        buf, EMA_WEIGHTS, net, X_mean, X_std,
                        alpha=alpha, beta=beta
                    )
                else:
                    clean_Wfc[i] = grads['Wfc'][i]

            clean_grads['Wfc'] = clean_Wfc

            # -------------------------
            # DENOISE BIASES
            # -------------------------
            for k in bias_buffers:
                if buffer_full:
                    order = [(idx + 1 + i) % BUFFER_SIZE for i in range(BUFFER_SIZE)]
                    buf = bias_buffers[k][order]

                    clean_grads[k] = denoise_waveguard_batch(
                        buf, EMA_WEIGHTS, net, X_mean, X_std,
                        alpha=alpha, beta=beta
                    ).reshape(params[k].shape)
                else:
                    clean_grads[k] = grads[k]

            # -------------------------
            # ADAM UPDATE
            # -------------------------
            for k in params:
                g = clean_grads[k]

                m[k] = BETA1*m[k] + (1-BETA1)*g
                v[k] = BETA2*v[k] + (1-BETA2)*g**2

                mh = m[k] / (1-BETA1**t)
                vh = v[k] / (1-BETA2**t)

                params[k] -= lr * mh / (np.sqrt(vh) + EPS)

        # -------------------------
        # LOGGING
        # -------------------------
        if (epoch + 1) % 5 == 0:
            loss, _, test_acc = cnn_loss_and_grads(params, X_test, y_test)
            tl, _, tr_acc = cnn_loss_and_grads(params, X_train[:200], y_train[:200])

            history.append((epoch+1, tl, tr_acc, test_acc))
            print(f"  epoch {epoch+1:3d}  loss={tl:.4f}  train={tr_acc:.3f}  test={test_acc:.3f}")

    return history, params


# =============================================================
# BENCHMARK
# =============================================================

def benchmark():
    print("Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist_subset(
        n_train=2000, n_test=500, seed=0
    )
    print(f"Train: {X_train.shape}  Test: {X_test.shape}\n")

    print("Training WaveGuard model...")
    net, X_mean, X_std = train_waveguard(n_samples=10000, n_epochs=50)

    print("\n--- Adam (baseline) ---")
    h_adam, _ = run_adam_cnn(X_train, y_train, X_test, y_test)

    print("\n--- WaveGuard (per-filter) ---")
    h_wg, _ = run_perfilter_waveguard_cnn(
        X_train, y_train, X_test, y_test,
        net, X_mean, X_std, alpha=2.0, beta=1.0
    )

    print("\n--- Final Comparison ---")
    print(f"{'optimizer':>25} {'train_loss':>12} {'train_acc':>10} {'test_acc':>10}")
    print("-" * 60)

    for name, h in [("Adam", h_adam), ("WaveGuard", h_wg)]:
        if h:
            e, tl, ta, tea = h[-1]
            print(f"{name:>25} {tl:>12.4f} {ta:>10.3f} {tea:>10.3f}")


if __name__ == "__main__":
    benchmark()