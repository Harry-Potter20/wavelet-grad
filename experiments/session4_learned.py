import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problems.rosenbrock import rosenbrock
from core.wavelets import haar_forward, haar_inverse
from core.wavelets import make_scale_weight_vector, soft_wavelet_denoise, sigmoid, sigmoid_grad
from core.buffers import CircularGradientBuffer

# =============================================================
# SESSION 4: LEARNING THE SCALE WEIGHTS
# =============================================================
#
# Goal: learn raw_params such that soft_wavelet_denoise produces
# a gradient estimate closer to the true gradient than hard
# thresholding does.
#
# Training signal: run Adam, collect (buffer_snapshot, next_gradient) pairs.
#   - buffer_snapshot: last 8 gradients before step t
#   - next_gradient:   the raw gradient at step t (our "label")
#
# Loss: MSE between denoised[-1] and next_gradient
#
# We differentiate through the wavelet transform manually.
# This is good practice — you'll understand autograd better
# for having done it by hand first.
# =============================================================


# --- Step 1: Collect training data from Adam ---

def collect_gradient_data(n_steps=500, buffer_capacity=8,
                           lr=0.01, start=(-1.0, 1.0)):
    """
    Run Adam and collect (signal, target) pairs.

    signal: buffer contents (last `buffer_capacity` gradients) for param x
    target: the raw gradient at the next step

    Returns:
        pairs_x: list of (signal, target) for gx
        pairs_y: list of (signal, target) for gy
    """
    x, y = start
    buf_x = CircularGradientBuffer(capacity=buffer_capacity, grad_shape=(1,))
    buf_y = CircularGradientBuffer(capacity=buffer_capacity, grad_shape=(1,))

    mx, my = 0.0, 0.0
    vx, vy = 0.0, 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    pairs_x, pairs_y = [], []

    for t in range(1, n_steps + 1):
        loss, gx, gy = rosenbrock(x, y)

        # If buffer is full, record the snapshot before writing new gradient
        if buf_x.is_full():
            snap_x = buf_x.read().flatten()
            snap_y = buf_y.read().flatten()
            # Target: the current raw gradient (what we want to predict)
            pairs_x.append((snap_x.copy(), gx))
            pairs_y.append((snap_y.copy(), gy))

        buf_x.write(np.array([gx]))
        buf_y.write(np.array([gy]))

        # Standard Adam update
        mx = beta1 * mx + (1 - beta1) * gx
        my = beta1 * my + (1 - beta1) * gy
        vx = beta2 * vx + (1 - beta2) * gx**2
        vy = beta2 * vy + (1 - beta2) * gy**2

        mx_hat = mx / (1 - beta1**t)
        my_hat = my / (1 - beta1**t)
        vx_hat = vx / (1 - beta2**t)
        vy_hat = vy / (1 - beta2**t)

        x = x - lr * mx_hat / (vx_hat**0.5 + eps)
        y = y - lr * my_hat / (vy_hat**0.5 + eps)

    return pairs_x, pairs_y


pairs_x, pairs_y = collect_gradient_data(n_steps=500)
print(f"Collected {len(pairs_x)} training pairs")
print(f"Signal shape: {pairs_x[0][0].shape}")
print(f"Sample target gx: {pairs_x[0][1]:.4f}")
print(f"Sample target gy: {pairs_y[0][1]:.4f}")


# --- Step 2: Training loop ---
#
# Loss for one (signal, target) pair:
#   L = (denoised[-1] - target)²
#
# We need dL/d(raw_params) to update the parameters.
#
# Chain rule:
#   dL/d(raw_params[k])
#     = dL/d(denoised[-1])
#     * d(denoised[-1])/d(weighted_coeffs)
#     * d(weighted_coeffs)/d(weights)
#     * d(weights)/d(raw_params[k])
#
# Breaking it down:
#   dL/d(denoised[-1])          = 2 * (denoised[-1] - target)
#   d(denoised[-1])/d(w_coeffs) = last row of IDWT matrix (we'll approximate)
#   d(weighted_coeffs[i])/d(weights[i]) = coeffs[i]
#   d(weights[k_group])/d(raw_params[k]) = sigmoid_grad(raw_params[k])
#
# We'll use numerical gradients first to verify, then analytical.

def compute_loss(signal, target, raw_params):
    """MSE loss for one training pair."""
    denoised, _, _, _ = soft_wavelet_denoise(signal, raw_params)
    return (denoised - target)**2


def numerical_gradient(signal, target, raw_params, eps=1e-5):
    """
    Finite difference gradient of loss w.r.t. raw_params.
    Used to verify our analytical gradient.

    For each parameter p[k]:
        dL/dp[k] ≈ (L(p[k]+eps) - L(p[k]-eps)) / (2*eps)
    """
    grad = np.zeros_like(raw_params)
    for k in range(len(raw_params)):
        params_plus  = raw_params.copy(); params_plus[k]  += eps
        params_minus = raw_params.copy(); params_minus[k] -= eps
        loss_plus  = compute_loss(signal, target, params_plus)
        loss_minus = compute_loss(signal, target, params_minus)
        grad[k] = (loss_plus - loss_minus) / (2 * eps)
    return grad


def analytical_gradient(signal, target, raw_params):
    """
    Analytical gradient of loss w.r.t. raw_params.

    Derived via chain rule through:
    raw_params → sigmoid → weights → weighted_coeffs → IDWT → denoised[-1] → loss
    """
    import math
    n = len(signal)
    J = int(math.log2(n))

    denoised_val, coeffs, weighted_coeffs, weights = soft_wavelet_denoise(
        signal, raw_params
    )

    # dL/d(denoised_val)
    dL_dout = 2.0 * (denoised_val - target)

    # d(denoised[-1])/d(weighted_coeffs[i])
    # The IDWT is a linear operation. The last output element is a
    # linear combination of all weighted_coeffs.
    # We get this by perturbing each coefficient and seeing the effect.
    dout_dwcoeffs = np.zeros(n)
    eps = 1e-7
    for i in range(n):
        wc_plus  = weighted_coeffs.copy(); wc_plus[i]  += eps
        wc_minus = weighted_coeffs.copy(); wc_minus[i] -= eps
        dout_dwcoeffs[i] = (haar_inverse(wc_plus)[-1] -
                            haar_inverse(wc_minus)[-1]) / (2 * eps)

    # d(weighted_coeffs[i])/d(weights[i]) = coeffs[i]
    # d(weights[group_k])/d(raw_params[k]) = sigmoid_grad(raw_params[k])
    grad = np.zeros(J)
    for k in range(J):
        start = 2**k
        end   = 2**(k+1)
        # Sum over all coefficients in this scale group
        dL_dwk = np.sum(dL_dout * dout_dwcoeffs[start:end] * coeffs[start:end])
        grad[k] = dL_dwk * sigmoid_grad(raw_params[k])

    return grad


# --- Step 3: Verify numerical vs analytical gradient ---
print("\n--- Gradient check ---")
signal, target = pairs_x[0]
raw_params = np.zeros(3)   # J=3 for capacity-8 buffer

num_grad  = numerical_gradient(signal, target, raw_params)
anal_grad = analytical_gradient(signal, target, raw_params)

print(f"Numerical gradient:  {np.round(num_grad, 6)}")
print(f"Analytical gradient: {np.round(anal_grad, 6)}")
print(f"Max difference: {np.max(np.abs(num_grad - anal_grad)):.2e}")
print(f"Gradients match: {np.allclose(num_grad, anal_grad, atol=1e-4)}")

# --- Step 4: Train the scale weights ---

def train_scale_weights(pairs, n_epochs=50, lr=0.01, seed=0):
    """
    Train raw_params via gradient descent on the collected pairs.

    Args:
        pairs:    list of (signal, target) tuples
        n_epochs: passes over the full dataset
        lr:       learning rate for raw_params update

    Returns:
        raw_params: trained parameters
        loss_history: mean loss per epoch
    """
    np.random.seed(seed)
    J = 3  # log2(8) — three detail scales
    raw_params = np.zeros(J)  # initialize: sigmoid(0) = 0.5, all scales half-weighted
    loss_history = []

    for epoch in range(n_epochs):
        epoch_losses = []

        # Shuffle pairs each epoch
        indices = np.random.permutation(len(pairs))

        for idx in indices:
            signal, target = pairs[idx]

            # Forward
            loss = compute_loss(signal, target, raw_params)
            epoch_losses.append(loss)

            # Backward
            grad = analytical_gradient(signal, target, raw_params)

            # Update raw_params
            raw_params = raw_params - lr * grad

        mean_loss = np.mean(epoch_losses)
        loss_history.append(mean_loss)

        if epoch % 10 == 0:
            weights = sigmoid(raw_params)
            print(f"Epoch {epoch:3d}  loss={mean_loss:.6f}  "
                  f"weights={np.round(weights, 3)}")

    return raw_params, loss_history


print("\n--- Training scale weights on gx ---")
raw_params_x, loss_x = train_scale_weights(pairs_x, n_epochs=50, lr=0.01)

print("\n--- Training scale weights on gy ---")
raw_params_y, loss_y = train_scale_weights(pairs_y, n_epochs=50, lr=0.01)

# --- Step 5: Analyze learned weights ---
print("\n--- Learned weights ---")
weights_x = sigmoid(raw_params_x)
weights_y = sigmoid(raw_params_y)

print(f"\ngx scale weights:")
print(f"  scale-3 (coarsest detail): {weights_x[0]:.4f}")
print(f"  scale-2 (mid detail):      {weights_x[1]:.4f}")
print(f"  scale-1 (finest detail):   {weights_x[2]:.4f}")

print(f"\ngy scale weights:")
print(f"  scale-3 (coarsest detail): {weights_y[0]:.4f}")
print(f"  scale-2 (mid detail):      {weights_y[1]:.4f}")
print(f"  scale-1 (finest detail):   {weights_y[2]:.4f}")

print(f"\nInterpretation:")
for name, w in [("gx", weights_x), ("gy", weights_y)]:
    dominant = ["scale-3", "scale-2", "scale-1"][np.argmax(w)]
    suppressed = ["scale-3", "scale-2", "scale-1"][np.argmin(w)]
    print(f"  {name}: keeps {dominant} most, suppresses {suppressed} most")
    
# --- Step 6: WaveletAdam with learned weights ---
from core.optimizers import run_adam
from core.optimizers import run_learned_wavelet_adam


print("\n--- Final benchmark ---")

h_adam = run_adam(lr=0.01, n_steps=2000)
converged_adam = next((i for i,(x,y,l) in enumerate(h_adam)
                       if abs(x-1)<1e-4 and abs(y-1)<1e-4), None)
last = h_adam[-1]
print(f"Adam                loss={last[2]:.6f}  converged={converged_adam}")

h_learned = run_learned_wavelet_adam(
    lr=0.01, n_steps=2000,
    raw_params_x=raw_params_x,
    raw_params_y=raw_params_y,
)
converged_learned = next((i for i,(x,y,l) in enumerate(h_learned)
                          if abs(x-1)<1e-4 and abs(y-1)<1e-4), None)
last = h_learned[-1]
print(f"LearnedWaveletAdam  loss={last[2]:.6f}  converged={converged_learned}")

# Also compare against best hard threshold from session 3
from core.optimizers import run_wavelet_adam
h_hard = run_wavelet_adam(lr=0.01, n_steps=2000,
                           buffer_capacity=8, threshold=0.5)
converged_hard = next((i for i,(x,y,l) in enumerate(h_hard)
                       if abs(x-1)<1e-4 and abs(y-1)<1e-4), None)
last = h_hard[-1]
print(f"HardWaveletAdam     loss={last[2]:.6f}  converged={converged_hard}")

# Diagnostic 1: train longer
print("\n--- Diagnostic: train longer ---")
raw_params_x_long, loss_x_long = train_scale_weights(
    pairs_x, n_epochs=200, lr=0.01
)
raw_params_y_long, loss_y_long = train_scale_weights(
    pairs_y, n_epochs=200, lr=0.01
)

h_long = run_learned_wavelet_adam(
    lr=0.01, n_steps=2000,
    raw_params_x=raw_params_x_long,
    raw_params_y=raw_params_y_long,
)
converged_long = next((i for i,(x,y,l) in enumerate(h_long)
                       if abs(x-1)<1e-4 and abs(y-1)<1e-4), None)
last = h_long[-1]
print(f"LearnedWaveletAdam 200 epochs  loss={last[2]:.6f}  converged={converged_long}")

# Diagnostic 2: test on held-out starting point
print("\n--- Diagnostic: held-out start ---")
pairs_x_held, pairs_y_held = collect_gradient_data(
    n_steps=500, start=(-0.5, 0.5)   # different start
)

# Use weights trained on (-1, 1) — tests generalization
h_held = run_learned_wavelet_adam(
    lr=0.01, n_steps=2000,
    raw_params_x=raw_params_x_long,
    raw_params_y=raw_params_y_long,
    start=(-0.5, 0.5),
)
converged_held = next((i for i,(x,y,l) in enumerate(h_held)
                       if abs(x-1)<1e-4 and abs(y-1)<1e-4), None)
last = h_held[-1]
print(f"Held-out start (-0.5, 0.5)  loss={last[2]:.6f}  converged={converged_held}")

h_adam_held = run_adam(lr=0.01, n_steps=2000, start=(-0.5, 0.5))
converged_ah = next((i for i,(x,y,l) in enumerate(h_adam_held)
                     if abs(x-1)<1e-4 and abs(y-1)<1e-4), None)
last = h_adam_held[-1]
print(f"Adam held-out start          loss={last[2]:.6f}  converged={converged_ah}")