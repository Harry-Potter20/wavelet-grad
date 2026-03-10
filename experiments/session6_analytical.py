import numpy as np
import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problems.rosenbrock import rosenbrock
from core.buffers import CircularGradientBuffer
from core.wavelets import haar_forward, haar_inverse, make_scale_weight_vector
from core.meta_net import MetaNet, sigmoid

from collections import deque

# =============================================================
# SESSION 6: ANALYTICAL META-GRADIENT
#
# We derive dL/d(scale_weights) analytically through:
#
#   scale_weights
#        ↓  make_scale_weight_vector
#   weight_vec
#        ↓  haar_forward, elementwise multiply, haar_inverse
#   gx_clean
#        ↓  Adam moment updates
#   mx_hat, vx_hat
#        ↓  Adam parameter update
#   x_new
#        ↓  rosenbrock
#   loss_new
#
# We build the gradient one step at a time, verify each piece
# against numerical gradient before moving to the next.
# =============================================================


# ---- Shared setup ----
BUFFER_CAPACITY = 8
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
ADAM_LR = 0.01

def make_test_case(seed=42):
    """Generate a reproducible (sig_x, sig_y, mx, my, vx, vy, x, y, t) state."""
    np.random.seed(seed)
    sig_x = np.random.randn(BUFFER_CAPACITY)
    sig_y = np.random.randn(BUFFER_CAPACITY)
    mx, my = np.random.randn(), np.random.randn()
    vx, vy = np.abs(np.random.randn()), np.abs(np.random.randn())
    x,  y  = np.random.randn(), np.random.randn()
    t = 50  # mid-training step
    return sig_x, sig_y, mx, my, vx, vy, x, y, t


# =============================================================
# STEP 1: Gradient through the wavelet denoising
#
# gx_clean = haar_inverse(haar_forward(sig_x) * weight_vec)[-1]
#
# The Haar operations are linear. So:
#   d(gx_clean)/d(weight_vec[i]) = ?
#
# Let C = haar_forward(sig_x)  (fixed — doesn't depend on weights)
# Let W = weight_vec
# Let gx_clean = haar_inverse(C * W)[-1]
#
# Since haar_inverse is linear:
#   d(gx_clean)/d(W[i]) = haar_inverse(C * e_i)[-1]
# where e_i is the i-th standard basis vector.
#
# This is just: apply haar_inverse to a vector that is zero
# everywhere except position i where it equals C[i].
# =============================================================

def dgx_clean_dweightvec(sig_x, weight_vec):
    """
    Gradient of gx_clean w.r.t. each element of weight_vec.

    Returns array of shape (BUFFER_CAPACITY,) where entry i is
    d(gx_clean)/d(weight_vec[i]).
    """
    coeffs = haar_forward(sig_x)   # C — fixed
    n = len(sig_x)
    grad = np.zeros(n)

    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = coeffs[i]         # C * e_i
        grad[i] = haar_inverse(e_i)[-1]

    return grad

# At module level — compute once
_IDWT_LAST_ROW_CACHE = {}

def get_idwt_last_row(n):
    """Cache the last row of the IDWT matrix for a given length."""
    if n not in _IDWT_LAST_ROW_CACHE:
        identity = np.eye(n)
        idwt_matrix = np.array([haar_inverse(identity[j]) for j in range(n)])
        _IDWT_LAST_ROW_CACHE[n] = idwt_matrix[:, -1]
    return _IDWT_LAST_ROW_CACHE[n]


def dgx_clean_dweightvec_fast(sig_x):
    coeffs   = haar_forward(sig_x)
    last_row = get_idwt_last_row(len(sig_x))
    return last_row * coeffs

# Verify against numerical gradient
print("--- Step 1: d(gx_clean)/d(weight_vec) ---")

sig_x, sig_y, mx, my, vx, vy, x, y, t = make_test_case()
scale_weights = np.array([0.6, 0.7, 0.5])
weight_vec = make_scale_weight_vector(scale_weights, BUFFER_CAPACITY)
coeffs_x = haar_forward(sig_x)

# Analytical
anal_grad_wv = dgx_clean_dweightvec(sig_x, weight_vec)

# Numerical
eps = 1e-5
num_grad_wv = np.zeros(BUFFER_CAPACITY)
for i in range(BUFFER_CAPACITY):
    wv_plus  = weight_vec.copy(); wv_plus[i]  += eps
    wv_minus = weight_vec.copy(); wv_minus[i] -= eps
    gxp = haar_inverse(haar_forward(sig_x) * wv_plus)[-1]
    gxm = haar_inverse(haar_forward(sig_x) * wv_minus)[-1]
    num_grad_wv[i] = (gxp - gxm) / (2*eps)

max_diff = np.max(np.abs(anal_grad_wv - num_grad_wv))
print(f"Max difference: {max_diff:.2e}")
print(f"Step 1 passed: {max_diff < 1e-6}")


# =============================================================
# STEP 2: Gradient through the Adam update
#
# x_new = x - lr * mx_hat / (√vx_hat + ε)
#
# where:
#   mx_hat = (β₁·mx_old + (1-β₁)·gx_clean) / (1-β₁ᵗ)
#   vx_hat = (β₂·vx_old + (1-β₂)·gx_clean²) / (1-β₂ᵗ)
#
# We need dx_new/d(gx_clean).
#
# Let:
#   a = 1 - β₁ᵗ  (bias correction for 1st moment)
#   b = 1 - β₂ᵗ  (bias correction for 2nd moment)
#   m = β₁·mx_old + (1-β₁)·gx_clean   (raw 1st moment)
#   v = β₂·vx_old + (1-β₂)·gx_clean²  (raw 2nd moment)
#
# Then:
#   mx_hat = m / a
#   vx_hat = v / b
#   x_new  = x - lr * (m/a) / (√(v/b) + ε)
#
# Quotient rule: d(f/g)/dx = (f'g - fg') / g²
#
# dm/d(gx_clean) = (1-β₁)
# dv/d(gx_clean) = (1-β₂) · 2·gx_clean
#
# dx_new/d(gx_clean)
#   = -lr * d/d(gx_clean) [ (m/a) / (√(v/b) + ε) ]
#
#   = -lr * [ (dm/d(gx)/a) · (√(v/b) + ε)
#             - (m/a) · (dv/d(gx))/(2b·√(v/b)) ]
#           / (√(v/b) + ε)²
#
# =============================================================

def dx_new_dgx_clean(gx_clean, mx_old, vx_old, t):
    """
    Gradient of x_new w.r.t. gx_clean through one Adam step.
    """
    a  = 1 - BETA1**t
    b  = 1 - BETA2**t

    m  = BETA1 * mx_old + (1 - BETA1) * gx_clean
    v  = BETA2 * vx_old + (1 - BETA2) * gx_clean**2

    mx_hat = m / a
    vx_hat = v / b

    denom  = np.sqrt(vx_hat) + EPS
    denom2 = denom**2

    # dm/d(gx_clean)
    dm_dgx = (1 - BETA1) / a

    # d(sqrt(vx_hat))/d(gx_clean)
    # vx_hat = v/b, dv/dgx = (1-β₂)*2*gx_clean
    dv_dgx       = (1 - BETA2) * 2 * gx_clean
    dvx_hat_dgx  = dv_dgx / b
    dsqrt_vx_dgx = dvx_hat_dgx / (2 * np.sqrt(vx_hat) + 1e-30)

    # Quotient rule: d(mx_hat/denom)/dgx
    #   = (dm_dgx * denom - mx_hat * dsqrt_vx_dgx) / denom²
    grad = -ADAM_LR * (dm_dgx * denom - mx_hat * dsqrt_vx_dgx) / denom2

    return grad


# Verify against numerical gradient
print("\n--- Step 2: dx_new/d(gx_clean) ---")

sig_x, sig_y, mx, my, vx, vy, x, y, t = make_test_case()
weight_vec = make_scale_weight_vector(np.array([0.6, 0.7, 0.5]), BUFFER_CAPACITY)
gx_clean = haar_inverse(haar_forward(sig_x) * weight_vec)[-1]

# Analytical
anal = dx_new_dgx_clean(gx_clean, mx, vx, t)

# Numerical
eps = 1e-5
def x_new_from_gx(gc):
    m  = BETA1*mx + (1-BETA1)*gc
    v  = BETA2*vx + (1-BETA2)*gc**2
    mh = m / (1-BETA1**t)
    vh = v / (1-BETA2**t)
    return x - ADAM_LR * mh / (np.sqrt(vh) + EPS)

num = (x_new_from_gx(gx_clean + eps) - x_new_from_gx(gx_clean - eps)) / (2*eps)

diff = abs(anal - num)
print(f"Analytical: {anal:.8f}")
print(f"Numerical:  {num:.8f}")
print(f"Difference: {diff:.2e}")
print(f"Step 2 passed: {diff < 1e-6}")

# =============================================================
# STEP 3: Full chain — dL/d(scale_weights)
#
# Assembling all pieces via chain rule:
#
#   dL/d(scale_weights[k])
#     = dL/d(loss_new)                          = 1 (loss_new is the loss)
#     * d(loss_new)/d(x_new)                    = gx at (x_new, y_new)
#     * d(x_new)/d(gx_clean)                    = Step 2
#     * d(gx_clean)/d(weight_vec)               = Step 1
#     * d(weight_vec)/d(scale_weights[k])       = grouping by scale
#
# The last term: weight_vec elements in scale group k all equal
# sigmoid(scale_weights[k]), so their derivative w.r.t.
# scale_weights[k] is sigmoid_grad(scale_weights[k]).
#
# We sum over all elements in the group because each one
# contributes independently to gx_clean through weight_vec.
# =============================================================

def analytical_meta_gradient(sig_x, sig_y, mx, my, vx, vy,
                               x, y, t, scale_weights):
    """
    Full analytical gradient of loss_new w.r.t. scale_weights.

    Returns grad of shape (3,) — one entry per scale group.
    """
    import math
    J = int(math.log2(BUFFER_CAPACITY))

    weight_vec = make_scale_weight_vector(scale_weights, BUFFER_CAPACITY)

    # Forward: compute gx_clean, gy_clean
    gx_clean = haar_inverse(haar_forward(sig_x) * weight_vec)[-1]
    gy_clean = haar_inverse(haar_forward(sig_y) * weight_vec)[-1]

    # Adam step
    mxn = BETA1*mx + (1-BETA1)*gx_clean
    myn = BETA1*my + (1-BETA1)*gy_clean
    vxn = BETA2*vx + (1-BETA2)*gx_clean**2
    vyn = BETA2*vy + (1-BETA2)*gy_clean**2
    a, b = 1-BETA1**t, 1-BETA2**t
    mxh, myh = mxn/a, myn/a
    vxh, vyh = vxn/b, vyn/b
    x_new = x - ADAM_LR * mxh / (np.sqrt(vxh) + EPS)
    y_new = y - ADAM_LR * myh / (np.sqrt(vyh) + EPS)

    # d(loss)/d(x_new), d(loss)/d(y_new)
    loss_new, dloss_dx, dloss_dy = rosenbrock(x_new, y_new)

    # d(x_new)/d(gx_clean), d(y_new)/d(gy_clean)
    dx_dgx = dx_new_dgx_clean(gx_clean, mx, vx, t)
    dy_dgy = dx_new_dgx_clean(gy_clean, my, vy, t)

    # d(gx_clean)/d(weight_vec), d(gy_clean)/d(weight_vec)
    dgx_dwv = dgx_clean_dweightvec_fast(sig_x)
    dgy_dwv = dgx_clean_dweightvec_fast(sig_y)

    # Chain rule up to weight_vec
    # dL/d(weight_vec[i]) for x and y contributions
    dL_dwv = (dloss_dx * dx_dgx * dgx_dwv +
              dloss_dy * dy_dgy * dgy_dwv)

    # Sum over scale groups, multiply by sigmoid_grad
    from core.meta_net import sigmoid_grad
    grad = np.zeros(J)
    for k in range(J):
        start = 2**k
        end   = 2**(k+1)
        grad[k] = np.sum(dL_dwv[start:end]) * sigmoid_grad(scale_weights[k])

    return grad, loss_new


# Verify full chain against numerical gradient
print("\n--- Step 3: Full chain dL/d(scale_weights) ---")

sig_x, sig_y, mx, my, vx, vy, x, y, t = make_test_case()
scale_weights = np.array([0.6, 0.7, 0.5])

# Analytical
anal_grad, loss_new = analytical_meta_gradient(
    sig_x, sig_y, mx, my, vx, vy, x, y, t, scale_weights
)

# Numerical
eps = 1e-5
num_grad = np.zeros(3)
for k in range(3):
    sw_plus  = scale_weights.copy(); sw_plus[k]  += eps
    sw_minus = scale_weights.copy(); sw_minus[k] -= eps

    def loss_from_sw(sw):
        wv = make_scale_weight_vector(sw, BUFFER_CAPACITY)
        gxc = haar_inverse(haar_forward(sig_x) * wv)[-1]
        gyc = haar_inverse(haar_forward(sig_y) * wv)[-1]
        mxn = BETA1*mx + (1-BETA1)*gxc
        myn = BETA1*my + (1-BETA1)*gyc
        vxn = BETA2*vx + (1-BETA2)*gxc**2
        vyn = BETA2*vy + (1-BETA2)*gyc**2
        mxh = mxn/(1-BETA1**t); myh = myn/(1-BETA1**t)
        vxh = vxn/(1-BETA2**t); vyh = vyn/(1-BETA2**t)
        xn = x - ADAM_LR*mxh/(np.sqrt(vxh)+EPS)
        yn = y - ADAM_LR*myh/(np.sqrt(vyh)+EPS)
        l, _, _ = rosenbrock(xn, yn)
        return l

    num_grad[k] = (loss_from_sw(sw_plus) - loss_from_sw(sw_minus)) / (2*eps)

max_diff = np.max(np.abs(anal_grad - num_grad))
print(f"Analytical: {np.round(anal_grad, 8)}")
print(f"Numerical:  {np.round(num_grad, 8)}")
print(f"Max difference: {max_diff:.2e}")
print(f"Step 3 passed: {max_diff < 1e-5}")



# =============================================================
# STEP 4: MetaWaveletAdam with analytical gradients
#
# Replace the 6-forward-pass numerical gradient with the
# analytical gradient we just derived and verified.
#
# Also add:
#   - Windowed meta-loss (avg over last K steps)
#   - Adam for MetaNet updates (not SGD)
#   - Gradient clipping on meta-gradients
# =============================================================

from core.meta_net import MetaNet, sigmoid
from core.optimizers import run_adam


def compute_state_features(t, loss, gx, gy, x, y):
    """Training state features — no n_steps dependency."""
    log_step  = np.log(t + 1) / 10.0
    log_loss  = np.log(loss + 1e-8) / 10.0
    grad_norm = np.log(np.sqrt(gx**2 + gy**2) + 1e-8) / 5.0
    dist_norm = np.log(np.sqrt((x-1)**2 + (y-1)**2) + 1e-8) / 5.0
    return np.array([log_step, log_loss, grad_norm, dist_norm])


class AdamOptimizer:
    """
    Adam optimizer for the MetaNet parameters.
    Separate from the main problem optimizer — this one
    adapts the meta-network weights.
    """
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads, clip=1.0):
        self.t += 1
        for k in self.params:
            g = grads[k]
            # Clip
            gnorm = np.linalg.norm(g)
            if gnorm > clip:
                g = g * (clip / gnorm)
            self.m[k] = self.beta1*self.m[k] + (1-self.beta1)*g
            self.v[k] = self.beta2*self.v[k] + (1-self.beta2)*g**2
            mh = self.m[k] / (1-self.beta1**self.t)
            vh = self.v[k] / (1-self.beta2**self.t)
            self.params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def run_analytical_meta_adam(n_steps=2000, buffer_capacity=8,
                              meta_lr=0.001, adam_lr=0.01,
                              window_size=20, start=(-1.0, 1.0),
                              beta1=0.9, beta2=0.999, eps=1e-8):
    """
    MetaWaveletAdam with:
      - Analytical meta-gradients (no numerical perturbation)
      - Windowed meta-loss over last window_size steps
      - Adam for MetaNet updates
    """
    x, y = start
    history = []
    loss_window = deque(maxlen=window_size*2)  # store last 2 windows for meta-loss

    buf_x = CircularGradientBuffer(capacity=buffer_capacity, grad_shape=(1,))
    buf_y = CircularGradientBuffer(capacity=buffer_capacity, grad_shape=(1,))

    # Main problem moments
    mx, my = 0.0, 0.0
    vx, vy = 0.0, 0.0

    # MetaNet + its Adam optimizer
    net      = MetaNet(input_dim=4, hidden_dim=8, output_dim=3)
    meta_opt = AdamOptimizer(net.params, lr=meta_lr)

    for t in range(1, n_steps + 1):
        loss, gx, gy = rosenbrock(x, y)
        history.append((x, y, loss))
        loss_window.append(loss)
        if len(loss_window) > window_size * 2:
            loss_window.pop(0)

        buf_x.write(np.array([gx]))
        buf_y.write(np.array([gy]))

        if buf_x.is_full():
            # --- MetaNet forward ---
            state = compute_state_features(t, loss, gx, gy, x, y)
            scale_weights = net.forward(state)

            weight_vec = make_scale_weight_vector(scale_weights, buffer_capacity)
            sig_x = buf_x.read().flatten()
            sig_y = buf_y.read().flatten()

            gx_clean = haar_inverse(haar_forward(sig_x) * weight_vec)[-1]
            gy_clean = haar_inverse(haar_forward(sig_y) * weight_vec)[-1]

        else:
            gx_clean, gy_clean = gx, gy

        # --- Main Adam update ---
        mx = beta1*mx + (1-beta1)*gx_clean
        my = beta1*my + (1-beta1)*gy_clean
        vx = beta2*vx + (1-beta2)*gx_clean**2
        vy = beta2*vy + (1-beta2)*gy_clean**2
        mxh = mx/(1-beta1**t); myh = my/(1-beta1**t)
        vxh = vx/(1-beta2**t); vyh = vy/(1-beta2**t)
        x = x - adam_lr*mxh/(np.sqrt(vxh)+eps)
        y = y - adam_lr*myh/(np.sqrt(vyh)+eps)

        # --- MetaNet backward (only when buffer full and window populated) ---
        if buf_x.is_full() and len(loss_window) >= window_size * 2:
            # Windowed meta-loss: is recent loss lower than earlier loss?
            recent   = np.mean(list(loss_window)[-window_size:])
            earlier  = np.mean(list(loss_window)[:window_size])
            # Normalise by earlier loss
            meta_loss_signal = (recent - earlier) / (earlier + 1e-8)
            # Clip to avoid extreme values
            meta_loss_signal = np.clip(meta_loss_signal, -1.0, 1.0)

            # Analytical gradient of loss w.r.t. scale_weights
            # We use current state as a proxy for the window
            grad_sw, _ = analytical_meta_gradient(
                sig_x, sig_y, mx, my, vx, vy, x, y, t, scale_weights
            )

            # Scale by windowed signal — if improving, reinforce current weights
            # if worsening, push in the gradient direction
            grad_sw = grad_sw * meta_loss_signal

            # Backprop through MetaNet
            net.zero_grad()
            dL_dweights = grad_sw
            net.backward(dL_dweights)

            # Build grads dict for AdamOptimizer
            grads_dict = net.grads
            meta_opt.step(grads_dict)

    return history


# --- Benchmark ---
print("\n--- Session 6 benchmark ---")

h_adam = run_adam(lr=0.01, n_steps=2000)
converged_adam = next((i for i,(x,y,l) in enumerate(h_adam)
                       if abs(x-1)<1e-4 and abs(y-1)<1e-4), None)
print(f"Adam                  converged={converged_adam}  "
      f"loss={h_adam[-1][2]:.6f}")

h_analytical = run_analytical_meta_adam(n_steps=5000)
converged_anal = next((i for i,(x,y,l) in enumerate(h_analytical)
                       if abs(x-1)<1e-4 and abs(y-1)<1e-4), None)
print(f"AnalyticalMetaAdam    converged={converged_anal}  "
      f"loss={h_analytical[-1][2]:.6f}")

# Held-out
print("\n--- Held-out start (-0.5, 0.5) ---")
h_adam_h = run_adam(lr=0.01, n_steps=2000, start=(-0.5, 0.5))
converged_ah = next((i for i,(x,y,l) in enumerate(h_adam_h)
                     if abs(x-1)<1e-4 and abs(y-1)<1e-4), None)
print(f"Adam               converged={converged_ah}  "
      f"loss={h_adam_h[-1][2]:.6f}")

h_anal_h = run_analytical_meta_adam(n_steps=2000, start=(-0.5, 0.5))
converged_anh = next((i for i,(x,y,l) in enumerate(h_anal_h)
                      if abs(x-1)<1e-4 and abs(y-1)<1e-4), None)
print(f"AnalyticalMetaAdam converged={converged_anh}  "
      f"loss={h_anal_h[-1][2]:.6f}")

# Speed comparison — clean version
print("\n--- Speed: analytical vs numerical ---")
from core.buffers import CircularGradientBuffer
from core.wavelets import haar_forward, haar_inverse, make_scale_weight_vector

def numerical_meta_grad_cost(n_steps=500):
    """Measure cost of numerical gradient approach."""
    x, y = -1.0, 1.0
    mx, my, vx, vy = 0.0, 0.0, 0.0, 0.0
    buf_x = CircularGradientBuffer(capacity=8, grad_shape=(1,))
    buf_y = CircularGradientBuffer(capacity=8, grad_shape=(1,))
    scale_weights = np.array([0.5, 0.5, 0.5])

    for t in range(1, n_steps+1):
        loss, gx, gy = rosenbrock(x, y)
        buf_x.write(np.array([gx]))
        buf_y.write(np.array([gy]))

        if buf_x.is_full():
            sig_x = buf_x.read().flatten()
            sig_y = buf_y.read().flatten()
            eps_n = 1e-5
            for k in range(3):
                for sign in [1, -1]:
                    sw = scale_weights.copy()
                    sw[k] += sign * eps_n
                    wv = make_scale_weight_vector(sw, 8)
                    gxc = haar_inverse(haar_forward(sig_x) * wv)[-1]
                    gyc = haar_inverse(haar_forward(sig_y) * wv)[-1]
                    mxn = 0.9*mx + 0.1*gxc
                    vxn = 0.999*vx + 0.001*gxc**2
                    mxh = mxn/(1-0.9**t)
                    vxh = vxn/(1-0.999**t)
                    xn = x - 0.01*mxh/(np.sqrt(vxh)+1e-8)
                    rosenbrock(xn, y)

        mx = 0.9*mx + 0.1*gx
        my = 0.9*my + 0.1*gy
        vx = 0.999*vx + 0.001*gx**2
        vy = 0.999*vy + 0.001*gy**2
        mxh = mx/(1-0.9**t); myh = my/(1-0.9**t)
        vxh = vx/(1-0.999**t); vyh = vy/(1-0.999**t)
        x = x - 0.01*mxh/(np.sqrt(vxh)+1e-8)
        y = y - 0.01*myh/(np.sqrt(vyh)+1e-8)

start_time = time.perf_counter()
run_analytical_meta_adam(n_steps=500)
anal_time = time.perf_counter() - start_time

start_time = time.perf_counter()
numerical_meta_grad_cost(n_steps=500)
num_time = time.perf_counter() - start_time

print(f"Analytical: {anal_time*1000:.1f}ms")
print(f"Numerical:  {num_time*1000:.1f}ms")
print(f"Speedup:    {num_time/anal_time:.1f}x")