import numpy as np
import math
from collections import deque
from problems.rosenbrock import rosenbrock
from problems.noisy_rosenbrock import noisy_rosenbrock
from core.buffers import CircularGradientBuffer
from core.wavelets import (
    haar_forward,
    haar_inverse,
    make_scale_weight_vector,
    dgx_clean_dweightvec_fast,
    get_idwt_last_row,
    soft_wavelet_denoise,
)
from core.meta_net import MetaNet, sigmoid, sigmoid_grad
# STEP 2: SGD

def run_sgd(lr, n_steps, start=(-1.0, 1.0)):
    x, y = start
    history = []
    for i in range(n_steps):
        loss, gx, gy = rosenbrock(x, y)
        history.append((x, y, loss))
        
        x -= lr * gx
        y -= lr * gy
        
    return history


# STEP 3: ADAM

def run_adam(lr, n_steps, start=(-1.0, 1.0),
             beta1=0.9, beta2=0.999, eps=1e-8):
    x, y = start
    history = []

    # Moment estimates — initialized to zero
    mx, my = 0.0, 0.0   # 1st moments
    vx, vy = 0.0, 0.0   # 2nd moments

    for t in range(1, n_steps + 1):   # t starts at 1 (needed for bias correction)
        loss, gx, gy = rosenbrock(x, y)
        history.append((x, y, loss))

        # --- 1st moment update ---
        mx = beta1 * mx + (1 - beta1) * gx
        my = beta1 * my + (1 - beta1) * gy

        # --- 2nd moment update ---
        vx = beta2 * vx + (1 - beta2) * gx**2
        vy = beta2 * vy + (1 - beta2) * gy**2

        # --- Bias correction ---
        mx_hat = mx / (1 - beta1**t)
        my_hat = my / (1 - beta1**t)
        vx_hat = vx / (1 - beta2**t)
        vy_hat = vy / (1 - beta2**t)

        # --- Update ---
        x = x - lr * mx_hat / (vx_hat**0.5 + eps)
        y = y - lr * my_hat / (vy_hat**0.5 + eps)

    return history

# Adam Instrumented - Basically Adam with a Circular Gradient Buffer
def run_adam_instrumented(lr, n_steps, buffer_capacity=8, 
                          start=(-1.0, 1.0), 
                          beta1=0.9, beta2=0.999, eps=1e-8):
    '''
    Adam optimizer with circular gradient buffer.
    
    Basically uses the same update rules as Adam, no changes mathematically
    Just recording the gradient signal so we can analyze it.
    
    Returns:
    history: list of (x, y, loss) per step
    grad_log: array of shape (n_steps, 2) - raw gx, gy per step
    buffer_snapshots: list of buffer.read() arrays, one per step
                        (only populated once buffer is full)

    '''
    
    
    x, y = start
    history = []
    grad_log = []
    buffer_snapshots = []
    
    # One buffer per parameter - each tracks that parameter's gradient over time
    buf_x = CircularGradientBuffer(capacity=buffer_capacity, grad_shape=(1,))
    buf_y = CircularGradientBuffer(capacity=buffer_capacity, grad_shape=(1,))
    
    mx, my = 0.0, 0.0   # 1st moments
    vx, vy = 0.0, 0.0   # 2nd moments
    
    for t in range(1, n_steps + 1):
        loss, gx, gy = rosenbrock(x, y)
        history.append((x, y, loss))
        grad_log.append((gx, gy))
        
        # Write raw gradients to buffers
        buf_x.write(np.array([gx]))
        buf_y.write(np.array([gy]))
        
        # Once buffer is full, snapshot its contents for analysis
        if buf_x.is_full():
           snapshot_x = buf_x.read().flatten()  # shape (capacity,)
           snapshot_y = buf_y.read().flatten()  # shape (capacity,)
           buffer_snapshots.append((snapshot_x, snapshot_y))
        else:
            buffer_snapshots.append(None)

        # --- Adam updates ---
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

    return history, np.array(grad_log), buffer_snapshots

# WaveletSGD
def run_wavelet_sgd(lr, n_steps, buffer_capacity=8,
                    threshold=1.0, start=(-1.0, 1.0)):
    """
    WaveletSGD — first version of our algorithm.

    Idea:
        Instead of using the raw gradient at each step,
        use a wavelet-denoised gradient reconstructed from
        the last `buffer_capacity` steps.

    For the first `buffer_capacity` steps (before buffer is full),
    fall back to raw gradient — we don't have enough history yet.

    This is deliberately simple. No momentum, no adaptive scaling.
    We want to isolate the effect of wavelet denoising alone.
    """
    
    x, y = start
    history = []

    buf_x = CircularGradientBuffer(capacity=buffer_capacity, grad_shape=(1,))
    buf_y = CircularGradientBuffer(capacity=buffer_capacity, grad_shape=(1,))

    for step in range(n_steps):
        loss, gx, gy = rosenbrock(x, y)
        history.append((x, y, loss))

        buf_x.write(np.array([gx]))
        buf_y.write(np.array([gy]))

        if buf_x.is_full():
            # --- Wavelet denoising ---
            sig_x = buf_x.read().flatten()
            sig_y = buf_y.read().flatten()

            # Forward
            coeffs_x = haar_forward(sig_x)
            coeffs_y = haar_forward(sig_y)

            # Threshold detail coefficients
            coeffs_x[1:] = np.where(np.abs(coeffs_x[1:]) > threshold,
                                     coeffs_x[1:], 0.0)
            coeffs_y[1:] = np.where(np.abs(coeffs_y[1:]) > threshold,
                                     coeffs_y[1:], 0.0)

            # Inverse — take last value as denoised gradient
            gx_clean = haar_inverse(coeffs_x)[-1]
            gy_clean = haar_inverse(coeffs_y)[-1]

        else:
            # Buffer not full yet — use raw gradient
            gx_clean = gx
            gy_clean = gy

        # SGD update with denoised gradient
        x = x - lr * gx_clean
        y = y - lr * gy_clean

    return history

def run_wavelet_adam(lr, n_steps, buffer_capacity=8,
                     threshold=1.0, start=(-1.0, 1.0),
                     beta1=0.9, beta2=0.999, eps=1e-8):
    """
    WaveletAdam — wavelet denoising + Adam adaptive scaling.

    Difference from run_adam:
        Before the Adam moment updates, we replace the raw gradient
        with a wavelet-denoised version.

    Everything else is identical to Adam.
    """
    
    x, y = start
    history = []

    buf_x = CircularGradientBuffer(capacity=buffer_capacity, grad_shape=(1,))
    buf_y = CircularGradientBuffer(capacity=buffer_capacity, grad_shape=(1,))

    mx, my = 0.0, 0.0
    vx, vy = 0.0, 0.0

    for t in range(1, n_steps + 1):
        loss, gx, gy = rosenbrock(x, y)
        history.append((x, y, loss))

        buf_x.write(np.array([gx]))
        buf_y.write(np.array([gy]))

        if buf_x.is_full():
            # Denoise gradient before feeding into Adam
            sig_x = buf_x.read().flatten()
            sig_y = buf_y.read().flatten()

            cx = haar_forward(sig_x)
            cy = haar_forward(sig_y)

            cx[1:] = np.where(np.abs(cx[1:]) > threshold, cx[1:], 0.0)
            cy[1:] = np.where(np.abs(cy[1:]) > threshold, cy[1:], 0.0)

            gx = haar_inverse(cx)[-1]
            gy = haar_inverse(cy)[-1]
            # gx, gy now carry denoised signal into Adam moments below

        # Adam update (identical structure — just using denoised gx, gy)
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

    return history

def run_learned_wavelet_adam(lr, n_steps, raw_params_x, raw_params_y,
                              buffer_capacity=8, start=(-1.0, 1.0),
                              beta1=0.9, beta2=0.999, eps=1e-8):
    """
    WaveletAdam with learned per-scale soft weights.
    Same structure as run_wavelet_adam but uses soft_wavelet_denoise
    instead of hard thresholding.
    """
    
    x, y = start
    history = []

    buf_x = CircularGradientBuffer(capacity=buffer_capacity, grad_shape=(1,))
    buf_y = CircularGradientBuffer(capacity=buffer_capacity, grad_shape=(1,))

    mx, my = 0.0, 0.0
    vx, vy = 0.0, 0.0

    for t in range(1, n_steps + 1):
        loss, gx, gy = rosenbrock(x, y)
        history.append((x, y, loss))

        buf_x.write(np.array([gx]))
        buf_y.write(np.array([gy]))

        if buf_x.is_full():
            sig_x = buf_x.read().flatten()
            sig_y = buf_y.read().flatten()

            gx, _, _, _ = soft_wavelet_denoise(sig_x, raw_params_x)
            gy, _, _, _ = soft_wavelet_denoise(sig_y, raw_params_y)

        # Adam update with denoised gradients
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

    return history

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



def compute_state_features(t, loss, gx, gy, x, y,
                            buf_x=None, buf_y=None):
    """
    Training state features for MetaNet input.

    5 features when buffers provided, 4 without (backward compatible).
    """
    log_step  = np.log(t + 1) / 10.0
    log_loss  = np.log(loss + 1e-8) / 10.0
    grad_norm = np.log(np.sqrt(gx**2 + gy**2) + 1e-8) / 5.0
    dist_norm = np.log(np.sqrt((x-1)**2 + (y-1)**2) + 1e-8) / 5.0

    features = [log_step, log_loss, grad_norm, dist_norm]

    if buf_x is not None and buf_y is not None:
        # Gradient variance — proxy for noise level
        # High variance = noisy or unstable gradients
        var_x = np.var(buf_x.read().flatten())
        var_y = np.var(buf_y.read().flatten())
        grad_var = np.log(0.5*(var_x + var_y) + 1e-8) / 5.0
        features.append(grad_var)

    return np.array(features)

# Adam optimizer for MetaNet parameters.
class AdamOptimizer:
    """Adam optimizer for MetaNet parameters."""
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
            gnorm = np.linalg.norm(g)
            if gnorm > clip:
                g = g * (clip / gnorm)
            self.m[k] = self.beta1*self.m[k] + (1-self.beta1)*g
            self.v[k] = self.beta2*self.v[k] + (1-self.beta2)*g**2
            mh = self.m[k] / (1-self.beta1**self.t)
            vh = self.v[k] / (1-self.beta2**self.t)
            self.params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def analytical_meta_gradient(sig_x, sig_y, mx, my, vx, vy,
                               x, y, t, scale_weights,
                               buffer_capacity=8,
                               beta1=0.9, beta2=0.999,
                               eps=1e-8, adam_lr=0.01):
    """Full analytical gradient of loss_new w.r.t. scale_weights."""
    import math
    J = int(math.log2(buffer_capacity))

    weight_vec = make_scale_weight_vector(scale_weights, buffer_capacity)
    gx_clean = haar_inverse(haar_forward(sig_x) * weight_vec)[-1]
    gy_clean = haar_inverse(haar_forward(sig_y) * weight_vec)[-1]

    mxn = beta1*mx + (1-beta1)*gx_clean
    myn = beta1*my + (1-beta1)*gy_clean
    vxn = beta2*vx + (1-beta2)*gx_clean**2
    vyn = beta2*vy + (1-beta2)*gy_clean**2
    a, b = 1-beta1**t, 1-beta2**t
    mxh = mxn/a; myh = myn/a
    vxh = vxn/b; vyh = vyn/b
    x_new = x - adam_lr*mxh/(np.sqrt(vxh)+eps)
    y_new = y - adam_lr*myh/(np.sqrt(vyh)+eps)

    loss_new, dloss_dx, dloss_dy = rosenbrock(x_new, y_new)

    def dx_dgx(gc, mx_o, vx_o):
        a_ = 1-beta1**t; b_ = 1-beta2**t
        m_ = beta1*mx_o + (1-beta1)*gc
        v_ = beta2*vx_o + (1-beta2)*gc**2
        mh_ = m_/a_; vh_ = v_/b_
        denom = np.sqrt(vh_) + eps
        dm = (1-beta1)/a_
        dv = (1-beta2)*2*gc/b_
        ds = dv/(2*np.sqrt(vh_)+1e-30)
        return -adam_lr*(dm*denom - mh_*ds)/denom**2

    dx_dgxc = dx_dgx(gx_clean, mx, vx)
    dy_dgyc = dx_dgx(gy_clean, my, vy)

    dgx_dwv = dgx_clean_dweightvec_fast(sig_x)
    dgy_dwv = dgx_clean_dweightvec_fast(sig_y)

    dL_dwv = (dloss_dx*dx_dgxc*dgx_dwv +
              dloss_dy*dy_dgyc*dgy_dwv)

    grad = np.zeros(J)
    for k in range(J):
        start = 2**k; end = 2**(k+1)
        grad[k] = np.sum(dL_dwv[start:end]) * sigmoid_grad(scale_weights[k])

    return grad, loss_new

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



if __name__ == "__main__":
    history_more = run_sgd(lr=0.0001, n_steps=5000)
    last = history_more[-1]
    print(f"\nSGD  lr=0.001  50000 steps")
    print(f"  End: x={last[0]:.4f}  y={last[1]:.4f}  loss={last[2]:.6f}")

