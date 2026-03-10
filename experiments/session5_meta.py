import numpy as np
import sys 
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problems.rosenbrock import rosenbrock
from core.buffers import CircularGradientBuffer
from core.wavelets import haar_forward, haar_inverse, make_scale_weight_vector
from core.meta_net import MetaNet

# ===========================================
# SESSION 5: ONLINE META-LEARNING

# The meta network updates every step — not in a separate
# training phase. At each step:
#
#   1. Compute training state features
#   2. Forward pass through MetaNet → scale weights
#   3. Denoise gradient using those weights
#   4. Adam update using denoised gradient
#   5. Compute meta-loss: did denoising help?
#   6. Backward pass through MetaNet → update network weights
#
# The meta-loss measures how good the denoised gradient was:
#   meta_loss = (loss_after_step - loss_before_step)
#
# A good denoised gradient decreases the loss more than the
# raw gradient would. We want to minimize meta_loss.
# =============================================================

def compute_state_features(t, n_steps, loss, gx, gy, x, y):
    """
    Encode current training state as a 4-vector.

    Features are normalized to roughly [-1, 1] so the network
    doesn't have to deal with wildly different scales.
    """
    log_step  = np.log(t + 1) / 10.0                        
    log_loss   = np.log(loss + 1e-8) / 10.0          # roughly [-1, 0]
    grad_norm  = np.log(np.sqrt(gx**2 + gy**2) + 1e-8) / 5.0  # normalized
    dist_norm  = np.log(np.sqrt((x-1)**2 + (y-1)**2) + 1e-8) / 5.0

    return np.array([log_step, log_loss, grad_norm, dist_norm])


def run_meta_wavelet_adam(n_steps=2000, buffer_capacity=8,
                          meta_lr=0.001, adam_lr=0.01,
                          start=(-1.0, 1.0),
                          beta1=0.9, beta2=0.999, eps=1e-8):
    """
    WaveletAdam with online MetaNet updates.
    """
    x, y = start
    history = []

    buf_x = CircularGradientBuffer(capacity=buffer_capacity, grad_shape=(1,))
    buf_y = CircularGradientBuffer(capacity=buffer_capacity, grad_shape=(1,))

    mx, my = 0.0, 0.0
    vx, vy = 0.0, 0.0

    net = MetaNet(input_dim=4, hidden_dim=8, output_dim=3)

    meta_losses = []

    for t in range(1, n_steps + 1):
        loss, gx, gy = rosenbrock(x, y)
        history.append((x, y, loss))

        buf_x.write(np.array([gx]))
        buf_y.write(np.array([gy]))

        if buf_x.is_full():
            # --- Step 1: get scale weights from MetaNet ---
            state = compute_state_features(t, n_steps, loss, gx, gy, x, y)
            scale_weights = net.forward(state)   # shape (3,)

            # --- Step 2: build full weight vector and denoise ---
            weight_vec = make_scale_weight_vector(scale_weights, buffer_capacity)

            sig_x = buf_x.read().flatten()
            sig_y = buf_y.read().flatten()

            cx = haar_forward(sig_x) * weight_vec
            cy = haar_forward(sig_y) * weight_vec

            gx_clean = haar_inverse(cx)[-1]
            gy_clean = haar_inverse(cy)[-1]

        else:
            # Buffer not full — use raw gradient, no meta update
            gx_clean = gx
            gy_clean = gy
            meta_losses.append(None)

            # Adam update and continue
            mx = beta1 * mx + (1 - beta1) * gx_clean
            my = beta1 * my + (1 - beta1) * gy_clean
            vx = beta2 * vx + (1 - beta2) * gx_clean**2
            vy = beta2 * vy + (1 - beta2) * gy_clean**2
            mx_hat = mx / (1 - beta1**t)
            my_hat = my / (1 - beta1**t)
            vx_hat = vx / (1 - beta2**t)
            vy_hat = vy / (1 - beta2**t)
            x = x - adam_lr * mx_hat / (vx_hat**0.5 + eps)
            y = y - adam_lr * my_hat / (vy_hat**0.5 + eps)
            continue

        # --- Step 3: Adam update with denoised gradient ---
        mx = beta1 * mx + (1 - beta1) * gx_clean
        my = beta1 * my + (1 - beta1) * gy_clean
        vx = beta2 * vx + (1 - beta2) * gx_clean**2
        vy = beta2 * vy + (1 - beta2) * gy_clean**2

        mx_hat = mx / (1 - beta1**t)
        my_hat = my / (1 - beta1**t)
        vx_hat = vx / (1 - beta2**t)
        vy_hat = vy / (1 - beta2**t)

        x_new = x - adam_lr * mx_hat / (vx_hat**0.5 + eps)
        y_new = y - adam_lr * my_hat / (vy_hat**0.5 + eps)

        # --- Step 4: meta-loss and backward ---
        # Baseline: what would loss be without denoising?
        mx_raw = beta1 * mx + (1 - beta1) * gx
        my_raw = beta1 * my + (1 - beta1) * gy
        vx_raw = beta2 * vx + (1 - beta2) * gx**2
        vy_raw = beta2 * vy + (1 - beta2) * gy**2
        mxr_hat = mx_raw / (1 - beta1**t)
        myr_hat = my_raw / (1 - beta1**t)
        vxr_hat = vx_raw / (1 - beta2**t)
        vyr_hat = vy_raw / (1 - beta2**t)
        x_raw = x - adam_lr * mxr_hat / (vxr_hat**0.5 + eps)
        y_raw = y - adam_lr * myr_hat / (vyr_hat**0.5 + eps)
        loss_baseline, _, _ = rosenbrock(x_raw, y_raw)
        
        # Meta-loss: how much better/worse than raw Adam
        # Negative = denoising helped, Positive = denoising hurt

        loss_new, _, _ = rosenbrock(x_new, y_new)
        meta_loss = loss_new - loss_baseline

        meta_losses.append(meta_loss)

        # Gradient of meta_loss w.r.t. scale_weights
        # We use numerical gradient here — analytical through
        # Adam update is complex (Session 6 topic)
        dL_dweights = np.zeros(3)
        eps_meta = 1e-5
        for k in range(3):
            sw_plus  = scale_weights.copy(); sw_plus[k]  += eps_meta
            sw_minus = scale_weights.copy(); sw_minus[k] -= eps_meta

            wv_plus  = make_scale_weight_vector(sw_plus,  buffer_capacity)
            wv_minus = make_scale_weight_vector(sw_minus, buffer_capacity)

            # Recompute denoised gradients with perturbed weights
            gxp = haar_inverse(haar_forward(sig_x) * wv_plus)[-1]
            gyp = haar_inverse(haar_forward(sig_y) * wv_plus)[-1]
            gxm = haar_inverse(haar_forward(sig_x) * wv_minus)[-1]
            gym = haar_inverse(haar_forward(sig_y) * wv_minus)[-1]

            # Recompute Adam step with perturbed gradients
            mxp = beta1*mx + (1-beta1)*gxp
            myp = beta1*my + (1-beta1)*gyp
            vxp = beta2*vx + (1-beta2)*gxp**2
            vyp = beta2*vy + (1-beta2)*gyp**2
            mxp_hat = mxp / (1-beta1**t)
            myp_hat = myp / (1-beta1**t)
            vxp_hat = vxp / (1-beta2**t)
            vyp_hat = vyp / (1-beta2**t)
            xp = x - adam_lr * mxp_hat / (vxp_hat**0.5 + eps)
            yp = y - adam_lr * myp_hat / (vyp_hat**0.5 + eps)

            mxm = beta1*mx + (1-beta1)*gxm
            mym = beta1*my + (1-beta1)*gym
            vxm = beta2*vx + (1-beta2)*gxm**2
            vym = beta2*vy + (1-beta2)*gym**2
            mxm_hat = mxm / (1-beta1**t)
            mym_hat = mym / (1-beta1**t)
            vxm_hat = vxm / (1-beta2**t)
            vym_hat = vym / (1-beta2**t)
            xm = x - adam_lr * mxm_hat / (vxm_hat**0.5 + eps)
            ym = y - adam_lr * mym_hat / (vym_hat**0.5 + eps)

            loss_plus,  _, _ = rosenbrock(xp, yp)
            loss_minus, _, _ = rosenbrock(xm, ym)
            dL_dweights[k] = (loss_plus - loss_minus) / (2 * eps_meta)

        # Backward through MetaNet
        net.zero_grad()
        net.backward(dL_dweights)
        net.update(meta_lr)

        x, y = x_new, y_new

    return history, meta_losses


# --- Run and benchmark ---
from core.optimizers import run_adam

print("Running benchmark...")

h_adam = run_adam(lr=0.01, n_steps=2000)
converged_adam = next((i for i,(x,y,l) in enumerate(h_adam)
                       if abs(x-1)<1e-4 and abs(y-1)<1e-4), None)
last_adam = h_adam[-1]

h_meta, meta_losses = run_meta_wavelet_adam(n_steps=2000, meta_lr=0.001)
converged_meta = next((i for i,(x,y,l) in enumerate(h_meta)
                       if abs(x-1)<1e-4 and abs(y-1)<1e-4), None)
last_meta = h_meta[-1]

print(f"\nAdam              loss={last_adam[2]:.6f}  converged={converged_adam}")
print(f"MetaWaveletAdam   loss={last_meta[2]:.6f}  converged={converged_meta}")

# Show how meta_loss evolves — is the network learning?
valid_meta = [l for l in meta_losses if l is not None]
n = len(valid_meta)
print(f"\nMeta-loss (post-step loss):")
print(f"  First 100 steps avg:  {np.mean(valid_meta[:100]):.6f}")
print(f"  Middle 100 steps avg: {np.mean(valid_meta[n//2-50:n//2+50]):.6f}")
print(f"  Last 100 steps avg:   {np.mean(valid_meta[-100:]):.6f}")

# Held-out evaluation
print("\n--- Held-out start (-0.5, 0.5) ---")
h_adam_h = run_adam(lr=0.01, n_steps=2000, start=(-0.5, 0.5))
converged_ah = next((i for i,(x,y,l) in enumerate(h_adam_h)
                     if abs(x-1)<1e-4 and abs(y-1)<1e-4), None)
last_ah = h_adam_h[-1]

h_meta_h, _ = run_meta_wavelet_adam(n_steps=2000, start=(-0.5, 0.5))
converged_mh = next((i for i,(x,y,l) in enumerate(h_meta_h)
                     if abs(x-1)<1e-4 and abs(y-1)<1e-4), None)
last_mh = h_meta_h[-1]

print(f"Adam            loss={last_ah[2]:.6f}  converged={converged_ah}")
print(f"MetaWaveletAdam loss={last_mh[2]:.6f}  converged={converged_mh}")

# --- Diagnostic: track scale weights over time ---
def run_meta_wavelet_adam_tracked(n_steps=2000, buffer_capacity=8,
                                   meta_lr=0.0001, adam_lr=0.01,
                                   start=(-1.0, 1.0),
                                   beta1=0.9, beta2=0.999, eps=1e-8):
    """Same as run_meta_wavelet_adam but records scale weights each step."""
    x, y = start
    history = []
    weight_log = []   # scale weights at each step

    buf_x = CircularGradientBuffer(capacity=buffer_capacity, grad_shape=(1,))
    buf_y = CircularGradientBuffer(capacity=buffer_capacity, grad_shape=(1,))

    mx, my = 0.0, 0.0
    vx, vy = 0.0, 0.0

    net = MetaNet(input_dim=4, hidden_dim=8, output_dim=3)

    for t in range(1, n_steps + 1):
        loss, gx, gy = rosenbrock(x, y)
        history.append((x, y, loss))

        buf_x.write(np.array([gx]))
        buf_y.write(np.array([gy]))

        if buf_x.is_full():
            state = compute_state_features(t, n_steps, loss, gx, gy, x, y)
            scale_weights = net.forward(state)
            weight_log.append((t, scale_weights.copy()))

            weight_vec = make_scale_weight_vector(scale_weights, buffer_capacity)
            sig_x = buf_x.read().flatten()
            sig_y = buf_y.read().flatten()

            cx = haar_forward(sig_x) * weight_vec
            cy = haar_forward(sig_y) * weight_vec
            gx_clean = haar_inverse(cx)[-1]
            gy_clean = haar_inverse(cy)[-1]

            # Baseline
            mx_raw = beta1*mx + (1-beta1)*gx
            my_raw = beta1*my + (1-beta1)*gy
            vx_raw = beta2*vx + (1-beta2)*gx**2
            vy_raw = beta2*vy + (1-beta2)*gy**2
            mxr_hat = mx_raw / (1-beta1**t)
            myr_hat = my_raw / (1-beta1**t)
            vxr_hat = vx_raw / (1-beta2**t)
            vyr_hat = vy_raw / (1-beta2**t)
            x_raw = x - adam_lr * mxr_hat / (vxr_hat**0.5 + eps)
            y_raw = y - adam_lr * myr_hat / (vyr_hat**0.5 + eps)
            loss_baseline, _, _ = rosenbrock(x_raw, y_raw)

            # Adam update
            mx = beta1*mx + (1-beta1)*gx_clean
            my = beta1*my + (1-beta1)*gy_clean
            vx = beta2*vx + (1-beta2)*gx_clean**2
            vy = beta2*vy + (1-beta2)*gy_clean**2
            mx_hat = mx / (1-beta1**t)
            my_hat = my / (1-beta1**t)
            vx_hat = vx / (1-beta2**t)
            vy_hat = vy / (1-beta2**t)
            x_new = x - adam_lr * mx_hat / (vx_hat**0.5 + eps)
            y_new = y - adam_lr * my_hat / (vy_hat**0.5 + eps)
            loss_new, _, _ = rosenbrock(x_new, y_new)
            meta_loss = loss_new - loss_baseline

            # Meta backward
            dL_dweights = np.zeros(3)
            eps_meta = 1e-5
            for k in range(3):
                sw_plus  = scale_weights.copy(); sw_plus[k]  += eps_meta
                sw_minus = scale_weights.copy(); sw_minus[k] -= eps_meta
                wv_plus  = make_scale_weight_vector(sw_plus,  buffer_capacity)
                wv_minus = make_scale_weight_vector(sw_minus, buffer_capacity)
                gxp = haar_inverse(haar_forward(sig_x) * wv_plus)[-1]
                gyp = haar_inverse(haar_forward(sig_y) * wv_plus)[-1]
                gxm = haar_inverse(haar_forward(sig_x) * wv_minus)[-1]
                gym = haar_inverse(haar_forward(sig_y) * wv_minus)[-1]
                mxp = beta1*mx + (1-beta1)*gxp
                myp = beta1*my + (1-beta1)*gyp
                vxp = beta2*vx + (1-beta2)*gxp**2
                vyp = beta2*vy + (1-beta2)*gyp**2
                mxp_hat = mxp/(1-beta1**t); myp_hat = myp/(1-beta1**t)
                vxp_hat = vxp/(1-beta2**t); vyp_hat = vyp/(1-beta2**t)
                xp = x - adam_lr*mxp_hat/(vxp_hat**0.5+eps)
                yp = y - adam_lr*myp_hat/(vyp_hat**0.5+eps)
                mxm = beta1*mx + (1-beta1)*gxm
                mym = beta1*my + (1-beta1)*gym
                vxm = beta2*vx + (1-beta2)*gxm**2
                vym = beta2*vy + (1-beta2)*gym**2
                mxm_hat = mxm/(1-beta1**t); mym_hat = mym/(1-beta1**t)
                vxm_hat = vxm/(1-beta2**t); vym_hat = vym/(1-beta2**t)
                xm = x - adam_lr*mxm_hat/(vxm_hat**0.5+eps)
                ym = y - adam_lr*mym_hat/(vym_hat**0.5+eps)
                lp, _, _ = rosenbrock(xp, yp)
                lm, _, _ = rosenbrock(xm, ym)
                dL_dweights[k] = (lp - lm) / (2*eps_meta)

            net.zero_grad()
            net.backward(dL_dweights)
            net.update(meta_lr)
            x, y = x_new, y_new

        else:
            mx = beta1*mx + (1-beta1)*gx
            my = beta1*my + (1-beta1)*gy
            vx = beta2*vx + (1-beta2)*gx**2
            vy = beta2*vy + (1-beta2)*gy**2
            mx_hat = mx/(1-beta1**t); my_hat = my/(1-beta1**t)
            vx_hat = vx/(1-beta2**t); vy_hat = vy/(1-beta2**t)
            x = x - adam_lr*mx_hat/(vx_hat**0.5+eps)
            y = y - adam_lr*my_hat/(vy_hat**0.5+eps)

    return history, weight_log


print("\n--- Scale weight evolution ---")
h_tracked, weight_log = run_meta_wavelet_adam_tracked(n_steps=2000)

# Sample weights at different stages
stages = [
    ("Early   (step  50)", 0),
    ("Early   (step 100)", 1),
    ("Mid     (step 500)", 2),
    ("Late    (step 999)", 3),
]

for label, idx in stages:
    if idx < len(weight_log):
        step, w = weight_log[idx * (len(weight_log)//4)]
        print(f"  {label}: scale3={w[0]:.3f}  scale2={w[1]:.3f}  scale1={w[2]:.3f}")