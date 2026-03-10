import numpy as np
import math
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problems.noisy_rosenbrock import noisy_rosenbrock
from core.buffers import CircularGradientBuffer
from core.wavelets import haar_forward, haar_inverse, make_scale_weight_vector
from core.optimizers import (
    run_adam, run_noisy_adam,
    analytical_meta_gradient,
    compute_state_features,
    AdamOptimizer,
)
from core.meta_net import MetaNet
from experiments.session7_noisy import run_noisy_wavelet_adam, noise_adaptive_params

# =============================================================
# SESSION 8: METANET ON NOISY TRAJECTORIES
#
# Goals:
#   1. Add gradient variance as 5th input feature
#   2. Train MetaNet on noisy trajectories (σ=1.0, σ=2.0)
#   3. Compare MetaWaveletAdam vs Adam across noise levels
#
# Key insight: MetaNet must see noisy training to learn denoising.
# A network trained only on clean gradients has never needed to
# suppress fine scales — it won't know when to do it.
# =============================================================

BUFFER_CAPACITY = 8
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
ADAM_LR = 0.01


# =============================================================
# NOISY META WAVELET ADAM
#
# Same structure as run_analytical_meta_adam from Session 6,
# but:
#   - Uses noisy_rosenbrock instead of rosenbrock
#   - Passes buffers to compute_state_features (5 features)
#   - MetaNet input_dim=5 instead of 4
# =============================================================

def run_noisy_meta_wavelet_adam(sigma, n_steps=2000,
                                 seed=0, buffer_capacity=8,
                                 meta_lr=0.001, adam_lr=0.01,
                                 window_size=20, start=(-1.0, 1.0),
                                 beta1=0.9, beta2=0.999, eps=1e-8,
                                 net=None, train=True):
    """
    MetaWaveletAdam trained on noisy gradients.
    Uses 5-feature state including gradient variance.

    If net is provided, uses it directly (inference mode).
    If net is None, creates a new one and trains online.
    """
    from collections import deque
    rng = np.random.default_rng(seed)

    x, y = start
    history = []
    loss_window = deque(maxlen=window_size * 2)

    buf_x = CircularGradientBuffer(capacity=buffer_capacity, grad_shape=(1,))
    buf_y = CircularGradientBuffer(capacity=buffer_capacity, grad_shape=(1,))

    mx, my = 0.0, 0.0
    vx, vy = 0.0, 0.0

    training_mode = train # <- explicit flag
    if net is None:
        net = MetaNet(input_dim=5, hidden_dim=16, output_dim=3)
    meta_opt = AdamOptimizer(net.params, lr=meta_lr)

    for t in range(1, n_steps + 1):
        loss, gx, gy = noisy_rosenbrock(x, y, sigma=sigma, rng=rng)
        history.append((x, y, loss))
        loss_window.append(loss)

        buf_x.write(np.array([gx]))
        buf_y.write(np.array([gy]))

        if buf_x.is_full():
            # 5-feature state including variance
            state = compute_state_features(
                t, loss, gx, gy, x, y, buf_x, buf_y
            )
            scale_weights = net.forward(state)

            weight_vec = make_scale_weight_vector(scale_weights, buffer_capacity)
            sig_x = buf_x.read().flatten()
            sig_y = buf_y.read().flatten()

            gx_clean = haar_inverse(haar_forward(sig_x) * weight_vec)[-1]
            gy_clean = haar_inverse(haar_forward(sig_y) * weight_vec)[-1]
        else:
            gx_clean, gy_clean = gx, gy

        # Main Adam update
        mx = beta1*mx + (1-beta1)*gx_clean
        my = beta1*my + (1-beta1)*gy_clean
        vx = beta2*vx + (1-beta2)*gx_clean**2
        vy = beta2*vy + (1-beta2)*gy_clean**2
        mxh = mx/(1-beta1**t); myh = my/(1-beta1**t)
        vxh = vx/(1-beta2**t); vyh = vy/(1-beta2**t)
        x = x - adam_lr*mxh/(np.sqrt(vxh)+eps)
        y = y - adam_lr*myh/(np.sqrt(vyh)+eps)

        # MetaNet update (training mode only)
        if training_mode and buf_x.is_full() and \
           len(loss_window) >= window_size * 2:
            recent  = np.mean(list(loss_window)[-window_size:])
            earlier = np.mean(list(loss_window)[:window_size])
            meta_loss_signal = (recent - earlier) / (earlier + 1e-8)
            meta_loss_signal = np.clip(meta_loss_signal, -1.0, 1.0)

            grad_sw, _ = analytical_meta_gradient(
                sig_x, sig_y, mx, my, vx, vy, x, y, t, scale_weights
            )
            grad_sw = grad_sw * meta_loss_signal

            net.zero_grad()
            net.backward(grad_sw)
            meta_opt.step(net.grads)

    return history, net


def find_optimal_weights(sig_x, sig_y, mx, my, vx, vy,
                          x, y, t, sigma, rng):
    candidates = [
        np.array([0.9, 0.9, 0.9]),
        np.array([0.9, 0.7, 0.3]),
        np.array([0.9, 0.5, 0.1]),
        np.array([0.7, 0.5, 0.3]),
        np.array([0.5, 0.5, 0.5]),
        np.array([1.0, 0.8, 0.2]),
        np.array([0.8, 0.8, 0.8]),
        np.array([1.0, 0.6, 0.1]),
        np.array([0.6, 0.4, 0.2]),
        np.array([1.0, 1.0, 0.1]),
    ]

    best_loss = np.inf
    best_weights = candidates[4]  # neutral default

    for cand in candidates:
        cand_clipped = np.clip(cand, 0.01, 0.99)
        wv = make_scale_weight_vector(
            np.log(cand_clipped / (1 - cand_clipped)), BUFFER_CAPACITY
        )
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
        ln, _, _ = noisy_rosenbrock(xn, yn, sigma=sigma, rng=rng)

        if ln < best_loss:
            best_loss = ln
            best_weights = cand

    return best_weights

def collect_trajectory(sigma, n_steps=500, seed=0):
    """
    Run Adam on noisy Rosenbrock and record:
    - state features at each step
    - optimal scale weights found by gradient descent
    """
    rng = np.random.default_rng(seed)

    buf_x = CircularGradientBuffer(capacity=BUFFER_CAPACITY, grad_shape=(1,))
    buf_y = CircularGradientBuffer(capacity=BUFFER_CAPACITY, grad_shape=(1,))

    x, y = -1.0, 1.0
    mx, my = 0.0, 0.0
    vx, vy = 0.0, 0.0

    data = []

    for t in range(1, n_steps + 1):
        loss, gx, gy = noisy_rosenbrock(x, y, sigma=sigma, rng=rng)

        buf_x.write(np.array([gx]))
        buf_y.write(np.array([gy]))

        if buf_x.is_full():
            sig_x = buf_x.read().flatten()
            sig_y = buf_y.read().flatten()

            state = compute_state_features(t, loss, gx, gy, x, y, buf_x, buf_y)

            best_weights = find_optimal_weights(
                sig_x, sig_y, mx, my, vx, vy, x, y, t, sigma, rng
            )
            data.append((state, best_weights))

        # Standard Adam update
        mx = BETA1*mx + (1-BETA1)*gx
        my = BETA1*my + (1-BETA1)*gy
        vx = BETA2*vx + (1-BETA2)*gx**2
        vy = BETA2*vy + (1-BETA2)*gy**2
        mxh = mx/(1-BETA1**t); myh = my/(1-BETA1**t)
        vxh = vx/(1-BETA2**t); vyh = vy/(1-BETA2**t)
        x = x - ADAM_LR*mxh/(np.sqrt(vxh)+EPS)
        y = y - ADAM_LR*myh/(np.sqrt(vyh)+EPS)

    return data

def train_offline(sigmas=[0.0, 0.5, 1.0, 2.0, 5.0],
                  n_trajectories=5, n_epochs=50, meta_lr=0.001):
    """
    Offline training loop:
    1. Collect trajectories at each noise level
    2. Train MetaNet on (state, optimal_weights) pairs via MSE
    3. Return trained network
    """
    print("Collecting trajectories...")
    dataset = []
    for sigma in sigmas:
        for seed in range(n_trajectories):
            traj = collect_trajectory(sigma, n_steps=500, seed=seed)
            dataset.extend(traj)
    print(f"  Collected {len(dataset)} training pairs")

    net = MetaNet(input_dim=5, hidden_dim=16, output_dim=3)
    meta_opt = AdamOptimizer(net.params, lr=meta_lr)

    print(f"Training for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        np.random.shuffle(dataset)
        epoch_loss = 0.0

        for state, target_weights in dataset:
            predicted = net.forward(state)

            # MSE loss: predicted scale weights vs optimal weights
            diff = predicted - target_weights
            mse = np.mean(diff**2)
            epoch_loss += mse

            # Gradient of MSE w.r.t. predicted weights
            dL_dpred = 2 * diff / len(diff)

            net.zero_grad()
            net.backward(dL_dpred)
            meta_opt.step(net.grads)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}  loss={epoch_loss/len(dataset):.6f}")

    return net


# =============================================================
# TRAINING: collect a trained net on mixed noise levels
# =============================================================

def train_meta_net(sigmas=[0.5, 1.0, 2.0],
                   n_steps=2000, n_episodes=3, meta_lr=0.001):
    """
    Train a single MetaNet across multiple noise levels and episodes.
    Returns the trained network.
    """
    print("Training MetaNet on noisy trajectories...")
    net = MetaNet(input_dim=5, hidden_dim=16, output_dim=3)

    for episode in range(n_episodes):
        for sigma in sigmas:
            seed = episode * len(sigmas) + sigmas.index(sigma)
            _, net = run_noisy_meta_wavelet_adam(
                sigma=sigma, n_steps=n_steps,
                seed=seed, meta_lr=meta_lr, net=net, train=True
            )
        print(f"Episode {episode + 1}/{n_episodes} complete")
            # Pass trained net to next episode as warm start
            # (net is modified in place during training)

    print(f"Training complete — {n_episodes} episodes × {len(sigmas)} noise levels")
    return net

def inspect_net(net):
    """Show what scale weights the net outputs in different conditions."""
    print("\n--- MetaNet scale weight inspection ---")
    print(f"{'condition':>30} {'scale3':>8} {'scale2':>8} {'scale1':>8}")
    print("-" * 58)

    from core.meta_net import sigmoid
    conditions = [
        ("clean, early",      [0.1, -2.0, -1.0, -0.5, -2.0]),
        ("clean, late",       [0.5, -8.0, -3.0, -3.0, -3.0]),
        ("noisy σ=1, early",  [0.1, -1.0, -0.5, -0.5,  0.5]),
        ("noisy σ=1, late",   [0.5, -5.0, -2.0, -2.0,  0.5]),
        ("noisy σ=5, early",  [0.1, -0.5,  0.5,  0.5,  2.0]),
        ("noisy σ=5, late",   [0.5, -3.0, -1.0, -1.0,  2.0]),
    ]

    for label, features in conditions:
        state = np.array(features)
        weights = net.forward(state)
        print(f"{label:>30} {weights[0]:>8.3f} {weights[1]:>8.3f} {weights[2]:>8.3f}")


# =============================================================
# BENCHMARK
# =============================================================

def benchmark(n_trials=3, n_steps=2000):
    print("\nTraining MetaNet offline...")
    trained_net = train_offline(sigmas=[0.0, 0.5, 1.0, 2.0, 5.0],
                                  n_trajectories=10, n_epochs=100, meta_lr=0.001)

    inspect_net(trained_net)
    
    # Sanity check — verify trained net differs from fresh net
    fresh_net = MetaNet(input_dim=5, hidden_dim=16, output_dim=3)
    test_state = np.array([0.3, -1.0, -0.5, -0.5, 0.5])
    trained_out = trained_net.forward(test_state)
    fresh_out = fresh_net.forward(test_state)
    print(f"\nSanity check:")
    print(f"  Trained net output: {np.round(trained_out, 4)}")
    print(f"  Fresh net output:   {np.round(fresh_out, 4)}")
    print(f"  Are they different: {not np.allclose(trained_out, fresh_out)}")

    sigma_levels = [0.0, 0.5, 1.0, 2.0, 5.0]
    print(f"\n{'sigma':>8} {'optimizer':>25} {'final_loss':>12} {'converged':>10}")
    print("-" * 62)

    for sigma in sigma_levels:
        results = {}

        # Adam baseline
        losses = []
        for seed in range(n_trials):
            h = run_noisy_adam(sigma, n_steps, seed)
            losses.append(h[-1][2])
        results['Adam'] = np.mean(losses)

        # Static WaveletAdam
        losses = []
        for seed in range(n_trials):
            if sigma == 0.0:
                h = run_noisy_adam(sigma, n_steps, seed)
            else:
                h = run_noisy_wavelet_adam(sigma, n_steps, seed,
                        raw_params=noise_adaptive_params(sigma))
            losses.append(h[-1][2])
        results['StaticWavelet'] = np.mean(losses)

        # MetaWaveletAdam (inference with trained net)
        losses = []
        for seed in range(n_trials):
            h, _ = run_noisy_meta_wavelet_adam(
                sigma=sigma, n_steps=n_steps,
                seed=seed, net=trained_net, train=False
            )
            losses.append(h[-1][2])
        results['MetaWavelet'] = np.mean(losses)

        for name, loss in results.items():
            print(f"{sigma:>8.1f} {name:>25} {loss:>12.6f}")
        print()


benchmark()