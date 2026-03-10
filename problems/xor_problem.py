import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# =============================================================
# XOR CLASSIFICATION PROBLEM
#
# 4 data points, 2 classes, not linearly separable.
# Requires a hidden layer to solve.
#
# Input:  X = [[0,0], [0,1], [1,0], [1,1]]
# Target: y = [0, 1, 1, 0]
#
# Network: 2 → 4 → 1  (sigmoid activations)
# Loss:    binary cross-entropy
#
# Gradients:
#   - Noisy near saddle points early in training
#   - Small and precise near the minimum
#   - High frequency oscillations during basin transitions
# =============================================================

# Fixed dataset — all 4 XOR points
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)

y = np.array([0, 1, 1, 0], dtype=float)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def xor_loss_and_grads(params):
    """
    Forward + backward pass for XOR network.

    Network: 2 → 4 → 1
      h = sigmoid(X @ W1 + b1)      shape: (4, 4)
      y_hat = sigmoid(h @ W2 + b2)  shape: (4, 1)
      loss = mean binary cross-entropy

    Args:
        params: dict with keys W1(4x2), b1(4,), W2(1x4), b2(1,)

    Returns:
        loss:  scalar
        grads: dict with same keys as params
    """
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']

    # Forward pass
    z1    = X @ W1.T + b1          # (4, 4)
    h     = sigmoid(z1)             # (4, 4)
    z2    = h @ W2.T + b2          # (4, 1)
    y_hat = sigmoid(z2).flatten()  # (4,)

    # Binary cross-entropy loss
    eps = 1e-8
    loss = -np.mean(y * np.log(y_hat + eps) +
                    (1 - y) * np.log(1 - y_hat + eps))

    # Backward pass
    n = len(y)                    
    dz2    = (y_hat - y) / n      # (4,)
    dz2    = dz2.reshape(-1, 1)                  # (4, 1)

    dW2 = dz2.T @ h                              # (1, 4)
    db2 = dz2.sum(axis=0)                        # (1,)

    dh  = dz2 @ W2                               # (4, 4)
    dz1 = dh * h * (1 - h)                       # (4, 4)

    dW1 = dz1.T @ X                              # (4, 2)
    db1 = dz1.sum(axis=0)                        # (4,)

    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
    return loss, grads

def xor_loss_and_grads_noisy(params, sigma=0.1, rng=None):
    """XOR with Gaussian noise added to gradients."""
    loss, grads = xor_loss_and_grads(params)
    if rng is None:
        rng = np.random.default_rng()
    noisy_grads = {}
    for k in grads:
        noise = rng.standard_normal(grads[k].shape) * sigma
        noisy_grads[k] = grads[k] + noise
    return loss, noisy_grads

def init_params(seed=0):
    """Xavier-initialized parameters for 2→4→1 network."""
    rng = np.random.default_rng(seed)
    return {
        'W1': rng.standard_normal((4, 2)) * np.sqrt(1/2),
        'b1': np.zeros(4),
        'W2': rng.standard_normal((1, 4)) * np.sqrt(1/4),
        'b2': np.zeros(1),
    }


def accuracy(params):
    """Classification accuracy on all 4 XOR points."""
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    h     = sigmoid(X @ W1.T + b1)
    y_hat = sigmoid(h @ W2.T + b2).flatten()
    preds = (y_hat > 0.5).astype(int)
    return np.mean(preds == y)


if __name__ == "__main__":
    # Verify gradients with numerical check
    params = init_params(seed=42)
    loss, grads = xor_loss_and_grads(params)
    print(f"Initial loss: {loss:.4f}")

    # Numerical gradient check
    eps = 1e-5
    max_diff = 0.0
    for key in params:
        p = params[key].flatten()
        g = grads[key].flatten()
        num_g = np.zeros_like(p)
        for i in range(len(p)):
            p[i] += eps
            params[key] = p.reshape(params[key].shape)
            loss_plus, _ = xor_loss_and_grads(params)
            p[i] -= 2*eps
            params[key] = p.reshape(params[key].shape)
            loss_minus, _ = xor_loss_and_grads(params)
            p[i] += eps
            params[key] = p.reshape(params[key].shape)
            num_g[i] = (loss_plus - loss_minus) / (2*eps)
        diff = np.max(np.abs(g - num_g))
        max_diff = max(max_diff, diff)
        print(f"  {key}: max_diff={diff:.2e}")

    print(f"\nOverall max diff: {max_diff:.2e}")
    print(f"Gradient check passed: {max_diff < 1e-5}")