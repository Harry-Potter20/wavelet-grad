import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# =============================================================
# CNN PROBLEM — MNIST with Small ConvNet (im2col implementation)
#
# Architecture:
#   Conv1: 1 → 8 filters, 3×3  → (8, 26, 26)
#   ReLU + stride-2 subsample  → (8, 13, 13)
#   Conv2: 8 → 16 filters, 3×3 → (16, 11, 11)
#   ReLU + stride-2 subsample  → (16, 5, 5)
#   Flatten                    → 400
#   FC: 400 → 10
# =============================================================


def im2col(x, kH, kW):
    N, C, H, W = x.shape
    H_out = H - kH + 1
    W_out = W - kW + 1
    cols = np.zeros((N, C * kH * kW, H_out * W_out))
    idx = 0
    for i in range(H_out):
        for j in range(W_out):
            patch = x[:, :, i:i+kH, j:j+kW]
            cols[:, :, idx] = patch.reshape(N, -1)
            idx += 1
    return cols


def col2im(cols, x_shape, kH, kW):
    N, C, H, W = x_shape
    H_out = H - kH + 1
    W_out = W - kW + 1
    x = np.zeros(x_shape)
    idx = 0
    for i in range(H_out):
        for j in range(W_out):
            patch = cols[:, :, idx].reshape(N, C, kH, kW)
            x[:, :, i:i+kH, j:j+kW] += patch
            idx += 1
    return x


def conv2d_forward(x, W, b):
    N, C, H, W_in = x.shape
    C_out, _, kH, kW = W.shape
    H_out = H - kH + 1
    W_out = W_in - kW + 1
    cols  = im2col(x, kH, kW)
    W_col = W.reshape(C_out, -1)
    out   = np.zeros((N, C_out, H_out * W_out))
    for n in range(N):
        out[n] = W_col @ cols[n] + b[:, None]
    return out.reshape(N, C_out, H_out, W_out), cols


def conv2d_backward(dout, cols, x_shape, W):
    N, C_in, H, W_in = x_shape
    C_out, _, kH, kW = W.shape
    dout_flat = dout.reshape(N, C_out, -1)
    W_col     = W.reshape(C_out, -1)
    dW        = np.zeros_like(W_col)
    dx_cols   = np.zeros_like(cols)
    for n in range(N):
        dW       += dout_flat[n] @ cols[n].T
        dx_cols[n] = W_col.T @ dout_flat[n]
    dW = dW.reshape(W.shape)
    dx = col2im(dx_cols, x_shape, kH, kW)
    db = dout.sum(axis=(0, 2, 3))
    return dW, db, dx


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def init_params(seed=0):
    """Xavier initialisation — infers FC size from dummy forward pass."""
    rng = np.random.default_rng(seed)
    W1  = rng.standard_normal((8,  1, 3, 3)) * 0.1
    b1  = np.zeros(8)
    W2  = rng.standard_normal((16, 8, 3, 3)) * 0.1
    b2  = np.zeros(16)

    dummy    = np.zeros((1, 1, 28, 28))
    z1, _    = conv2d_forward(dummy, W1, b1)
    p1       = relu(z1)[:, :, ::2, ::2]
    z2, _    = conv2d_forward(p1, W2, b2)
    p2       = relu(z2)[:, :, ::2, ::2]
    fc_in    = p2.reshape(1, -1).shape[1]

    Wfc = rng.standard_normal((10, fc_in)) * 0.1
    bfc = np.zeros(10)

    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'Wfc': Wfc, 'bfc': bfc}


def forward(params, X):
    W1, b1   = params['W1'], params['b1']
    W2, b2   = params['W2'], params['b2']
    Wfc, bfc = params['Wfc'], params['bfc']
    N        = X.shape[0]
    x        = X.reshape(N, 1, 28, 28)

    z1, col1 = conv2d_forward(x,  W1, b1)
    p1       = relu(z1)[:, :, ::2, ::2]
    z2, col2 = conv2d_forward(p1, W2, b2)
    p2       = relu(z2)[:, :, ::2, ::2]
    flat     = p2.reshape(N, -1)
    logits   = flat @ Wfc.T + bfc

    cache = (x, col1, col2, z1, z2, p1, p2, flat)
    return logits, cache


def loss_fn(logits, y):
    probs = softmax(logits)
    loss  = -np.log(probs[np.arange(len(y)), y] + 1e-8).mean()
    acc   = (np.argmax(probs, axis=1) == y).mean()
    return loss, probs, acc


def backward(probs, y, cache, params):
    x, col1, col2, z1, z2, p1, p2, flat = cache
    N     = len(y)
    grads = {}

    dlogits = probs.copy()
    dlogits[np.arange(N), y] -= 1
    dlogits /= N

    grads['Wfc'] = dlogits.T @ flat
    grads['bfc'] = dlogits.sum(axis=0)

    dflat = dlogits @ params['Wfc']
    dp2   = dflat.reshape(p2.shape)

    da2      = np.repeat(np.repeat(dp2, 2, axis=2), 2, axis=3)
    da2      = da2[:, :, :z2.shape[2], :z2.shape[3]]
    dz2      = da2 * (z2 > 0)
    dW2, db2, dp1 = conv2d_backward(dz2, col2, p1.shape, params['W2'])
    grads['W2'] = dW2
    grads['b2'] = db2

    da1      = np.repeat(np.repeat(dp1, 2, axis=2), 2, axis=3)
    da1      = da1[:, :, :z1.shape[2], :z1.shape[3]]
    dz1      = da1 * (z1 > 0)
    dW1, db1, _ = conv2d_backward(dz1, col1, x.shape, params['W1'])
    grads['W1'] = dW1
    grads['b1'] = db1

    return grads


def cnn_loss_and_grads(params, X, y):
    logits, cache = forward(params, X)
    loss, probs, acc = loss_fn(logits, y)
    grads = backward(probs, y, cache, params)
    return loss, grads, acc


def get_batches(X, y, batch_size, rng):
    n   = len(y)
    idx = rng.permutation(n)
    for start in range(0, n - batch_size + 1, batch_size):
        yield X[idx[start:start+batch_size]], y[idx[start:start+batch_size]]


if __name__ == "__main__":
    from problems.mnist_problem import load_mnist_subset
    X_train, y_train, _, _ = load_mnist_subset(n_train=100, seed=0)
    params = init_params()
    loss, grads, acc = cnn_loss_and_grads(params, X_train[:8], y_train[:8])
    print(f"Loss: {loss:.4f}  Acc: {acc:.3f}")
    print(f"W1 grad: {grads['W1'].shape}  W2 grad: {grads['W2'].shape}")
    print(f"Wfc grad: {grads['Wfc'].shape}")