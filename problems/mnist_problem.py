import numpy as np
import os

# =============================================================
# MNIST PROBLEM
#
# 2-layer MLP: 784 → 128 → 10
# Loss: cross-entropy with softmax
# Parameters:
#   W1: (128, 784)  ~100K params
#   b1: (128,)
#   W2: (10, 128)   ~1.3K params
#   b2: (10,)
# Total: ~101K parameters
#
# Data: subset of MNIST or synthetic if not available
# =============================================================


def load_mnist_subset(n_train=1000, n_test=200, seed=0):
    """
    Load MNIST subset. Falls back to synthetic data if MNIST
    is not available — preserves the gradient structure without
    requiring a download.
    """
    try:
        # Try loading from common locations
        import urllib.request
        import gzip
        import pickle

        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mnist.pkl.gz')
        os.makedirs(os.path.dirname(data_path), exist_ok=True)

        if not os.path.exists(data_path):
            print("Downloading MNIST...")
            url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
            urllib.request.urlretrieve(url, data_path)

        with gzip.open(data_path, 'rb') as f:
            train_set, _, test_set = pickle.load(f, encoding='latin1')

        rng = np.random.default_rng(seed)
        train_idx = rng.choice(len(train_set[0]), n_train, replace=False)
        test_idx  = rng.choice(len(test_set[0]),  n_test,  replace=False)

        X_train = train_set[0][train_idx].astype(float)
        y_train = train_set[1][train_idx].astype(int)
        X_test  = test_set[0][test_idx].astype(float)
        y_test  = test_set[1][test_idx].astype(int)

        print(f"Loaded MNIST: {n_train} train, {n_test} test")
        return X_train, y_train, X_test, y_test

    except Exception as e:
        print(f"MNIST unavailable ({e}), using synthetic data")
        return make_synthetic_mnist(n_train, n_test, seed)


def make_synthetic_mnist(n_train=1000, n_test=200, seed=0):
    """
    Synthetic data with MNIST-like properties:
    - 784-dimensional inputs (28×28)
    - 10 classes
    - Structured: each class has a characteristic pattern
    - Pixel values in [0, 1]
    """
    rng = np.random.default_rng(seed)
    n_classes = 10
    n_features = 784

    # Class prototypes — each class has a random sparse pattern
    prototypes = rng.standard_normal((n_classes, n_features)) * 0.3
    prototypes = np.clip(prototypes, 0, 1)

    def make_split(n, seed_offset):
        rng2 = np.random.default_rng(seed + seed_offset)
        labels = rng2.integers(0, n_classes, n)
        X = prototypes[labels] + rng2.standard_normal((n, n_features)) * 0.2
        X = np.clip(X, 0, 1)
        return X, labels

    X_train, y_train = make_split(n_train, 0)
    X_test,  y_test  = make_split(n_test,  1)
    return X_train, y_train, X_test, y_test


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(float)


def cross_entropy_loss(probs, y, n):
    eps = 1e-8
    return -np.mean(np.log(probs[np.arange(n), y] + eps))


def mnist_loss_and_grads(params, X_batch, y_batch):
    """
    Forward + backward pass for 784→128→10 MLP.

    Activations: ReLU hidden, softmax output.
    Loss: cross-entropy.

    Args:
        params:  dict with W1(128,784), b1(128,), W2(10,128), b2(10,)
        X_batch: (batch_size, 784)
        y_batch: (batch_size,) int labels

    Returns:
        loss:  scalar
        grads: dict matching params
        acc:   float accuracy on batch
    """
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    n = len(y_batch)

    # Forward
    z1 = X_batch @ W1.T + b1        # (n, 128)
    h1 = relu(z1)                    # (n, 128)
    z2 = h1 @ W2.T + b2             # (n, 10)
    probs = softmax(z2)              # (n, 10)

    loss = cross_entropy_loss(probs, y_batch, n)
    acc  = np.mean(np.argmax(probs, axis=1) == y_batch)

    # Backward
    dz2 = probs.copy()
    dz2[np.arange(n), y_batch] -= 1
    dz2 /= n                         # (n, 10)

    dW2 = dz2.T @ h1                # (10, 128)
    db2 = dz2.sum(axis=0)           # (10,)

    dh1 = dz2 @ W2                  # (n, 128)
    dz1 = dh1 * relu_grad(z1)       # (n, 128)

    dW1 = dz1.T @ X_batch           # (128, 784)
    db1 = dz1.sum(axis=0)           # (128,)

    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
    return loss, grads, acc


def init_params(seed=0):
    """Xavier initialisation for 784→128→10 MLP."""
    rng = np.random.default_rng(seed)
    return {
        'W1': rng.standard_normal((128, 784)) * np.sqrt(2/784),
        'b1': np.zeros(128),
        'W2': rng.standard_normal((10, 128)) * np.sqrt(2/128),
        'b2': np.zeros(10),
    }


def get_batches(X, y, batch_size, rng):
    """Yield random mini-batches."""
    n = len(y)
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start+batch_size]
        yield X[batch_idx], y[batch_idx]


if __name__ == "__main__":
    # Gradient check
    X_train, y_train, _, _ = make_synthetic_mnist(n_train=32, seed=42)
    params = init_params(seed=42)
    X_b, y_b = X_train[:8], y_train[:8]

    loss, grads, acc = mnist_loss_and_grads(params, X_b, y_b)
    print(f"Initial loss: {loss:.4f}  acc: {acc:.3f}")

    # Numerical gradient check on W2 (smaller matrix)
    eps = 1e-5
    max_diff = 0.0
    W2_flat = params['W2'].flatten()
    g_flat  = grads['W2'].flatten()
    num_g   = np.zeros_like(W2_flat)

    for i in range(min(20, len(W2_flat))):
        W2_flat[i] += eps
        params['W2'] = W2_flat.reshape(10, 128)
        lp, _, _ = mnist_loss_and_grads(params, X_b, y_b)
        W2_flat[i] -= 2*eps
        params['W2'] = W2_flat.reshape(10, 128)
        lm, _, _ = mnist_loss_and_grads(params, X_b, y_b)
        W2_flat[i] += eps
        params['W2'] = W2_flat.reshape(10, 128)
        num_g[i] = (lp - lm) / (2*eps)

    max_diff = np.max(np.abs(g_flat[:20] - num_g[:20]))
    print(f"Gradient check max diff: {max_diff:.2e}")
    print(f"Gradient check passed: {max_diff < 1e-5}")