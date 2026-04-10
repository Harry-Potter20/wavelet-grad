import numpy as np

# =============================================================
# LEARNED NOISE MODEL — IMPROVEMENT 5 (FINAL CLEAN VERSION)
# =============================================================


def extract_coefficient_stats(coeffs, N=8):
    features = []
    n = N
    start = 1

    while start < n:
        end = min(start * 2, n)
        level_coeffs = np.abs(coeffs[start:end])

        if len(level_coeffs) == 0:
            break

        mean_val = np.mean(level_coeffs)

        # Fix: handle small levels safely
        if len(level_coeffs) < 2:
            std_val = 1e-8
            skew = 0.0
            kurt = 0.0
        else:
            std_val = np.std(level_coeffs) + 1e-8
            normed = (level_coeffs - mean_val) / std_val
            skew = np.mean(normed**3)
            kurt = np.mean(normed**4) - 3.0

        max_val = np.max(level_coeffs)

        features.extend([mean_val, std_val, skew, kurt, max_val])
        start = end

    features = np.array(features)

    # Enforce fixed size (critical)
    if len(features) < 15:
        features = np.pad(features, (0, 15 - len(features)))
    elif len(features) > 15:
        features = features[:15]

    return features
# =============================================================
# NETWORK
# =============================================================

class NoiseModelNet:

    def __init__(self, input_dim=15):
        rng = np.random.default_rng(42)

        self.W1 = rng.standard_normal((16, input_dim)) * np.sqrt(2/input_dim)
        self.b1 = np.zeros(16)

        self.W2 = rng.standard_normal((8, 16)) * np.sqrt(2/16)
        self.b2 = np.zeros(8)

        self.W3 = rng.standard_normal((1, 8)) * np.sqrt(2/8)
        self.b3 = np.zeros(1)

        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._t = 0

    def _params(self):
        return {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3
        }

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_grad(self, z):
        return (z > 0).astype(float)

    def forward(self, x):
        self.x = x

        self.z1 = self.W1 @ x + self.b1
        self.h1 = self._relu(self.z1)

        self.z2 = self.W2 @ self.h1 + self.b2
        self.h2 = self._relu(self.z2)

        self.z3 = self.W3 @ self.h2 + self.b3

        return self.z3[0]  # log(sigma)

    def predict_sigma(self, x):
        log_sigma = self.forward(x)
        return np.exp(np.clip(log_sigma, -10, 5))

    def backward(self, dL_dout):
        # Output layer
        dz3 = np.array([dL_dout])
        dW3 = dz3[:, None] * self.h2[None, :]
        db3 = dz3

        # Layer 2
        dh2 = self.W3.T @ dz3
        dz2 = dh2 * self._relu_grad(self.z2)
        dW2 = dz2[:, None] * self.h1[None, :]
        db2 = dz2

        # Layer 1
        dh1 = self.W2.T @ dz2
        dz1 = dh1 * self._relu_grad(self.z1)
        dW1 = dz1[:, None] * self.x[None, :]
        db1 = dz1

        return {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2,
            'W3': dW3, 'b3': db3
        }

    def update(self, grads, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        params = self._params()

        for k in params:
            g = grads[k]

            self._m[k] = beta1*self._m[k] + (1-beta1)*g
            self._v[k] = beta2*self._v[k] + (1-beta2)*g**2

            mh = self._m[k] / (1-beta1**self._t)
            vh = self._v[k] / (1-beta2**self._t)

            params[k] -= lr * mh / (np.sqrt(vh) + eps)


# =============================================================
# DATA GENERATION
# =============================================================

def generate_training_data(n_samples=10000, N=8, seed=0):
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.wavelets import haar_forward

    rng = np.random.default_rng(seed)

    X, y = [], []
    noise_types = ['gaussian', 'rician', 'poisson']

    for _ in range(n_samples):
        log_sigma = rng.uniform(np.log(0.01), np.log(5.0))
        sigma = np.exp(log_sigma)

        # smooth signal
        true_signal = rng.standard_normal(N) * 0.5
        true_signal = np.cumsum(true_signal) * 0.1

        noise_type = rng.choice(noise_types)

        if noise_type == 'gaussian':
            noise = rng.standard_normal(N) * sigma

        elif noise_type == 'rician':
            noise_re = rng.standard_normal(N) * sigma
            noise_im = rng.standard_normal(N) * sigma
            noise = np.sqrt(noise_re**2 + noise_im**2) - sigma * np.sqrt(np.pi/2)

        else:  # poisson
            lam = np.maximum(np.abs(true_signal) * 10 + 1, 0.1)
            noise = (rng.poisson(lam * sigma) / sigma - lam) * sigma / 10

        buf = true_signal + noise

        coeffs = haar_forward(buf)
        features = extract_coefficient_stats(coeffs, N)

        X.append(features)
        y.append(log_sigma)

    return np.array(X), np.array(y)


# =============================================================
# TRAINING
# =============================================================

def train_noise_model(n_samples=20000, n_epochs=100, lr=0.001,
                       batch_size=64, seed=0):
    print("Generating training data...")
    X, y = generate_training_data(n_samples=n_samples, seed=seed)

    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    y = np.array(y)

    net = NoiseModelNet(input_dim=X.shape[1])
    rng = np.random.default_rng(seed)

    print(f"Training {n_samples} samples, {n_epochs} epochs, batch={batch_size}...")

    for epoch in range(n_epochs):
        idx = rng.permutation(n_samples)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_samples - batch_size + 1, batch_size):
            batch_idx = idx[start:start+batch_size]
            X_b = X_norm[batch_idx]   # (batch, input_dim)
            y_b = y[batch_idx]         # (batch,)

            # Accumulate gradients over batch
            batch_loss = 0.0
            accum_grads = {k: np.zeros_like(v)
                           for k, v in net._params().items()}

            for i in range(batch_size):
                log_sigma_hat = net.forward(X_b[i])
                loss = (log_sigma_hat - y_b[i])**2
                batch_loss += loss
                dL_dout = 2 * (log_sigma_hat - y_b[i])
                grads = net.backward(dL_dout)
                for k in accum_grads:
                    accum_grads[k] += grads[k] / batch_size

            net.update(accum_grads, lr=lr)
            epoch_loss += batch_loss / batch_size
            n_batches  += 1

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{n_epochs}  "
                  f"loss={epoch_loss/n_batches:.6f}")

    return net, X_mean, X_std

# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":
    from core.wavelets import haar_forward

    net, X_mean, X_std = train_noise_model(
        n_samples=20000,
        n_epochs=100
    )

    print("\nTesting...")

    rng = np.random.default_rng(99)

    errors = []

    for _ in range(100):
        sigma_true = np.exp(rng.uniform(np.log(0.01), np.log(5.0)))

        signal = rng.standard_normal(8) * 0.3
        buf = signal + rng.standard_normal(8) * sigma_true

        coeffs = haar_forward(buf)
        feats = extract_coefficient_stats(coeffs)

        feats_norm = (feats - X_mean) / X_std
        sigma_hat = net.predict_sigma(feats_norm)

        errors.append(abs(sigma_hat - sigma_true) / (sigma_true + 1e-8))

    print(f"Mean relative error: {np.mean(errors):.3f}")
    print(f"Median relative error: {np.median(errors):.3f}")
    
    
# =============================================================
# RESULT SUMMARY
#
# MAD estimator (closed form, zero training):
#   Median relative error: 1.224
#   90th pct error:        17.564
#
# Learned NoiseModelNet (20K samples, 100 epochs):
#   Median relative error: 2.015
#   90th pct error:        37.983
#
# Conclusion: with N=8 buffer, 8 coefficients per estimate is
# insufficient for a neural network to outperform MAD.
# MAD is a near-optimal estimator for Gaussian noise at this
# sample size. The learned model adds variance without reducing
# bias.
#
# Path forward:
#   1. Larger buffer (N=32) gives MAD more coefficients
#   2. End-to-end threshold learning (Path B) — bypass σ
#      estimation entirely, learn threshold from data
#   3. Uncertainty-aware denoising (Improvement 7)
# =============================================================