import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.wavelets import haar_forward, haar_inverse
from core.learned_noise_model import extract_coefficient_stats

# =============================================================
# LEARNED THRESHOLD CORRECTION
#
# Instead of predicting σ from scratch, learn a multiplicative
# correction to the MAD threshold:
#
#   T_final = T_MAD * exp(f(features))
#
# Network output f(x) is a scalar correction in log space:
#   f(x) = 0  → no correction, use MAD threshold as-is
#   f(x) > 0  → inflate threshold (more aggressive denoising)
#   f(x) < 0  → deflate threshold (less denoising)
#
# Training target:
#   log(T_optimal / T_MAD)
#
# Where T_optimal minimises reconstruction error on synthetic
# data where the true signal is known.
#
# Advantage over direct σ prediction:
#   - Network learns residual only — small correction range
#   - MAD already handles the main estimation
#   - Corrections are in log space — symmetric, stable
#   - Generalises across noise types automatically
# =============================================================


def compute_mad_threshold(coeffs, N=8):
    """MAD estimator threshold from finest-scale coefficients."""
    finest    = np.abs(coeffs[N//2:])
    sigma_mad = np.median(finest) / 0.6745
    threshold = sigma_mad * np.sqrt(2 * np.log(max(N, 2)))
    return threshold, sigma_mad


def soft_threshold(coeffs, threshold):
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)


def find_optimal_threshold(buf, true_signal, N=8):
    """
    Find threshold that minimises reconstruction error
    on known true signal.

    Searches over a grid of threshold multipliers applied
    to the MAD threshold.
    """
    coeffs = haar_forward(buf)
    T_mad, _ = compute_mad_threshold(coeffs, N)

    if T_mad < 1e-10:
        return T_mad, 1.0  # degenerate case

    # Grid search over multipliers
    multipliers = np.exp(np.linspace(-2.0, 2.0, 40))
    best_error = np.inf
    best_mult  = 1.0

    for mult in multipliers:
        T = T_mad * mult
        c_denoised = coeffs.copy()
        c_denoised[1:] = soft_threshold(coeffs[1:], T)
        denoised = haar_inverse(c_denoised)
        error = np.mean((denoised - true_signal)**2)
        if error < best_error:
            best_error = error
            best_mult  = mult

    return T_mad * best_mult, best_mult


def generate_correction_data(n_samples=20000, N=8, seed=0):
    """
    Generate (features, log_correction) training pairs.

    For each sample:
    1. Sample σ and noise type
    2. Generate true signal + noisy buffer
    3. Compute MAD threshold
    4. Find optimal threshold via grid search
    5. Target = log(T_optimal / T_MAD)
    6. Features = coefficient statistics
    """
    rng = np.random.default_rng(seed)
    noise_types = ['gaussian', 'rician', 'poisson']

    X, y = [], []

    for i in range(n_samples):
        # Sample noise level
        sigma = np.exp(rng.uniform(np.log(0.01), np.log(5.0)))

        # True smooth signal
        true_signal = np.cumsum(rng.standard_normal(N)) * 0.1

        # Noise type
        noise_type = rng.choice(noise_types)
        if noise_type == 'gaussian':
            noise = rng.standard_normal(N) * sigma
        elif noise_type == 'rician':
            nr = rng.standard_normal(N) * sigma
            ni = rng.standard_normal(N) * sigma
            noise = np.sqrt(nr**2 + ni**2) - sigma * np.sqrt(np.pi/2)
        else:
            lam = np.maximum(np.abs(true_signal) * 10 + 1, 0.1)
            noise = (rng.poisson(lam * sigma) / sigma - lam) * sigma / 10

        buf = true_signal + noise

        # Compute MAD threshold and optimal threshold
        coeffs = haar_forward(buf)
        _, best_mult = find_optimal_threshold(buf, true_signal, N)

        # Skip degenerate cases
        if best_mult <= 0 or np.isnan(best_mult) or np.isinf(best_mult):
            continue

        # Features from coefficient statistics
        features = extract_coefficient_stats(coeffs, N)

        # Target: log correction factor
        log_correction = np.log(best_mult)

        X.append(features)
        y.append(log_correction)

    print(f"Generated {len(X)} valid samples")
    print(f"Correction range: [{np.min(y):.3f}, {np.max(y):.3f}]")
    print(f"Mean correction:  {np.mean(y):.3f}  std: {np.std(y):.3f}")
    return np.array(X), np.array(y)


class CorrectionNet:
    """
    Small MLP predicting log(T_optimal / T_MAD).
    Input: coefficient statistics (8 features)
    Output: scalar correction in log space
    """

    def __init__(self, input_dim=8):
        rng = np.random.default_rng(42)
        self.W1 = rng.standard_normal((32, input_dim)) * np.sqrt(2/input_dim)
        self.b1 = np.zeros(32)
        self.W2 = rng.standard_normal((16, 32)) * np.sqrt(2/32)
        self.b2 = np.zeros(16)
        self.W3 = rng.standard_normal((1, 16)) * np.sqrt(2/16)
        self.b3 = np.zeros(1)

        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._t = 0

    def _params(self):
        return {'W1': self.W1, 'b1': self.b1,
                'W2': self.W2, 'b2': self.b2,
                'W3': self.W3, 'b3': self.b3}

    def _relu(self, z): return np.maximum(0, z)
    def _relu_grad(self, z): return (z > 0).astype(float)

    def forward(self, x):
        self.x  = x
        self.z1 = self.W1 @ x + self.b1
        self.h1 = self._relu(self.z1)
        self.z2 = self.W2 @ self.h1 + self.b2
        self.h2 = self._relu(self.z2)
        self.z3 = self.W3 @ self.h2 + self.b3
        return self.z3[0]

    def backward(self, dL_dout):
        dz3 = np.array([dL_dout])
        dW3 = dz3[:, None] * self.h2[None, :]
        db3 = dz3
        dh2 = self.W3.T @ dz3
        dz2 = dh2 * self._relu_grad(self.z2)
        dW2 = dz2[:, None] * self.h1[None, :]
        db2 = dz2
        dh1 = self.W2.T @ dz2
        dz1 = dh1 * self._relu_grad(self.z1)
        dW1 = dz1[:, None] * self.x[None, :]
        db1 = dz1
        return {'W1': dW1, 'b1': db1,
                'W2': dW2, 'b2': db2,
                'W3': dW3, 'b3': db3}

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


def train_correction_model(n_samples=20000, n_epochs=100,
                            lr=0.001, batch_size=64, seed=0):
    print("Generating correction training data...")
    X, y = generate_correction_data(n_samples=n_samples, seed=seed)

    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    net = CorrectionNet(input_dim=X.shape[1])
    rng = np.random.default_rng(seed)
    n   = len(y)

    print(f"Training {n} samples, {n_epochs} epochs, batch={batch_size}...")

    for epoch in range(n_epochs):
        idx        = rng.permutation(n)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n - batch_size + 1, batch_size):
            batch_idx = idx[start:start+batch_size]
            X_b = X_norm[batch_idx]
            y_b = y[batch_idx]

            accum_grads = {k: np.zeros_like(v)
                           for k, v in net._params().items()}
            batch_loss = 0.0

            for i in range(batch_size):
                pred = net.forward(X_b[i])
                loss = (pred - y_b[i])**2
                batch_loss += loss
                dL = 2 * (pred - y_b[i])
                grads = net.backward(dL)
                for k in accum_grads:
                    accum_grads[k] += grads[k] / batch_size

            net.update(accum_grads, lr=lr)
            epoch_loss += batch_loss / batch_size
            n_batches  += 1

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{n_epochs}  "
                  f"loss={epoch_loss/n_batches:.6f}")

    return net, X_mean, X_std


def apply_learned_threshold(buf, net, X_mean, X_std, N=8):
    """
    Denoise buffer using MAD + learned correction.

    Returns denoised signal (same length as buf).
    """
    coeffs = haar_forward(buf)
    T_mad, _ = compute_mad_threshold(coeffs, N)

    # Extract features and normalise
    features   = extract_coefficient_stats(coeffs, N)
    feats_norm = (features - X_mean) / X_std

    # Predict correction
    log_correction = net.forward(feats_norm)
    correction     = np.exp(np.clip(log_correction, -3.0, 3.0))
    T_final        = T_mad * correction

    # Apply corrected threshold
    c_denoised = coeffs.copy()
    c_denoised[1:] = soft_threshold(coeffs[1:], T_final)
    return haar_inverse(c_denoised)


if __name__ == "__main__":
    net, X_mean, X_std = train_correction_model(
        n_samples=10000, n_epochs=50
    )

    print("\nEvaluating...")
    rng = np.random.default_rng(99)

    mad_errors, learned_errors = [], []

    for _ in range(500):
        sigma = np.exp(rng.uniform(np.log(0.01), np.log(5.0)))
        true_signal = np.cumsum(rng.standard_normal(8)) * 0.1
        buf = true_signal + rng.standard_normal(8) * sigma

        # MAD baseline
        coeffs = haar_forward(buf)
        T_mad, _ = compute_mad_threshold(coeffs)
        c = coeffs.copy()
        c[1:] = soft_threshold(coeffs[1:], T_mad)
        denoised_mad = haar_inverse(c)
        mad_errors.append(np.mean((denoised_mad - true_signal)**2))

        # Learned correction
        denoised_learned = apply_learned_threshold(buf, net, X_mean, X_std)
        learned_errors.append(np.mean((denoised_learned - true_signal)**2))

    print(f"MAD MSE:     mean={np.mean(mad_errors):.6f}  "
          f"median={np.median(mad_errors):.6f}")
    print(f"Learned MSE: mean={np.mean(learned_errors):.6f}  "
          f"median={np.median(learned_errors):.6f}")
    improvement = (np.mean(mad_errors) - np.mean(learned_errors)) \
                  / np.mean(mad_errors) * 100
    print(f"Improvement: {improvement:.1f}%")