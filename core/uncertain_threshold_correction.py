import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.wavelets import haar_forward, haar_inverse
from core.learned_noise_model import extract_coefficient_stats
from core.learned_threshold_correction import (
    generate_correction_data,
    compute_mad_threshold,
    soft_threshold
)


class UncertainCorrectionNet:
    """
    Uncertainty-aware threshold correction network.

    Outputs (μ, log_var) — mean and log variance of
    log(T_optimal / T_MAD) prediction.

    Architecture: input → 32 → 16 → 2
    Two output heads share all hidden layers.
    """

    def __init__(self, input_dim=8):
        rng = np.random.default_rng(42)
        self.W1  = rng.standard_normal((32, input_dim)) * np.sqrt(2/input_dim)
        self.b1  = np.zeros(32)
        self.W2  = rng.standard_normal((16, 32)) * np.sqrt(2/32)
        self.b2  = np.zeros(16)
        # Two output heads — shared trunk
        self.W_mu  = rng.standard_normal((1, 16)) * np.sqrt(2/16)
        self.b_mu  = np.zeros(1)
        self.W_lv  = rng.standard_normal((1, 16)) * np.sqrt(2/16)
        self.b_lv  = np.zeros(1)

        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._t = 0

    def _params(self):
        return {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W_mu': self.W_mu, 'b_mu': self.b_mu,
            'W_lv': self.W_lv, 'b_lv': self.b_lv,
        }

    def _relu(self, z): return np.maximum(0, z)
    def _relu_grad(self, z): return (z > 0).astype(float)

    def forward(self, x):
        self.x  = x
        self.z1 = self.W1 @ x + self.b1
        self.h1 = self._relu(self.z1)
        self.z2 = self.W2 @ self.h1 + self.b2
        self.h2 = self._relu(self.z2)
        # Two heads
        self.mu   = (self.W_mu @ self.h2 + self.b_mu)[0]
        self.lv   = (self.W_lv @ self.h2 + self.b_lv)[0]
        # Clamp log_var for stability
        self.lv   = np.clip(self.lv, -6, 6)
        return self.mu, self.lv

    def predict(self, x):
        """Return (mu, uncertainty) where uncertainty = exp(0.5*log_var)."""
        mu, lv = self.forward(x)
        return mu, np.exp(0.5 * lv)

    def backward(self, dL_dmu, dL_dlv):
        """Backprop through both output heads."""
        # μ head
        dz_mu = np.array([dL_dmu])
        dW_mu = dz_mu[:, None] * self.h2[None, :]
        db_mu = dz_mu
        dh2_mu = self.W_mu.T @ dz_mu

        # log_var head
        dz_lv = np.array([dL_dlv])
        dW_lv = dz_lv[:, None] * self.h2[None, :]
        db_lv = dz_lv
        dh2_lv = self.W_lv.T @ dz_lv

        # Shared trunk — sum gradients from both heads
        dh2 = dh2_mu + dh2_lv
        dz2 = dh2 * self._relu_grad(self.z2)
        dW2 = dz2[:, None] * self.h1[None, :]
        db2 = dz2

        dh1 = self.W2.T @ dz2
        dz1 = dh1 * self._relu_grad(self.z1)
        dW1 = dz1[:, None] * self.x[None, :]
        db1 = dz1

        return {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2,
            'W_mu': dW_mu, 'b_mu': db_mu,
            'W_lv': dW_lv, 'b_lv': db_lv,
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
            
            
def train_uncertain_correction_model(n_samples=20000, n_epochs=100,
                                      lr=0.001, batch_size=64, seed=0):
    print("Generating correction training data...")
    X, y = generate_correction_data(n_samples=n_samples, seed=seed)

    # Inject ambiguity — simulate irreducible uncertainty in optimal threshold
    y = np.array(y) + np.random.default_rng(seed+1).normal(0, 0.2, size=len(y))
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    net = UncertainCorrectionNet(input_dim=X.shape[1])
    rng = np.random.default_rng(seed)
    n   = len(y)

    print(f"Training {n} samples, {n_epochs} epochs (NLL loss)...")

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
                mu, lv = net.forward(X_b[i])
                var     = np.exp(lv)
                target  = y_b[i]

                # Gaussian NLL
                nll = 0.5 * lv + 0.5 * (target - mu)**2 / (var + 1e-8)
                batch_loss += nll

                # Gradients of NLL
                dL_dmu = (mu - target) / (var + 1e-8)
                dL_dlv = 0.5 - 0.5 * (target - mu)**2 / (var + 1e-8)

                grads = net.backward(dL_dmu, dL_dlv)
                for k in accum_grads:
                    accum_grads[k] += grads[k] / batch_size

            net.update(accum_grads, lr=lr)
            epoch_loss += batch_loss / batch_size
            n_batches  += 1

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{n_epochs}  "
                  f"NLL={epoch_loss/n_batches:.6f}")

    return net, X_mean, X_std

def denoise_uncertain_batch(buf, ema_weights, net, X_mean, X_std, N=8):
    """
    Vectorised uncertainty-aware denoising.

    confidence = exp(-uncertainty) gates the learned correction:
      high uncertainty → confidence low → stay near MAD
      low uncertainty  → confidence high → apply full correction
    """
    from experiments.session11_adaptive import get_haar_matrix

    N_buf, D = buf.shape
    H = get_haar_matrix(N_buf)
    coeffs = H.T @ buf        # (N, D)
    detail = coeffs[1:]       # (N-1, D)

    # MAD baseline
    sigma_hat = np.median(np.abs(detail), axis=0) / 0.6745
    T_mad     = sigma_hat * np.sqrt(2 * np.log(N_buf))

    # Vectorised feature extraction
    feats = np.stack(
        [extract_coefficient_stats(coeffs[:, j], N_buf) for j in range(D)],
        axis=0
    )
    feats_norm = (feats - X_mean) / X_std

    # Vectorised forward — both heads
    z1 = feats_norm @ net.W1.T + net.b1
    h1 = np.maximum(0, z1)
    z2 = h1 @ net.W2.T + net.b2
    h2 = np.maximum(0, z2)

    mu  = (h2 @ net.W_mu.T + net.b_mu).squeeze()   # (D,)
    lv  = (h2 @ net.W_lv.T + net.b_lv).squeeze()   # (D,)
    lv  = np.clip(lv, -6, 6)

    uncertainty = np.exp(0.5 * lv)                  # (D,) std
    confidence = 1 / (1 + uncertainty)
    
    
    raw_correction = np.exp(np.clip(mu, -1.0, 1.0))

    # Noise-level gate
    noise_level = sigma_hat / (np.mean(np.abs(buf), axis=0) + 1e-8)
    gate = np.clip((noise_level - 0.1) / 0.5, 0.0, 1.0)

    # Uncertainty-modulated correction
    # Step 1: apply noise gate first
    base_correction = 1 + gate * (raw_correction - 1)

    # Step 2: uncertainty decides how much to trust it
    correction = 1 + confidence * (base_correction - 1)
    correction = np.clip(correction, 0.8, 2.0)

    thresholds      = T_mad * correction * (1 + 0.5 * uncertainty)
    denoised_coeffs = coeffs.copy()
    denoised_coeffs[1:] = np.sign(detail) * np.maximum(
        np.abs(detail) - thresholds[np.newaxis, :], 0
    )

    denoised_buf = H @ denoised_coeffs
    return ema_weights @ denoised_buf