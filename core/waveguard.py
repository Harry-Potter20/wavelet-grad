import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.learned_threshold_correction import (
    generate_correction_data, compute_mad_threshold
)
from core.uncertain_threshold_correction import (
    UncertainCorrectionNet, train_uncertain_correction_model
)
from core.learned_noise_model import extract_coefficient_stats

# =============================================================
# WAVEGUARD — ADAPTIVE SNR + UNCERTAINTY-AWARE DENOISING
#
# Combines three mechanisms:
#   1. MAD baseline threshold — robust, zero-tuning
#   2. Learned correction — residual adjustment to MAD
#   3. Dual gating — SNR gate × confidence gate
#
# Gate formula:
#   combined_gate = (snr_gate ** α) * (confidence ** β)
#
#   α — strictness on noise presence (SNR gate weight)
#   β — trust in learned model (confidence gate weight)
#
# Limiting cases:
#   α=1, β=0 → pure AdaptiveSNR (ignore model)
#   α=0, β=1 → pure confidence gating (ignore SNR)
#   α=1, β=1 → equal weight (default WaveGuard)
#
# Final correction:
#   correction = 1 + combined_gate * (raw_correction - 1)
#   T_final = T_MAD * correction
# =============================================================


def train_waveguard(n_samples=10000, n_epochs=50, seed=0):
    """
    Train the UncertainCorrectionNet for WaveGuard.
    Returns (net, X_mean, X_std).
    """
    return train_uncertain_correction_model(
        n_samples=n_samples, n_epochs=n_epochs, seed=seed
    )


def denoise_waveguard_batch(buf, ema_weights, net, X_mean, X_std,
                             alpha=1.0, beta=1.0, N=8):
    """
    WaveGuard vectorised denoising.

    Args:
        buf:         (N, D) gradient buffer
        ema_weights: (N,) EMA weights
        net:         trained UncertainCorrectionNet
        X_mean, X_std: feature normalisation stats
        alpha:       SNR gate exponent (strictness on noise)
        beta:        confidence gate exponent (model trust)

    Returns:
        clean: (D,) denoised gradient
    """
    from experiments.session11_adaptive import get_haar_matrix

    N_buf, D = buf.shape
    H = get_haar_matrix(N_buf)

    # Haar transform
    coeffs = H.T @ buf          # (N, D)
    detail = coeffs[1:]         # (N-1, D)

    # ── MAD baseline ──
    sigma_hat = np.median(np.abs(detail), axis=0) / 0.6745   # (D,)
    T_mad     = sigma_hat * np.sqrt(2 * np.log(N_buf))        # (D,)

    # ── SNR gate ──
    coarse_energy = np.sum(coeffs[:N_buf//2]**2, axis=0) + 1e-10
    fine_energy   = np.sum(coeffs[N_buf//2:]**2, axis=0) + 1e-10
    snr           = coarse_energy / fine_energy                # (D,)
    # High SNR → signal dominates → gate near 0 (don't correct)
    # Low SNR  → noise dominates  → gate near 1 (correct)
    snr_gate = 1.0 / (1.0 + snr)                             # (D,) in (0,1)

    # ── Learned correction + confidence ──
    feats = np.stack(
        [extract_coefficient_stats(coeffs[:, j], N_buf) for j in range(D)],
        axis=0
    )                                                          # (D, n_feats)
    feats_norm = (feats - X_mean) / X_std

    # Vectorised forward pass — both heads
    z1 = feats_norm @ net.W1.T + net.b1
    h1 = np.maximum(0, z1)
    z2 = h1 @ net.W2.T + net.b2
    h2 = np.maximum(0, z2)

    mu  = (h2 @ net.W_mu.T + net.b_mu).squeeze()   # (D,)
    lv  = (h2 @ net.W_lv.T + net.b_lv).squeeze()   # (D,)
    lv  = np.clip(lv, -6, 6)

    uncertainty = np.exp(0.5 * lv)   # predictive std

    # normalize uncertainty into a usable range
    uncertainty_norm = uncertainty / (np.mean(uncertainty) + 1e-8)

    # convert to confidence (smooth bounded mapping)
    confidence = 1.0 / (1.0 + uncertainty_norm)
    raw_correction = np.exp(np.clip(mu, -1.5, 1.5)) # (D,)

    # ── Combined gate (α, β parameterised) ──
    combined_gate = (
        alpha * snr_gate +
        beta  * confidence
    ) / (alpha + beta + 1e-8)
    
    # ── Final correction ──
    correction = 1.0 + combined_gate * (raw_correction - 1.0)
    correction = np.clip(correction, 0.5, 3.0)
    
    # prevent extreme threshold collapse
    thresholds = np.clip(T_mad * correction,
                        0.2 * T_mad,
                        2.5 * T_mad)

    # ── Soft threshold ──
    denoised_coeffs = coeffs.copy()
    denoised_coeffs[1:] = np.sign(detail) * np.maximum(
        np.abs(detail) - thresholds[np.newaxis, :], 0
    )

    # ── Inverse transform + EMA ──
    denoised_buf = H @ denoised_coeffs
    return ema_weights @ denoised_buf


if __name__ == "__main__":
    # Quick verification
    import numpy as np
    from core.uncertain_threshold_correction import train_uncertain_correction_model

    print("Training WaveGuard model...")
    net, X_mean, X_std = train_waveguard(n_samples=3000, n_epochs=20)

    rng = np.random.default_rng(42)
    buf = rng.standard_normal((8, 13)) * 0.5
    ema = np.ones(8) / 8

    clean = denoise_waveguard_batch(buf, ema, net, X_mean, X_std,
                                     alpha=1.0, beta=1.0)
    print(f"Input shape:  {buf.shape}")
    print(f"Output shape: {clean.shape}")
    print(f"Output sample: {np.round(clean[:4], 4)}")
    print("WaveGuard works")