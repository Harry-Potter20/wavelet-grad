import numpy as np
from core.wavelets import haar_forward, haar_inverse
# =============================================================
# PCA JOINT GRADIENT DENOISER
#
# Instead of denoising each parameter independently,
# we denoise the full gradient vector jointly using PCA.
#
# Algorithm:
#   1. Collect N gradient snapshots — shape (N, D)
#   2. Form Gram matrix K = G @ G.T — shape (N, N)
#   3. Decompose K via SVD — O(N³) not O(D³)
#   4. Extract top-k principal directions in param space
#   5. Project current gradient onto those directions
#   6. Denoise in low-dimensional PCA space
#   7. Reconstruct full gradient
#
# Key insight: N=8 (buffer size) so Gram matrix is always 8×8
# regardless of how many parameters D the network has.
# Computation scales as O(N²D) not O(D³).
#
# This captures inter-parameter correlations that per-element
# denoising misses — correlated gradient oscillations across
# parameters are identified and suppressed together.
# =============================================================


class PCAGradientDenoiser:
    """
    Joint gradient denoiser using Gram matrix PCA.

    Usage:
        denoiser = PCAGradientDenoiser(n_components=4)
        denoiser.update(grad_flat)      # call each step
        clean = denoiser.denoise(grad_flat)  # when buffer full
    """

    def __init__(self, n_components=4, buffer_size=8,
                 scale_weights=None):
        """
        Args:
            n_components: number of principal components to keep (k)
                          must be <= buffer_size
            buffer_size:  number of gradient snapshots to maintain
            scale_weights: array of shape (n_components,) — attenuation
                          per component. None = keep all equally.
        """
        assert n_components <= buffer_size, \
            f"n_components ({n_components}) must be <= buffer_size ({buffer_size})"

        self.k    = n_components
        self.N    = buffer_size
        self.dim  = None           # set on first update
        self.buf  = None           # shape (N, D) once initialized
        self.ptr  = 0
        self.count = 0

        if scale_weights is not None:
            assert len(scale_weights) == n_components
            self.scale_weights = np.array(scale_weights)
        else:
            self.scale_weights = np.ones(n_components)

    def update(self, grad_flat):
        """
        Add a new gradient snapshot to the buffer.

        Args:
            grad_flat: 1D array of shape (D,)
        """
        if self.dim is None:
            self.dim = len(grad_flat)
            self.buf = np.zeros((self.N, self.dim))

        self.buf[self.ptr] = grad_flat
        self.ptr  = (self.ptr + 1) % self.N
        self.count += 1

    @property
    def is_full(self):
        return self.count >= self.N

    def _get_ordered_buffer(self):
        """Return buffer in chronological order."""
        if self.count < self.N:
            return self.buf[:self.count]
        order = [(self.ptr + i) % self.N for i in range(self.N)]
        return self.buf[order]

    def denoise(self, grad_flat):
        """
        Denoise gradient using PCA projection.

        Returns denoised gradient of same shape as input.
        If buffer not full, returns input unchanged.
        """
        if not self.is_full:
            return grad_flat.copy()

        G = self._get_ordered_buffer()   # (N, D)

        # Center the buffer
        mean_grad = G.mean(axis=0)       # (D,)
        G_centered = G - mean_grad       # (N, D)

        # Gram matrix — O(N²D) instead of O(D³)
        K = G_centered @ G_centered.T   # (N, N)

        # SVD of Gram matrix
        try:
            U, S_sq, _ = np.linalg.svd(K)
        except np.linalg.LinAlgError:
            return grad_flat.copy()

        # Avoid sqrt of negative values from numerical noise
        S_sq = np.maximum(S_sq, 0)
        S    = np.sqrt(S_sq)

        # Top-k principal directions in parameter space
        # V_k shape: (D, k)
        k = min(self.k, np.sum(S > 1e-10))
        if k == 0:
            return grad_flat.copy()

        U_k = U[:, :k]                           # (N, k)
        S_k = S[:k]                              # (k,)
        V_k = G_centered.T @ U_k / (S_k + 1e-10) # (D, k)

        # Normalize columns of V_k
        norms = np.linalg.norm(V_k, axis=0, keepdims=True)
        V_k = V_k / (norms + 1e-10)

        # Project current gradient (centered) onto principal directions
        g_centered = grad_flat - mean_grad       # (D,)
        coords     = V_k.T @ g_centered          # (k,)

        # Apply scale weights — suppress noisy components
        w = self.scale_weights[:k]
        coords_denoised = coords * w             # (k,)

        # Reconstruct
        g_denoised = V_k @ coords_denoised + mean_grad  # (D,)

        return g_denoised
    
    
    def denoise_with_temporal(self, grad_flat, haar_weight=0.7):
        """Two-stage: PCA spatial + wavelet temporal denoising."""
        if not self.is_full:
            return grad_flat.copy()

        G = self._get_ordered_buffer()
        mean_grad  = G.mean(axis=0)
        G_centered = G - mean_grad
        K          = G_centered @ G_centered.T

        try:
            U, S_sq, _ = np.linalg.svd(K)
        except np.linalg.LinAlgError:
            return grad_flat.copy()

        S_sq = np.maximum(S_sq, 0)
        S    = np.sqrt(S_sq)
        k    = min(self.k, np.sum(S > 1e-10))
        if k == 0:
            return grad_flat.copy()

        U_k = U[:, :k]
        S_k = S[:k]
        V_k = G_centered.T @ U_k / (S_k + 1e-10)
        norms = np.linalg.norm(V_k, axis=0, keepdims=True)
        V_k = V_k / (norms + 1e-10)

        G_coords = G_centered @ V_k          # (N, k)
        g_centered = grad_flat - mean_grad
        raw_coords = V_k.T @ g_centered      # (k,)
        smooth_coords = np.zeros(k)

    
        for i in range(k):
            signal = G_coords[:, i]
            coeffs = haar_forward(signal)
            w = self.scale_weights[i] if i < len(self.scale_weights) else 0.5
            coeffs[1:] *= w
            smooth_coords[i] = haar_inverse(coeffs)[-1]

        coords_final = (1 - haar_weight)*raw_coords + haar_weight*smooth_coords
        return V_k @ coords_final + mean_grad
    

    def set_scale_weights(self, weights):
        """Update scale weights — called by MetaNet."""
        self.scale_weights = np.array(weights[:self.k])
        
        
