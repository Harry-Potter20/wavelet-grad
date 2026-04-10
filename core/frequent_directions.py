import numpy as np

# =============================================================
# FREQUENT DIRECTIONS — ONLINE LOW-RANK GRADIENT BUFFER (FIXED)
#
# Fixes applied:
#   1. Noise estimated from singular values (not coords)
#   2. Removed orthogonal noise injection
#   3. Correct VisuShrink threshold uses log(D), not log(k)
# =============================================================


class FrequentDirections:
    def __init__(self, sketch_size=16, D=None):
        assert sketch_size >= 2, "sketch_size must be at least 2"
        self.k   = sketch_size
        self.D   = D
        self.ptr = 0
        self.n   = 0
        self._Vt    = None
        self._sigma = None
        # Initialise sketch immediately if D is known
        if D is not None:
            self.S = np.zeros((2 * self.k, D))
        else:
            self.S = None
    def _init_sketch(self, D):
        self.D = D
        self.S = np.zeros((2 * self.k, D))

    def update(self, grad_flat):
        if self.D is None:
            self._init_sketch(len(grad_flat))

        self.S[self.ptr] = grad_flat
        self.ptr += 1
        self.n   += 1

        if self.ptr == 2 * self.k:
            self._compress()

    def _compress(self):
        try:
            U, sigma, Vt = np.linalg.svd(self.S, full_matrices=False)
        except np.linalg.LinAlgError:
            self.S[:] = 0
            self.ptr = 0
            return

        # Number of singular values returned is min(2k, D)
        n_sv = len(sigma)
        k    = min(self.k, n_sv - 1)   # need at least k+1 to do bias correction
        if k < 1:
            self.S[:] = 0
            self.ptr = 0
            return

        sigma_k = sigma[:k]
        Vt_k    = Vt[:k]

        delta = sigma[k]**2 if n_sv > k else 0.0
        sigma_corrected = np.sqrt(np.maximum(sigma_k**2 - delta, 0))

        self._sigma = sigma_corrected
        self._Vt    = Vt_k

        self.S[:k] = np.diag(sigma_corrected) @ Vt_k
        self.S[k:] = 0
        self.ptr = k
        
    @property
    def is_ready(self):
        return self._Vt is not None

    def denoise(self, grad_flat):
        """
        Project gradient onto top-k FD directions.
        No thresholding — FD identifies the signal subspace,
        Adam's momentum handles noise smoothing.
        """
        if not self.is_ready:
            return grad_flat.copy()

        Vt = self._Vt   # (k, D)

        # Project onto signal subspace and reconstruct
        # This removes components orthogonal to learned directions
        coords     = Vt @ grad_flat        # (k,)
        g_projected = Vt.T @ coords        # (D,) — low-rank projection

        # Blend with raw gradient — keep projection dominant
        # but don't completely discard orthogonal components
        alpha = 0.9
        return alpha * g_projected + (1 - alpha) * grad_flat
        
    def get_covariance_approx(self):
        if not self.is_ready:
            return None
        return self._Vt.T @ np.diag(self._sigma**2) @ self._Vt

    def memory_bytes(self):
        return self.S.nbytes if self.S is not None else 0

    def memory_vs_full_buffer(self, N=8):
        fd_size   = 2 * self.k * self.D
        full_size = N * self.D
        return fd_size / full_size