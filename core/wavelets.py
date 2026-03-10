import numpy as np

# HAAR WAVELET TRANSFORM
# The Haar transform is the simplest wavelet transform.
# It operates on a 1D signal of length N (must be power of 2).
#
# One level of decomposition:
#   approximation[i] = (signal[2i] + signal[2i+1]) / sqrt(2)
#   detail[i]        = (signal[2i] - signal[2i+1]) / sqrt(2)
#
# The 1/sqrt(2) factor preserves signal energy (L2 norm).
# Without it, the transform wouldn't be orthonormal.
#
# Full decomposition: recurse on approximation until length 1.
# =============================================================

def haar_forward(signal):
    """
    Forward Haar wavelet transform.

    Args:
        signal: 1D numpy array, length must be a power of 2

    Returns:
        coeffs: 1D array of same length, organized as:
                [coarsest_approx, coarse_detail, ..., fine_detail]

    Example:
        signal = [4, 6, 10, 2]
        coeffs = [5.5, -0.5, -1, 4]  (scaled by 1/sqrt(2) factors)
    """
    n = len(signal)
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError(f"Signal length must be a power of 2, got {n}")

    coeffs = signal.astype(float).copy()
    length = n

    while length > 1:
        half = length // 2
        temp = np.zeros(length)

        for i in range(half):
            # Approximation: average (scaled)
            temp[i]        = (coeffs[2*i] + coeffs[2*i+1]) / np.sqrt(2)
            # Detail: difference (scaled)
            temp[half + i] = (coeffs[2*i] - coeffs[2*i+1]) / np.sqrt(2)

        coeffs[:length] = temp
        length = half   # recurse on approximation half

    return coeffs

# HAAR INVERSE: 
# Basically the same as forward, but we reconstruct the signal from the coefficients.
     

def haar_inverse(coeffs):
    """
    Inverse Haar wavelet transform.

    Args:
        coeffs: 1D numpy array of Haar coefficients

    Returns:
        signal: 1D array, reconstructed signal
    """
    n = len(coeffs)
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError(f"Coefficients length must be a power of 2, got {n}")

    result = coeffs.astype(float).copy()
    length = 2 # start at coarsest level, work outward

    while length <= n:
        half = length // 2
        temp = np.zeros(length)

        for i in range(half):
            # Inverse detail: add (scaled)
            temp[2*i]     = (result[i] + result[half + i]) / np.sqrt(2)
            # Inverse approximation: subtract (scaled)
            temp[2*i + 1] = (result[i] - result[half + i]) / np.sqrt(2)

        result[:length] = temp
        length *= 2 

    return result

# Sanity Check

if __name__ == "__main__":
    # Test signal
    s = np.array([4.0, 6.0, 10.0, 2.0])
    c = haar_forward(s)
    print(f"Input signal: {s}")
    print(f"Haar coefficients: {np.round(c, 4)}")
    
    # Inverse transform
    reconstructed = haar_inverse(c)
    print(f"Reconstructed signal: {np.round(reconstructed, 4)}")
    # Check if the reconstruction is close to the original
    print(f"Reconstruction error: {np.round(np.linalg.norm(s - reconstructed), 4)}")
    # If reconstruction is perfect
    print(f"Reconstruction perfect? {np.allclose(s, reconstructed)}")
    
    # Show what each coefficient means
    print(f"\nCoefficient interpretation:")
    print(f" coeffs[0]: {c[0]:.4f} (coarsest approximation, (mean * sqrt(N)))")
    print(f" coeffs[1]: {c[1]:.4f} (coarse detail) (scale 2)")
    print(f" coeffs[2]: {c[2]:.4f} (fine detail) (scale 1, first half)")
    print(f" coeffs[3]: {c[3]:.4f} (finest detail) (scale 1, second half)")

    # Energy preservation check
    energy_input = np.sum(s**2)
    energy_output = np.sum(c**2)
    print(f"Energy (input): {energy_input:.4f}")
    print(f"Energy (output): {energy_output:.4f}")
    print(f"Energy preserved? {np.isclose(energy_input, energy_output)}")
    
    # --- Step 4: Threshold detail coefficients ---
# The idea: zero out small detail cofficients (that's noise)
# Keep the approx. coefficient (True signal trend)
# Reconstruct - Result is a denoised gradient signal

def wavelet_denoise(signal, threshold):
    '''
    Apply Haar wavelet thresholding to denoise the signal.
    
    Essentially:
    Step 1: Apply Haar transform to get coefficients
    Step 2: Zero out small detail coefficients
    Step 3: Apply inverse Haar transform to reconstruct signal
    Step 4: Return the last value - Denoised gradient for this step
    '''
    coeffs = haar_forward(signal)
    
    #coeffs[0] - The approx., do not threshold
    #coeffs[1:] - The details, apply thresholding
    thresholded = coeffs.copy()
    # Zero out detail coefficients below the threshold
    # coeffs[0] - The approx., do not threshold
    detail_mask = np.abs(thresholded[1:]) <= threshold
    thresholded[1:][detail_mask] = 0.0
    denoised_signal = haar_inverse(thresholded)

    #Return the last value - Denoised gradient for this step
    return denoised_signal[-1], coeffs, thresholded

# =============================================================
# LEARNED SOFT WAVELET DENOISING
# =============================================================
#
# Replace the hard binary threshold with learned per-scale weights.
#
# For a signal of length N = 2^J, we have J+1 groups of coefficients:
#   group 0: coeffs[0]          (approximation)
#   group 1: coeffs[1]          (scale J detail)
#   group 2: coeffs[2:4]        (scale J-1 detail)
#   ...
#   group J: coeffs[N//2:]      (scale 1 detail, finest)
#
# Each group gets one learned weight in [0, 1].
# We enforce [0,1] via sigmoid: weight = sigmoid(raw_param)
#
# The approximation weight (group 0) is always fixed at 1.0
# — we never attenuate the mean trend.
# =============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def make_scale_weight_vector(raw_params, n):
    """
    Convert raw learnable parameters into a weight vector of length n.

    Args:
        raw_params: array of shape (J,) — one raw param per detail scale
                    (NOT including the approximation, which is always 1.0)
        n:          signal length (power of 2)

    Returns:
        weights: array of shape (n,) where:
                 weights[0]      = 1.0  (approximation, fixed)
                 weights[1]      = sigmoid(raw_params[0])  (coarsest detail)
                 weights[2:4]    = sigmoid(raw_params[1])  (next scale)
                 ...
                 weights[n//2:]  = sigmoid(raw_params[-1]) (finest detail)

    Example (n=8, J=3):
        raw_params = [p0, p1, p2]
        weights    = [1.0, s(p0), s(p1), s(p1), s(p2), s(p2), s(p2), s(p2)]
    """
    import math
    J = int(math.log2(n))
    assert len(raw_params) == J, f"Need {J} params for n={n}, got {len(raw_params)}"

    weights = np.ones(n)
    # Fill each scale group
    # Scale group k (k=0 is coarsest detail, k=J-1 is finest):
    #   starts at index 2^k, has 2^k entries
    for k in range(J):
        start = 2**k
        end   = 2**(k+1)
        weights[start:end] = sigmoid(raw_params[k])

    return weights


def soft_wavelet_denoise(signal, raw_params):
    """
    Denoise using learned per-scale soft weights.

    Args:
        signal:     1D array, length power of 2
        raw_params: learnable parameters, shape (J,) where J=log2(len(signal))

    Returns:
        denoised_value: scalar — last value of reconstructed signal
        coeffs:         raw Haar coefficients
        weighted_coeffs: coefficients after applying scale weights
        weights:        the actual weight vector used
    """
    coeffs = haar_forward(signal)
    weights = make_scale_weight_vector(raw_params, len(signal))
    weighted_coeffs = coeffs * weights
    denoised_signal = haar_inverse(weighted_coeffs)
    return denoised_signal[-1], coeffs, weighted_coeffs, weights


_IDWT_LAST_ROW_CACHE = {}

def get_idwt_last_row(n):
    """Cache the last row of the IDWT matrix for a given length."""
    if n not in _IDWT_LAST_ROW_CACHE:
        identity = np.eye(n)
        idwt_matrix = np.array([haar_inverse(identity[j]) for j in range(n)])
        _IDWT_LAST_ROW_CACHE[n] = idwt_matrix[:, -1]
    return _IDWT_LAST_ROW_CACHE[n]

def dgx_clean_dweightvec(sig_x, weight_vec):
    """
    Gradient of gx_clean w.r.t. each element of weight_vec.
    Loop version — used for testing against fast version.

    Returns array of shape (n,) where entry i is
    d(gx_clean)/d(weight_vec[i]).
    """
    coeffs = haar_forward(sig_x)
    n = len(sig_x)
    grad = np.zeros(n)

    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = coeffs[i]
        grad[i] = haar_inverse(e_i)[-1]

    return grad

def dgx_clean_dweightvec_fast(sig_x):
    coeffs   = haar_forward(sig_x)
    last_row = get_idwt_last_row(len(sig_x))
    return last_row * coeffs
