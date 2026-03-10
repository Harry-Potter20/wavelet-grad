# wavelet-grad

**Multi-scale wavelet gradient denoising for neural network optimizers.**

WaveletAdam outperforms Adam under gradient noise by separating signal from noise across frequency scales — the same principle used in signal processing, applied to the optimizer update step.

---

## Core Result

On a 2→4→1 XOR network with injected gradient noise:

| Noise (σ) | Adam | WaveletAdam | PCAWavelet-tuned |
|-----------|------|-------------|-----------------|
| 0.00 | 9/10 solved | 9/10 solved | 9/10 solved |
| 0.05 | 7/10 solved | **10/10 solved** | **10/10 solved** |
| 0.10 | 1/10 solved | **9/10 solved** | 7/10 solved |
| 0.20 | 0/10 solved | **2/10 solved** | 2/10 solved |

At σ=0.05, WaveletAdam solves XOR 10/10 vs Adam's 7/10 with 6× lower final loss.

---

## How It Works

Standard Adam smooths gradients with exponential moving averages at a single fixed timescale (β₁=0.9). This averages ~10 recent gradients — adequate for mild noise but blind to frequency structure.

WaveletAdam applies a Haar wavelet transform to the circular gradient buffer before the Adam update:

```
gradient buffer (8 steps)
        ↓
   Haar forward transform
        ↓
   scale weights × coefficients     ← learned or hand-tuned
        ↓
   Haar inverse transform
        ↓
   denoised gradient → Adam update
```

The Haar transform decomposes the gradient signal into frequency scales. High-frequency coefficients (scale1) capture rapid gradient changes — mostly noise. Low-frequency coefficients (scale3) capture slow drift — mostly signal. Applying learned scale weights suppresses noise selectively.

---

## Project Structure

```
wavelet-grad/
├── core/
│   ├── buffers.py          # CircularGradientBuffer — O(1) writes, power-of-2
│   ├── wavelets.py         # Haar transform, analytical gradients, IDWT cache
│   ├── optimizers.py       # SGD, Adam, WaveletAdam, MetaWaveletAdam
│   ├── meta_net.py         # MetaNet — learns scale weights from state features
│   └── pca_denoiser.py     # PCAGradientDenoiser — Gram matrix joint denoising
├── problems/
│   ├── rosenbrock.py       # Clean Rosenbrock (2D)
│   ├── noisy_rosenbrock.py # Rosenbrock + Gaussian gradient noise
│   └── xor_problem.py      # XOR neural network (2→4→1)
├── experiments/
│   ├── session1_basics.py        # SGD, Adam baseline
│   ├── session2_3_wavelets.py    # Haar transform fundamentals
│   ├── session4_learned.py       # Static learned scale weights
│   ├── session5_meta.py          # Online MetaNet (numerical gradient)
│   ├── session6_analytical.py    # Analytical meta-gradient derivation
│   ├── session7_noisy.py         # Noise robustness experiments
│   ├── session8_meta_noisy.py    # Offline MetaNet training
│   ├── session9_neural.py        # XOR neural network benchmark
│   └── session10_pca.py          # PCA joint gradient denoising
├── tests/
│   ├── test_buffers.py       # 6 tests
│   ├── test_wavelets.py      # 11 tests
│   ├── test_optimizers.py    # 11 tests
│   ├── test_meta_net.py      # 6 tests
│   └── test_pca_denoiser.py  # 9 tests
└── viz/
    └── plot_utils.py         # Convergence curves, noise comparison charts
```

---

## Optimizers

### WaveletAdam
Static scale weights applied via Haar transform. Best results with aggressive fine-scale suppression:
```python
from core.optimizers import run_noisy_wavelet_adam
history, params = run_noisy_wavelet_adam(
    sigma=0.05, n_steps=2000, seed=0,
    raw_params=np.array([1.0, 0.0, -1.0])  # coarse→keep, fine→suppress
)
```

### PCAGradientDenoiser
Joint denoising of the full parameter gradient vector using Gram matrix PCA. Captures inter-parameter correlations that per-element denoising misses. Scales to large networks — Gram matrix is always N×N (buffer size), never D×D (parameter count):
```python
from core.pca_denoiser import PCAGradientDenoiser
denoiser = PCAGradientDenoiser(
    n_components=4, buffer_size=8,
    scale_weights=np.array([1.0, 0.1, 0.0, 0.0])
)
denoiser.update(grad_flat)
clean_grad = denoiser.denoise(grad_flat)
```

### MetaWaveletAdam
MetaNet (5→16→3) predicts scale weights from training state features: step, loss, gradient norm, distance to optimum, gradient variance. Trained offline via discrete candidate search + MSE supervision:
```python
from experiments.session8_meta_noisy import train_offline, run_noisy_meta_wavelet_adam
net = train_offline(sigmas=[0.0, 0.5, 1.0, 2.0, 5.0], n_epochs=100)
history, _ = run_noisy_meta_wavelet_adam(sigma=1.0, net=net, train=False)
```

---

## Installation

```bash
git clone https://github.com/chuksie45/wavelet-grad
cd wavelet-grad
pip install -r requirements.txt
```

## Running Tests

```bash
python -m pytest tests/          # all 45 tests
python tests/test_optimizers.py  # individual files
```

## Running Experiments

```bash
python experiments/session9_neural.py    # XOR benchmark
python experiments/session10_pca.py     # PCA comparison
```

---

## Key Concepts

**Why wavelets?** Adam's EMA averages ~10 steps uniformly. The Haar wavelet decomposes those same 10 steps into frequency bands — coarse (slow trend), medium, fine (rapid changes). Noise lives in fine bands. Signal lives in coarse bands. Applying different weights per band is strictly more expressive than uniform averaging.

**Why Gram matrix PCA?** For a network with D parameters and buffer size N, the covariance matrix is D×D — intractable at scale. The Gram matrix K = G·Gᵀ is N×N, always 8×8 for our buffer. The top-k eigenvectors of K give the same principal directions as the full covariance decomposition, at O(N²D) cost instead of O(D³).

**Why offline training?** Online meta-learning with windowed loss has a fundamental credit assignment problem — scale weights from 20 steps ago affected the current loss window. Offline training collects (state, optimal_weights) pairs by trying candidate weight configs at each step and supervising the MetaNet with MSE. Clean training signal, better convergence.

---

## Roadmap

- [ ] Adaptive denoising (VisuShrink — estimate noise level online)
- [ ] Exponential moving buffer (recency-weighted gradient history)
- [ ] Per-layer denoising for MNIST/CIFAR-10
- [ ] Conditional PCA + wavelet (SNR-adaptive per component)
- [ ] Learned noise model (input: gradient distribution statistics)

---

## License

MIT
