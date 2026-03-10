import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problems.rosenbrock import rosenbrock
from core.optimizers import run_adam, run_adam_instrumented, run_sgd, run_wavelet_sgd
from core.optimizers import run_wavelet_adam
from core.wavelets import haar_forward, haar_inverse, wavelet_denoise


# --- Step 1: Run instrumented Adam ---
history, grad_log, snapshots = run_adam_instrumented(
    lr=0.01, n_steps=200, buffer_capacity=8
)

print(f"Steps run: {len(history)}")
print(f"grad_log shape: {grad_log.shape}")
print(f"First snapshot at step: {next(i for i,s in enumerate(snapshots) if s is not None)}")

# --- Step 2: Inspect one buffer snapshot ---
# Find first valid snapshot
first_idx = next(i for i, s in enumerate(snapshots) if s is not None)
snap_x, snap_y = snapshots[first_idx]

print(f"\nBuffer snapshot at step {first_idx}:")
print(f"  gx over last 8 steps: {np.round(snap_x, 4)}")
print(f"  gy over last 8 steps: {np.round(snap_y, 4)}")

# --- Step 3: Apply Haar to the snapshot ---
coeffs_x = haar_forward(snap_x)
coeffs_y = haar_forward(snap_y)

print(f"\nHaar coefficients (gx signal):")
print(f"  {np.round(coeffs_x, 4)}")
print(f"\nHaar coefficients (gy signal):")
print(f"  {np.round(coeffs_y, 4)}")

print(f"\nCoefficient layout for length-8 signal:")
print(f"  coeffs[0]      = coarsest approximation (mean trend)")
print(f"  coeffs[1]      = scale-3 detail")
print(f"  coeffs[2:4]    = scale-2 detail")
print(f"  coeffs[4:8]    = scale-1 detail (finest, most noise)")


# Apply to the snapshot
threshold = 1.0

denoised_gx, coeffs_gx, thresh_gx = wavelet_denoise(snap_x, threshold)
denoised_gy, coeffs_gy, thresh_gy = wavelet_denoise(snap_y, threshold)

print(f"\n--- Denoising at threshold={threshold} ---")
print(f"\ngx signal:         {np.round(snap_x, 4)}")
print(f"raw coefficients:  {np.round(coeffs_gx, 4)}")
print(f"after threshold:   {np.round(thresh_gx, 4)}")
print(f"raw gradient:      {snap_x[-1]:.4f}")
print(f"denoised gradient: {denoised_gx:.4f}")

print(f"\ngy signal:         {np.round(snap_y, 4)}")
print(f"raw coefficients:  {np.round(coeffs_gy, 4)}")
print(f"after threshold:   {np.round(thresh_gy, 4)}")
print(f"raw gradient:      {snap_y[-1]:.4f}")
print(f"denoised gradient: {denoised_gy:.4f}")


# --- Step 5: Compare WaveletSGD vs SGD vs Adam ---
from core.optimizers import run_sgd, run_wavelet_sgd

print("\n--- Convergence comparison ---")

# SGD baseline
h_sgd = run_sgd(lr=0.001, n_steps=5000)
last = h_sgd[-1]
print(f"SGD          lr=0.001  5000 steps → loss={last[2]:.6f}")

# Adam baseline — we know this converges by step 2000
h_adam = run_adam(lr=0.01, n_steps=5000)
last = h_adam[-1]
converged = next((i for i,(x,y,l) in enumerate(h_adam)
                  if abs(x-1)<1e-4 and abs(y-1)<1e-4), None)
print(f"Adam         lr=0.01   5000 steps → loss={last[2]:.6f}  converged={converged}")

# WaveletSGD — try a few threshold values
for thresh in [0.5, 1.0, 2.0]:
    h_wav = run_wavelet_sgd(lr=0.001, n_steps=5000,
                            buffer_capacity=8, threshold=thresh)
    last = h_wav[-1]
    converged_w = next((i for i,(x,y,l) in enumerate(h_wav)
                        if abs(x-1)<1e-4 and abs(y-1)<1e-4), None)
    print(f"WaveletSGD   lr=0.001  thresh={thresh}  "
          f"→ loss={last[2]:.6f}  converged={converged_w}")
    

print("\n--- WaveletAdam vs Adam ---")

h_adam = run_adam(lr=0.01, n_steps=2000)
converged_adam = next((i for i,(x,y,l) in enumerate(h_adam)
                       if abs(x-1)<1e-4 and abs(y-1)<1e-4), None)
print(f"Adam          converged={converged_adam}")

for thresh in [0.5, 1.0, 2.0]:
    for cap in [8, 16]:
        h_wav = run_wavelet_adam(lr=0.01, n_steps=2000,
                                 buffer_capacity=cap, threshold=thresh)
        converged_w = next((i for i,(x,y,l) in enumerate(h_wav)
                            if abs(x-1)<1e-4 and abs(y-1)<1e-4), None)
        last = h_wav[-1]
        print(f"WaveletAdam   thresh={thresh}  cap={cap}  "
              f"→ loss={last[2]:.6f}  converged={converged_w}")
        
h = run_wavelet_adam(lr=0.01, n_steps=5000,
                     buffer_capacity=8, threshold=0.5)
converged = next((i for i,(x,y,l) in enumerate(h)
                  if abs(x-1)<1e-4 and abs(y-1)<1e-4), None)
last = h[-1]
print(f"WaveletAdam thresh=0.5 cap=8 5000 steps → "
      f"loss={last[2]:.6f}  converged={converged}")