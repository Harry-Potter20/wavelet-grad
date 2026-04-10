import os
import sys
import numpy as np
import pandas as pd

# --- Path setup ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from problems.xor_problem import xor_loss_and_grads_noisy, init_params, accuracy
from core.learned_threshold_correction import train_correction_model
from experiments.session11_adaptive import run_ema_visuShrink_adam
from experiments.session9_neural import run_xor_noisy_adam
from viz.plot_utils import plot_convergence, plot_scale_weight_evolution

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
N_TRIALS = 20         # number of seeds per optimizer
N_STEPS  = 2000
BUFFER_SIZE = 8
ALPHA = 1.0

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'validation')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# EMA weights for denoiser
decay = 1.0
EMA_WEIGHTS = np.array([decay**(BUFFER_SIZE-1-i) for i in range(BUFFER_SIZE)])
EMA_WEIGHTS /= EMA_WEIGHTS.sum()

# ---------------------------------------------------------
# HELPER: LearnedThreshold+Stable wrapper
# ---------------------------------------------------------
def run_learned_threshold_adam_wrapper(net, X_mean, X_std, sigma, seed):
    from experiments.session15_learned_denoiser import run_learned_threshold_adam
    return run_learned_threshold_adam(sigma, net, X_mean, X_std, n_steps=N_STEPS, seed=seed)


# ---------------------------------------------------------
# VALIDATION LOOP
# ---------------------------------------------------------
def validate():
    # 1️⃣ Train correction model
    print("Training correction model...")
    net, X_mean, X_std = train_correction_model(n_samples=10000, n_epochs=50)

    # 2️⃣ Noise levels
    sigma_levels = [0.0, 0.05, 0.1, 0.2]

    # 3️⃣ Collect results
    all_stats = []
    all_histories = {}

    for sigma in sigma_levels:
        print(f"\n=== Noise σ={sigma} ===")
        optimizers = {
            'Adam': lambda s: run_xor_noisy_adam(sigma, N_STEPS, seed=s),
            'EMAVisuShrink': lambda s: run_ema_visuShrink_adam(sigma, N_STEPS, seed=s),
            'LearnedThreshold+Stable': lambda s: run_learned_threshold_adam_wrapper(net, X_mean, X_std, sigma, s)
        }

        for name, fn in optimizers.items():
            print(f"Running {name}...")
            final_losses = []
            solved_flags  = []
            histories     = []

            for seed in range(N_TRIALS):
                h, p = fn(seed)
                final_losses.append(h[-1][1])
                solved_flags.append(accuracy(p) == 1.0)
                histories.append(h)

            # Save stats
            for seed, loss, solved in zip(range(N_TRIALS), final_losses, solved_flags):
                all_stats.append({
                    'sigma': sigma,
                    'optimizer': name,
                    'seed': seed,
                    'final_loss': loss,
                    'solved': solved
                })

            # Save histories for convergence plotting
            all_histories[f"{name}_σ{sigma}"] = histories

            print(f"{name}: mean loss={np.mean(final_losses):.6f}, solved {sum(solved_flags)}/{N_TRIALS}")

    # 4️⃣ Save CSV stats
    df_stats = pd.DataFrame(all_stats)
    csv_path = os.path.join(OUTPUT_DIR, "validation_stats.csv")
    df_stats.to_csv(csv_path, index=False)
    print(f"Saved stats CSV: {csv_path}")

    # 5️⃣ Plot convergence per sigma
    for sigma in sigma_levels:
        histories_list = []
        labels = []
        for name in optimizers:
            histories = all_histories[f"{name}_σ{sigma}"]
            # Average losses across trials
            avg_loss = np.mean([[l for _, _, l in h] for h in histories], axis=0)
            # Reformat to [(step, avg_loss, _)]
            histories_list.append([(i+1, l, 0) for i, l in enumerate(avg_loss)])
            labels.append(name)
        plot_convergence(histories_list, labels, title=f"Convergence σ={sigma}",
                         filename=f"convergence_sigma{sigma}.png")

    # 6️⃣ Optional: plot scale weight evolution for LearnedThreshold+Stable
    try:
        from experiments.session15_learned_denoiser import EMA_WEIGHTS
        # If you have saved weight histories from training the correction model, plot here
        # plot_scale_weight_evolution(weight_history, filename="scale_weights_validation.png")
    except Exception as e:
        print("Skipping scale weight plot:", e)


if __name__ == "__main__":
    validate()