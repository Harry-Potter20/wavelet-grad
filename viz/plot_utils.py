import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — works without display
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_convergence(histories, labels, title="Convergence", filename="convergence.png"):
    """
    Plot loss curves for multiple optimizers.

    Args:
        histories: list of history lists, each [(x,y,loss), ...]
        labels:    list of strings, one per history
        title:     plot title
        filename:  output filename in outputs/
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor('#0f0f0f')
    fig.patch.set_facecolor('#0f0f0f')

    colors = ['#00ff88', '#ff6b35', '#4ecdc4', '#ffe66d', '#a8dadc']

    for i, (history, label) in enumerate(zip(histories, labels)):
        losses = [l for _, _, l in history]
        # Smooth with moving average for readability
        k = 20
        smoothed = np.convolve(losses, np.ones(k)/k, mode='valid')
        color = colors[i % len(colors)]
        ax.plot(losses, alpha=0.15, color=color, linewidth=0.5)
        ax.plot(range(k-1, len(losses)), smoothed,
                color=color, linewidth=2, label=label)

    ax.set_yscale('log')
    ax.set_xlabel('Step', color='#888888')
    ax.set_ylabel('Loss (log scale)', color='#888888')
    ax.set_title(title, color='#ffffff', fontsize=14, pad=15)
    ax.tick_params(colors='#888888')
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#1a1a1a', edgecolor='#333333',
              labelcolor='#cccccc', fontsize=10)

    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_noise_comparison(sigma_levels, adam_losses, wavelet_losses, meta_losses,
                           filename="noise_comparison.png"):
    """
    Bar chart comparing final losses across noise levels.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor('#0f0f0f')
    fig.patch.set_facecolor('#0f0f0f')

    x = np.arange(len(sigma_levels))
    width = 0.25

    bars_adam    = ax.bar(x - width, adam_losses,    width,
                          label='Adam',         color='#00ff88', alpha=0.85)
    bars_wavelet = ax.bar(x,         wavelet_losses, width,
                          label='StaticWavelet', color='#ff6b35', alpha=0.85)
    bars_meta    = ax.bar(x + width, meta_losses,    width,
                          label='MetaWavelet',  color='#4ecdc4', alpha=0.85)

    ax.set_xlabel('Noise Level (σ)', color='#888888')
    ax.set_ylabel('Final Loss', color='#888888')
    ax.set_title('Final Loss vs Noise Level', color='#ffffff', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f'σ={s}' for s in sigma_levels], color='#888888')
    ax.tick_params(colors='#888888')
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#1a1a1a', edgecolor='#333333',
              labelcolor='#cccccc', fontsize=10)
    ax.set_facecolor('#0f0f0f')

    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_gradient_signal(sig_x, coeffs_x, weight_vec, filename="gradient_signal.png"):
    """
    Three-panel plot showing:
    1. Raw gradient signal in buffer
    2. Wavelet coefficients
    3. Scale weights applied
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('#0f0f0f')

    panels = [
        (axes[0], sig_x,     '#00ff88', 'Raw Gradient Buffer'),
        (axes[1], coeffs_x,  '#ff6b35', 'Wavelet Coefficients'),
        (axes[2], weight_vec,'#4ecdc4', 'Scale Weights'),
    ]

    for ax, data, color, title in panels:
        ax.set_facecolor('#0f0f0f')
        ax.bar(range(len(data)), data, color=color, alpha=0.8)
        ax.set_title(title, color='#ffffff', fontsize=11)
        ax.tick_params(colors='#888888')
        ax.spines['bottom'].set_color('#333333')
        ax.spines['left'].set_color('#333333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(0, color='#444444', linewidth=0.5)

    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_scale_weight_evolution(weight_history, filename="scale_weights.png"):
    """
    Plot how MetaNet scale weights evolve during training.

    Args:
        weight_history: list of (step, [w1, w2, w3]) tuples
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_facecolor('#0f0f0f')
    fig.patch.set_facecolor('#0f0f0f')

    steps = [s for s, _ in weight_history]
    w1 = [w[0] for _, w in weight_history]
    w2 = [w[1] for _, w in weight_history]
    w3 = [w[2] for _, w in weight_history]

    ax.plot(steps, w1, color='#00ff88', linewidth=2, label='scale3 (coarsest)')
    ax.plot(steps, w2, color='#ff6b35', linewidth=2, label='scale2 (mid)')
    ax.plot(steps, w3, color='#4ecdc4', linewidth=2, label='scale1 (finest)')
    ax.axhline(0.5, color='#444444', linewidth=1, linestyle='--', label='neutral (0.5)')

    ax.set_ylim(0, 1)
    ax.set_xlabel('Training Step', color='#888888')
    ax.set_ylabel('Scale Weight', color='#888888')
    ax.set_title('MetaNet Scale Weight Evolution', color='#ffffff', fontsize=14, pad=15)
    ax.tick_params(colors='#888888')
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#1a1a1a', edgecolor='#333333',
              labelcolor='#cccccc', fontsize=10)

    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")
    return path