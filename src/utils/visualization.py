import matplotlib.pyplot as plt
import numpy as np


def plot_history(history, title='Training History', ax=None):
    """Plot train and val loss curves from a history dict."""
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(history['train_loss'], label='train loss')
    if history.get('val_loss'):
        ax.plot(history['val_loss'], label='val loss')

    ax.set_title(title)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()


def compare_histories(histories, labels, title='Comparison', figsize=(10, 4)):
    """
    Overlay multiple training histories on a single plot.

    Args:
        histories : list of history dicts (from train())
        labels    : list of str, one label per history
        title     : plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for history, label in zip(histories, labels):
        axes[0].plot(history['train_loss'], label=label)
        if history.get('val_loss'):
            axes[1].plot(history['val_loss'], label=label)

    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Validation Loss')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def bar_comparison(labels, values, title='', ylabel='Accuracy', figsize=(8, 4)):
    """Bar chart for comparing a single metric across experiments."""
    fig, ax = plt.subplots(figsize=figsize)
    colors  = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    bars    = ax.bar(labels, values, color=colors, edgecolor='white')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=9)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.show()
