"""Create the paper-ready raw-image channel ablation figure."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "raw_image" / "channel_ablation_seed_results_final.csv"
OUTPUT = ROOT / "figures" / "fig_channel_ablation.png"

LABELS = {
    "6ch": "6 channels\n(all stains)",
    "3ch_dna_rna_mito": "3 channels\n(DNA, RNA, Mito)",
    "2ch_dna_mito": "2 channels*\n(DNA, Mito)",
}
ORDER = list(LABELS)
COLORS = ["#1f4e79", "#4c78a8", "#9db9d2"]


def summarize(metric: str):
    table = []
    for setting in ORDER:
        values = pd.read_csv(INPUT).query("setting == @setting")[metric].to_numpy()
        table.append((values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0, len(values)))
    return np.asarray(table)


def main():
    f1 = summarize("macro_f1")
    ba = summarize("balanced_accuracy")
    x = np.arange(len(ORDER))

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=False, constrained_layout=True)
    for ax, values, title in zip(axes, (f1, ba), ("Macro-F1", "Balanced accuracy")):
        bars = ax.bar(x, values[:, 0], yerr=values[:, 1], capsize=4, color=COLORS, edgecolor="black", linewidth=0.7)
        bars[2].set_hatch("//")
        for i, (mean, _std, n) in enumerate(values):
            ax.text(i, mean + 0.014, f"{mean:.3f}\n(n={int(n)})", ha="center", va="bottom", fontsize=8)
        ax.axhline(0.25, color="#666666", linestyle="--", linewidth=1, label="4-class chance")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x, [LABELS[item] for item in ORDER], fontsize=8)
        ax.set_ylim(0.20, 0.46)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Strict treatment-held-out score")
    axes[1].legend(frameon=False, loc="upper right", fontsize=8)
    fig.text(0.5, -0.03, "*2-channel setting was evaluated with one seed; error bars are omitted.", ha="center", fontsize=8)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
