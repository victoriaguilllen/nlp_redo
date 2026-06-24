"""
Genera los 3 plots del informe como PNG de alta resolución.
Ejecutar: python3.11 generate_plots.py
Salida:   figures/fig1_metrics.png
          figures/fig2_human_eval.png
          figures/fig3_throughput.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Paleta de colores consistente en todo el informe ─────────────────────────
C_GRAY   = "#7f7f7f"
C_BLUE   = "#1f77b4"
C_ORANGE = "#ff7f0e"
C_GREEN  = "#2ca02c"
C_RED    = "#d62728"

FONT = "DejaVu Sans"
plt.rcParams.update({
    "font.family":       FONT,
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
})


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 1 — Mejora relativa en métricas automáticas
# ══════════════════════════════════════════════════════════════════════════════
def fig1_metrics():
    metrics   = ["BLEU", "ROUGE-L", "BERTScore ×10"]
    tf_vals   = [39.5,   16.6,      5.0]
    uns_vals  = [26.9,   16.4,      4.6]

    x   = np.arange(len(metrics))
    w   = 0.32
    fig, ax = plt.subplots(figsize=(7, 4.2))

    bars1 = ax.bar(x - w/2, tf_vals,  w, label="Transformers LoRA",
                   color=C_BLUE,   alpha=0.88, zorder=3)
    bars2 = ax.bar(x + w/2, uns_vals, w, label="Unsloth LoRA (MLX)",
                   color=C_ORANGE, alpha=0.88, zorder=3)

    # value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=9, color=C_BLUE, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=9, color=C_ORANGE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Relative improvement over baseline (%)")
    ax.set_ylim(0, 47)
    ax.legend(loc="upper right")
    ax.set_title("Figure 1 — Automatic Metric Improvement after LoRA Fine-tuning\n"
                 "(BERTScore ×10 for scale visibility)", pad=8)

    plt.tight_layout()
    out = FIG_DIR / "fig1_metrics.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 2 — Evaluación humana por criterio y modelo
# ══════════════════════════════════════════════════════════════════════════════
def fig2_human_eval():
    criteria = ["Helpfulness", "Factuality", "Instruction-\nFollowing"]
    orig_scores = [2.8, 3.4, 2.2]
    tf_scores   = [2.6, 3.2, 3.6]
    uns_scores  = [2.4, 2.6, 3.0]

    x = np.arange(len(criteria))
    w = 0.25

    fig, ax = plt.subplots(figsize=(7, 4.5))

    b1 = ax.bar(x - w, orig_scores, w, label="Original (0-shot)",
                color=C_GRAY,   alpha=0.88, zorder=3)
    b2 = ax.bar(x,     tf_scores,   w, label="Transformers LoRA",
                color=C_BLUE,   alpha=0.88, zorder=3)
    b3 = ax.bar(x + w, uns_scores,  w, label="Unsloth LoRA (MLX)",
                color=C_ORANGE, alpha=0.88, zorder=3)

    for bars, color in [(b1, C_GRAY), (b2, C_BLUE), (b3, C_ORANGE)]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.04,
                    f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=9,
                    color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(criteria, ha="center")
    ax.set_ylabel("Mean score (1–5)")
    ax.set_ylim(0, 5.3)
    ax.axhline(3, color="gray", linestyle=":", linewidth=0.8, alpha=0.6, zorder=2)
    ax.legend(loc="upper left")
    ax.set_title("Figure 2 — Human Evaluation: Mean Score per Criterion\n"
                 "(5 prompts × 3 criteria; dotted line = mid-scale)", pad=8)

    plt.tight_layout()
    out = FIG_DIR / "fig2_human_eval.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 3 — Throughput de inferencia por framework
# ══════════════════════════════════════════════════════════════════════════════
def fig3_throughput():
    frameworks  = ["vLLM\n(CPU — macOS)", "Transformers\n(PyTorch MPS)",
                   "Unsloth\n(mlx-lm)", "Ollama\n(llama.cpp Metal)"]
    throughputs = [30.0, 66.6, 106.7, 268.4]
    stds        = [0,    2.8,  3.6,   1.0]
    colors      = [C_RED, C_GRAY, C_ORANGE, C_BLUE]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    y = np.arange(len(frameworks))
    bars = ax.barh(y, throughputs, xerr=stds,
                   color=colors, alpha=0.88, height=0.55,
                   error_kw=dict(elinewidth=1.4, capsize=4, ecolor="black"),
                   zorder=3)

    for i, (val, std) in enumerate(zip(throughputs, stds)):
        label = f"{val:.1f} tok/s" if std == 0 else f"{val:.1f} ± {std:.1f} tok/s"
        ax.text(val + 4, i, label, va="center", fontsize=9.5, fontweight="bold",
                color=colors[i])

    ax.set_yticks(y)
    ax.set_yticklabels(frameworks)
    ax.set_xlabel("Throughput (tokens / second)")
    ax.set_xlim(0, 320)
    ax.set_title("Figure 3 — Inference Throughput on Apple M4 Pro\n"
                 "(mean ± std, 3 trials × 5 prompts; temperature=0, max_tokens=256)",
                 pad=8)

    # annotation for vLLM
    ax.annotate("No GPU support\non macOS",
                xy=(30, 0), xytext=(100, 0.4),
                fontsize=8.5, color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2))

    plt.tight_layout()
    out = FIG_DIR / "fig3_throughput.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generando figuras...")
    fig1_metrics()
    fig2_human_eval()
    fig3_throughput()
    print(f"\nFiguras guardadas en: figures/")
    print("  fig1_metrics.png   → Figure 1 del informe")
    print("  fig2_human_eval.png → Figure 2 del informe")
    print("  fig3_throughput.png → Figure 3 del informe")
