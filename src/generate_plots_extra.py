"""
Genera 4 figuras adicionales para el informe usando datos reales de los JSONs.
Ejecutar: python3.11 generate_plots_extra.py
Salida:   figures/fig4_latency_boxplot.png
          figures/fig5_human_heatmap.png
          figures/fig6_toks_vs_length.png
          figures/fig7_training_comparison.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import json
from pathlib import Path

ROOT    = Path(__file__).parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

C_GRAY   = "#7f7f7f"
C_BLUE   = "#1f77b4"
C_ORANGE = "#ff7f0e"
C_GREEN  = "#2ca02c"
C_RED    = "#d62728"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

# ── Load benchmark data ────────────────────────────────────────────────────────
def load_trials(name):
    path = ROOT / "results" / "benchmarks" / f"benchmark_{name}.json"
    d = json.loads(path.read_text())
    return d.get("trials", [])

tf_trials   = load_trials("transformers")
uns_trials  = load_trials("unsloth")
oll_trials  = load_trials("ollama")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Box plot: distribución de latencia por framework
# ══════════════════════════════════════════════════════════════════════════════
def fig4_latency_boxplot():
    tf_lat  = [t["latency_s"] for t in tf_trials]
    uns_lat = [t["latency_s"] for t in uns_trials]
    oll_lat = [t["latency_s"] for t in oll_trials]
    vllm_lat = [8.5]  # CPU fallback, single estimate

    data   = [tf_lat, uns_lat, oll_lat, vllm_lat]
    labels = ["Transformers\n(PyTorch MPS)", "Unsloth\n(mlx-lm)", "Ollama\n(llama.cpp Metal)", "vLLM\n(CPU — no GPU)"]
    colors = [C_BLUE, C_ORANGE, C_GREEN, C_RED]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bp = ax.boxplot(data, patch_artist=True, widths=0.45,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.3),
                    capprops=dict(linewidth=1.3),
                    flierprops=dict(marker="o", markersize=4, alpha=0.6))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Add individual points (jittered)
    np.random.seed(42)
    for i, (d, color) in enumerate(zip(data, colors), 1):
        if len(d) > 1:
            jitter = np.random.uniform(-0.1, 0.1, len(d))
            ax.scatter([i + j for j in jitter], d, color=color,
                       s=28, zorder=5, alpha=0.85, edgecolors="white", linewidths=0.5)

    # vLLM marker: single estimate
    ax.scatter([4], vllm_lat, color=C_RED, marker="*", s=120,
               zorder=6, label="vLLM: single estimate (CPU fallback)")

    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels, ha="center")
    ax.set_ylabel("Latency per request (seconds)")
    ax.set_title("Figure 4 — Latency Distribution per Inference Framework\n"
                 "(15 measurements = 5 prompts × 3 trials; points show individual runs)",
                 pad=8)

    # Annotate high variance explanation
    ax.annotate("High spread = response-\nlength dependency",
                xy=(1, 4.2), xytext=(1.6, 4.8),
                fontsize=8.5, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.0))

    ax.legend(fontsize=8.5, loc="upper right")
    plt.tight_layout()
    out = FIG_DIR / "fig4_latency_boxplot.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Heatmap de evaluación humana por prompt
# ══════════════════════════════════════════════════════════════════════════════
def fig5_human_heatmap():
    # Rows: 5 prompts × 3 criteria = 15 cells per model
    # scores[model][prompt][criterion]  — criteria: H, F, IF
    prompts = [
        "P1: Capital\nof Australia",
        "P2: Scrambled\neggs recipe",
        "P3: Sentiment\nclassification",
        "P4: 4-line\npoem",
        "P5: Coffee\nshop names",
    ]
    criteria = ["Helpfulness", "Factuality", "Instr.-Following"]

    # scores[model] is shape (5 prompts, 3 criteria)
    orig_scores = np.array([
        [3, 2, 4],
        [1, 2, 1],
        [4, 5, 2],
        [3, 4, 1],
        [3, 4, 3],
    ], dtype=float)

    tf_scores = np.array([
        [2, 1, 4],
        [3, 3, 3],
        [4, 5, 4],
        [3, 4, 4],
        [1, 3, 3],
    ], dtype=float)

    uns_scores = np.array([
        [2, 1, 4],
        [4, 4, 4],
        [3, 3, 3],
        [1, 2, 1],
        [2, 3, 3],
    ], dtype=float)

    models = ["Original\n(0-shot)", "Transformers\nLoRA", "Unsloth\nLoRA"]
    all_scores = [orig_scores, tf_scores, uns_scores]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)

    for ax, scores, model in zip(axes, all_scores, models):
        im = ax.imshow(scores, cmap="RdYlGn", vmin=1, vmax=5, aspect="auto")
        ax.set_xticks(range(3))
        ax.set_xticklabels(criteria, rotation=20, ha="right", fontsize=9)
        ax.set_yticks(range(5))
        ax.set_yticklabels(prompts if ax == axes[0] else [], fontsize=9)
        ax.set_title(model, fontsize=11, fontweight="bold", pad=6)
        ax.grid(False)

        # Value annotations
        for r in range(5):
            for c in range(3):
                val = int(scores[r, c])
                color = "black" if 2 <= val <= 4 else "white"
                ax.text(c, r, str(val), ha="center", va="center",
                        fontsize=12, fontweight="bold", color=color)

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, orientation="vertical",
                        fraction=0.03, pad=0.03, shrink=0.85)
    cbar.set_label("Score (1=poor → 5=excellent)", fontsize=9)
    cbar.set_ticks([1, 2, 3, 4, 5])

    fig.suptitle("Figure 5 — Human Evaluation Scores per Prompt and Criterion\n"
                 "(colour: red=1 poor → green=5 excellent)",
                 fontsize=12, y=1.02)

    plt.tight_layout()
    out = FIG_DIR / "fig5_human_heatmap.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Scatter: tok/s vs output tokens (shows latency-length relationship)
# ══════════════════════════════════════════════════════════════════════════════
def fig6_toks_vs_length():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel A: latency vs n_output_tokens (scatter + fit)
    ax = axes[0]
    for trials, label, color in [
        (tf_trials,  "Transformers (MPS)", C_BLUE),
        (uns_trials, "Unsloth (mlx-lm)",   C_ORANGE),
        (oll_trials, "Ollama (Metal)",      C_GREEN),
    ]:
        lengths = [t["n_output_tokens"] for t in trials]
        lats    = [t["latency_s"]       for t in trials]
        ax.scatter(lengths, lats, color=color, alpha=0.82,
                   s=50, label=label, edgecolors="white", linewidths=0.5)

    # Linear fit for Transformers to show the trend
    tf_x = np.array([t["n_output_tokens"] for t in tf_trials])
    tf_y = np.array([t["latency_s"]       for t in tf_trials])
    m, b = np.polyfit(tf_x, tf_y, 1)
    r2 = np.corrcoef(tf_x, tf_y)[0, 1] ** 2
    xs = np.linspace(tf_x.min(), tf_x.max(), 100)
    ax.plot(xs, m*xs + b, "--", color=C_BLUE, alpha=0.55, linewidth=1.5,
            label=f"TF linear fit (R²={r2:.2f})")

    ax.set_xlabel("Output length (tokens)")
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("(a) Latency scales linearly with output length\n(explains Transformers' high latency std ±1.5 s)")
    ax.legend(fontsize=8.5)

    # Panel B: mean tok/s per prompt per framework
    ax2 = axes[1]
    prompt_ids = sorted(set(t["prompt_id"] for t in tf_trials))
    x = np.arange(len(prompt_ids))
    w = 0.26

    tf_tps  = [np.mean([t["tok_per_s"] for t in tf_trials  if t["prompt_id"]==p]) for p in prompt_ids]
    uns_tps = [np.mean([t["tok_per_s"] for t in uns_trials if t["prompt_id"]==p]) for p in prompt_ids]
    oll_tps = [np.mean([t["tok_per_s"] for t in oll_trials if t["prompt_id"]==p]) for p in prompt_ids]

    ax2.bar(x - w, tf_tps,  w, label="Transformers", color=C_BLUE,   alpha=0.85)
    ax2.bar(x,     uns_tps, w, label="Unsloth",       color=C_ORANGE, alpha=0.85)
    ax2.bar(x + w, oll_tps, w, label="Ollama",        color=C_GREEN,  alpha=0.85)

    p_labels = ["P1\nCapital", "P2\nEggs", "P3\nSentim.", "P4\nPoem", "P5\nCoffee"]
    ax2.set_xticks(x)
    ax2.set_xticklabels(p_labels, fontsize=9.5)
    ax2.set_ylabel("Throughput (tok/s)")
    ax2.set_ylim(0, 310)
    ax2.set_title("(b) Throughput is consistent across prompts\n(tok/s stable per framework regardless of task)")
    ax2.legend(fontsize=8.5)

    fig.suptitle("Figure 6 — Latency vs. Output Length and Per-Prompt Throughput Consistency",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out = FIG_DIR / "fig6_toks_vs_length.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Training comparison: 4 metrics side by side (radar-free, 2×2 grid)
# ══════════════════════════════════════════════════════════════════════════════
def fig7_training_comparison():
    frameworks = ["Transformers\n(MPS)", "Unsloth\n(MLX)"]
    colors = [C_BLUE, C_ORANGE]

    metrics = {
        "Training time (min)": [28.62, 11.20],
        "Final training loss":  [1.6895, 1.7091],
        "Peak GPU/MLX mem (GB)": [2364.4/1024, 3511.5/1024],
        "Peak RAM delta (GB)":   [215.9/1024,  4316.9/1024],
    }

    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes = axes.flatten()

    for ax, (metric, vals) in zip(axes, metrics.items()):
        bars = ax.bar(frameworks, vals, color=colors, alpha=0.85, width=0.45)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.02,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        ax.set_title(metric, fontsize=10.5)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", alpha=0.35, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Highlight the winner per panel
    axes[0].patches[1].set_edgecolor("gold")  # Unsloth faster
    axes[0].patches[1].set_linewidth(2.5)
    axes[1].patches[0].set_edgecolor("gold")  # Transformers lower loss
    axes[1].patches[0].set_linewidth(2.5)

    fig.suptitle("Figure 7 — Training Framework Comparison (gold border = winner per metric)\n"
                 "Effective batch: Transformers 16 vs Unsloth 8 — explains loss difference",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out = FIG_DIR / "fig7_training_comparison.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generando figuras adicionales...")
    fig4_latency_boxplot()
    fig5_human_heatmap()
    fig6_toks_vs_length()
    fig7_training_comparison()
    print("\nFiguras guardadas en: figures/")
    print("  fig4_latency_boxplot.png → distribución de latencia con box plots")
    print("  fig5_human_heatmap.png   → heatmap de evaluación humana por prompt")
    print("  fig6_toks_vs_length.png  → latencia vs longitud + throughput por prompt")
    print("  fig7_training_comparison.png → 4 métricas de entrenamiento comparadas")
