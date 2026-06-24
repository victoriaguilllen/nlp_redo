"""
Genera todas las figuras del proyecto (v2 actualizado).
Sobrescribe las existentes y añade nuevas comparativas.

Uso:
    python3.11 src/generate_figures_v2.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT    = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT     = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# ── Colores y estilo ──────────────────────────────────────────────────────────
C = {
    "orig":  "#888888",
    "tf":    "#2E86AB",   # azul
    "u1":    "#E07B39",   # naranja
    "u2":    "#3BB273",   # verde
    "ollama":"#9B59B6",   # morado
}
LABELS = {
    "orig":  "Original\n(0-shot)",
    "tf":    "Transformers\nLoRA v1",
    "u1":    "Unsloth\nLoRA v1",
    "u2":    "Unsloth\nLoRA v2",
    "ollama":"Ollama\n(Metal)",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
DPI = 150

# ── Cargar datos ──────────────────────────────────────────────────────────────
orig = json.loads((RESULTS/"original_model"/"metrics.json").read_text())
tf1  = json.loads((RESULTS/"transformers_lora"/"eval"/"metrics.json").read_text())
u1   = json.loads((RESULTS/"unsloth_lora"/"eval"/"metrics.json").read_text())
u2   = json.loads((RESULTS/"unsloth_lora_v2"/"eval"/"metrics.json").read_text())

comp = json.loads((RESULTS/"comparison"/"human_eval_comparison.json").read_text())
hs   = comp["summary"]

bt  = json.loads((RESULTS/"benchmarks"/"benchmark_transformers.json").read_text())
bu1 = json.loads((RESULTS/"benchmarks"/"benchmark_unsloth.json").read_text())
bu2 = json.loads((RESULTS/"benchmarks"/"benchmark_unsloth_v2.json").read_text())
bol = json.loads((RESULTS/"benchmarks"/"benchmark_ollama.json").read_text())

tfl = json.loads((RESULTS/"transformers_lora"/"training_log.json").read_text())
ul1 = json.loads((RESULTS/"unsloth_lora"/"training_log.json").read_text())
ul2 = json.loads((RESULTS/"unsloth_lora_v2"/"training_log.json").read_text())

SCORES = {
    "original":    [[3,2,4],[1,2,1],[4,5,2],[3,4,1],[3,4,3]],
    "transformers":[[2,1,4],[3,3,3],[4,5,4],[3,4,4],[1,3,3]],
    "unsloth":     [[2,1,4],[4,4,4],[3,3,3],[1,2,1],[2,3,3]],
    "unsloth_v2":  [[3,3,4],[4,4,4],[5,5,5],[2,4,3],[1,3,1]],
}

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Mejora relativa sobre baseline (BLEU, ROUGE-L, BERTScore)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1():
    metrics   = ["BLEU", "ROUGE-L", "BERTScore ×10"]
    keys      = ["BLEU", "ROUGE-L", "BERTScore_F1_mean"]
    scale     = [1, 1, 10]
    base_vals = [orig[k]*s for k, s in zip(keys, scale)]

    models = [
        ("tf",  tf1,  "Transformers LoRA v1"),
        ("u1",  u1,   "Unsloth LoRA v1"),
        ("u2",  u2,   "Unsloth LoRA v2"),
    ]

    x   = np.arange(len(metrics))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, (key, data, label) in enumerate(models):
        vals = [(data[k]*s - b)/b*100 for k, s, b in zip(keys, scale, base_vals)]
        bars = ax.bar(x + (i-1)*w, vals, w, color=C[key], label=label,
                      zorder=3, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                    f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=C[key])

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel("Relative improvement over baseline (%)", fontsize=11)
    ax.set_title("Figure 1 — Automatic Metric Improvement after LoRA Fine-tuning\n"
                 "(BERTScore ×10 for scale visibility)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max([tf1["BLEU"]/orig["BLEU"]-1 for _ in [1]])*100*1.25 + 5)
    ax.set_ylim(0, 75)
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(OUT/"fig1_metrics.png", dpi=DPI)
    plt.close()
    print("  fig1_metrics.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Evaluación humana: media por criterio
# ═══════════════════════════════════════════════════════════════════════════════
def fig2():
    criteria = ["Helpfulness", "Factuality", "Instruction-\nFollowing"]
    dims     = ["helpfulness_mean", "factuality_mean", "instruction_following_mean"]
    models   = [("orig","original"),("tf","transformers"),("u1","unsloth"),("u2","unsloth_v2")]

    x = np.arange(len(criteria))
    w = 0.20
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, (key, hkey) in enumerate(models):
        vals = [hs[hkey][d] for d in dims]
        bars = ax.bar(x + (i-1.5)*w, vals, w, color=C[key],
                      label=LABELS[key].replace("\n"," "),
                      zorder=3, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.04,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=C[key])

    ax.axhline(3, color="gray", linestyle=":", linewidth=1.2, label="Mid-scale (3)")
    ax.set_xticks(x)
    ax.set_xticklabels(criteria, fontsize=12)
    ax.set_ylabel("Mean score (1–5)", fontsize=11)
    ax.set_ylim(0, 5.8)
    ax.set_title("Figure 2 — Human Evaluation: Mean Score per Criterion\n"
                 "(5 prompts × 3 criteria; dotted line = mid-scale)", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT/"fig2_human_eval.png", dpi=DPI)
    plt.close()
    print("  fig2_human_eval.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Throughput horizontal bar con barras de error (±1 std)
# ═══════════════════════════════════════════════════════════════════════════════
def fig3():
    import statistics as _st
    names  = ["Ollama\n(llama.cpp Metal)", "Unsloth v1\n(mlx-lm, base)",
              "Unsloth v2\n(mlx-lm, +LoRA)", "Transformers\n(PyTorch MPS)"]
    colors = [C["ollama"], C["u1"], C["u2"], C["tf"]]

    # Calcular media y std desde los trials individuales
    all_b  = [bol, bu1, bu2, bt]
    means  = [_st.mean(r["tok_per_s"] for r in b["trials"]) for b in all_b]
    stds   = [_st.stdev(r["tok_per_s"] for r in b["trials"]) for b in all_b]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y = np.arange(len(names))
    bars = ax.barh(y, means, color=colors, height=0.55, zorder=3, edgecolor="white")
    ax.errorbar(means, y, xerr=stds, fmt="none", color="black",
                capsize=5, capthick=1.5, linewidth=1.5, zorder=5)

    for bar, v, s in zip(bars, means, stds):
        ax.text(v + s + 3, bar.get_y()+bar.get_height()/2,
                f"{v:.1f} ± {s:.1f} tok/s", va="center", fontsize=9.5, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Throughput (tokens / second)", fontsize=11)
    ax.set_xlim(0, 340)
    ax.set_title("Figure 3 — Inference Throughput on Apple M4 Pro\n"
                 "(mean ± 1 std, 3 trials × 5 prompts; temperature=0, max_tokens=256)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT/"fig3_throughput.png", dpi=DPI)
    plt.close()
    print("  fig3_throughput.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Boxplot latencia (todos los frameworks)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4():
    data_tf  = [r["latency_s"] for r in bt["trials"]]
    data_u1  = [r["latency_s"] for r in bu1["trials"]]
    data_u2  = [r["latency_s"] for r in bu2["trials"]]
    data_ol  = [r["latency_s"] for r in bol["trials"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot([data_tf, data_u1, data_u2, data_ol],
                    patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.4),
                    capprops=dict(linewidth=1.4),
                    flierprops=dict(marker="o", markersize=5))

    box_colors = [C["tf"], C["u1"], C["u2"], C["ollama"]]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    for i, data in enumerate([data_tf, data_u1, data_u2, data_ol], 1):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(data))
        ax.scatter([i+j for j in jitter], data, color=box_colors[i-1],
                   s=22, alpha=0.7, zorder=5)

    ax.set_xticks([1,2,3,4])
    ax.set_xticklabels(["Transformers\n(PyTorch MPS)", "Unsloth v1\n(mlx-lm, base)",
                         "Unsloth v2\n(mlx-lm, +LoRA)", "Ollama\n(Metal)"], fontsize=10)
    ax.set_ylabel("Latency per request (seconds)", fontsize=11)
    ax.set_title("Figure 4 — Latency Distribution per Inference Framework\n"
                 "(15 measurements = 5 prompts × 3 trials; points show individual runs)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT/"fig4_latency_boxplot.png", dpi=DPI)
    plt.close()
    print("  fig4_latency_boxplot.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Heatmap evaluación humana por prompt (ahora incluye v2)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5():
    prompt_labels = ["P1\nCapital", "P2\nHuevos", "P3\nSentim.", "P4\nPoema", "P5\nCafet."]
    crit_labels   = ["Helpfulness", "Factuality", "Instr.-Following"]
    model_keys    = ["original", "transformers", "unsloth", "unsloth_v2"]
    model_titles  = ["Original\n(0-shot)", "Transformers\nLoRA v1",
                     "Unsloth\nLoRA v1", "Unsloth\nLoRA v2"]

    cmap = plt.cm.RdYlGn
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)

    for ax, mkey, mtitle in zip(axes, model_keys, model_titles):
        mat = np.array(SCORES[mkey], dtype=float)
        im  = ax.imshow(mat, cmap=cmap, vmin=1, vmax=5, aspect="auto")
        ax.set_xticks(range(3))
        ax.set_xticklabels(crit_labels, fontsize=9, rotation=30, ha="right")
        ax.set_yticks(range(5))
        ax.set_yticklabels(prompt_labels, fontsize=9)
        ax.set_title(mtitle, fontsize=10, fontweight="bold")
        for r in range(5):
            for c in range(3):
                ax.text(c, r, str(int(mat[r, c])), ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if mat[r,c] < 2.5 else "black")

    fig.colorbar(im, ax=axes[-1], label="Score (1=poor → 5=excellent)", shrink=0.8)
    fig.suptitle("Figure 5 — Human Evaluation Scores per Prompt and Criterion\n"
                 "(colour: red=1 poor → green=5 excellent)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT/"fig5_human_heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  fig5_human_heatmap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 (NUEVA) — Radar chart: comparativa global todos los modelos
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_radar():
    # Dimensiones normalizadas a 0–1
    # BLEU (max ~0.10), ROUGE (max ~0.27), BERT (0.865–0.875 → zoom), H, F, IF (/5), latencia inversa
    def norm_bleu(v):  return v / 0.10
    def norm_rouge(v): return v / 0.27
    def norm_bert(v):  return (v - 0.864) / (0.876 - 0.864)  # zoom en rango real
    def norm_h(v):     return v / 5
    def norm_f(v):     return v / 5
    def norm_if(v):    return v / 5
    def norm_lat(v):   return 1 - (v - 0.5) / (3.5 - 0.5)   # menor latencia → mayor score

    models_radar = [
        ("orig",  [norm_bleu(orig["BLEU"]),  norm_rouge(orig["ROUGE-L"]),  norm_bert(orig["BERTScore_F1_mean"]),
                   norm_h(2.8),  norm_f(3.4),  norm_if(2.2),  norm_lat(bt["mean_latency_s"])]),
        ("tf",    [norm_bleu(tf1["BLEU"]),   norm_rouge(tf1["ROUGE-L"]),   norm_bert(tf1["BERTScore_F1_mean"]),
                   norm_h(2.6),  norm_f(3.2),  norm_if(3.6),  norm_lat(bt["mean_latency_s"])]),
        ("u1",    [norm_bleu(u1["BLEU"]),    norm_rouge(u1["ROUGE-L"]),    norm_bert(u1["BERTScore_F1_mean"]),
                   norm_h(2.4),  norm_f(2.6),  norm_if(3.0),  norm_lat(bu1["mean_latency_s"])]),
        ("u2",    [norm_bleu(u2["BLEU"]),    norm_rouge(u2["ROUGE-L"]),    norm_bert(u2["BERTScore_F1_mean"]),
                   norm_h(3.0),  norm_f(3.8),  norm_if(3.4),  norm_lat(bu2["mean_latency_s"])]),
    ]

    categories = ["BLEU", "ROUGE-L", "BERTScore\n(zoom)", "Helpfulness\n/5",
                  "Factuality\n/5", "Instr.-\nFollowing /5", "Speed\n(inv. latency)"]
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for key, vals in models_radar:
        vals_plot = vals + vals[:1]
        ax.plot(angles, vals_plot, color=C[key], linewidth=2.2, linestyle="solid")
        ax.fill(angles, vals_plot, color=C[key], alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8, color="gray")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)

    patches = [mpatches.Patch(color=C[k], label=LABELS[k].replace("\n"," "))
               for k, _ in models_radar]
    ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.35, 1.15),
              fontsize=10, framealpha=0.9)
    ax.set_title("Figure 6 — Model Comparison Radar\n"
                 "(all axes normalised 0–1; larger area = better)", fontsize=12, pad=20)
    fig.tight_layout()
    fig.savefig(OUT/"fig6_radar.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  fig6_radar.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Training comparison (v1 + v2)
# ═══════════════════════════════════════════════════════════════════════════════
def fig7():
    groups = [
        ("Training time (min)",
         [tfl["results"]["training_time_min"], ul1["results"]["training_time_min"], ul2["results"]["training_time_min"]]),
        ("Final training loss",
         [tfl["results"]["training_loss"], ul1["results"]["training_loss"], ul2["results"]["training_loss"]]),
        ("Peak GPU/MLX mem (GB)",
         [tfl["memory"]["mps_peak_mb"]/1024, ul1["memory"]["mlx_peak_train_mb"]/1024, ul2["memory"]["mlx_peak_train_mb"]/1024]),
        ("Train samples (k)",
         [tfl.get("max_train_samples",2000)/1000, ul1.get("max_train_samples",2000)/1000, ul2.get("max_train_samples",6000)/1000]),
    ]
    bar_colors = [C["tf"], C["u1"], C["u2"]]
    bar_labels = ["Transformers\nLoRA v1", "Unsloth\nLoRA v1", "Unsloth\nLoRA v2"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, (title, vals) in zip(axes, groups):
        x = np.arange(3)
        best_idx = np.argmin(vals) if "loss" in title or "mem" in title or "time" in title else np.argmax(vals)
        bars = ax.bar(x, vals, color=bar_colors, zorder=3, edgecolor="white", linewidth=0.5)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(vals)*1.25)

    fig.suptitle("Figure 7 — Training Comparison: TF v1 vs Unsloth v1 vs Unsloth v2\n"
                 "(gold border = winner per metric)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT/"fig7_training_comparison.png", dpi=DPI)
    plt.close()
    print("  fig7_training_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 (NUEVA) — Quality vs Speed scatter: el gran trade-off
# ═══════════════════════════════════════════════════════════════════════════════
def fig8_quality_vs_speed():
    # quality score = media de (BLEU norm + ROUGE norm + human_overall/5)
    # latency = mean latency per request

    def quality(bleu, rouge, bert, human_overall):
        return (bleu/0.10 + rouge/0.27 + (bert-0.865)/0.011 + human_overall/5) / 4

    pts = [
        ("orig", quality(orig["BLEU"], orig["ROUGE-L"], orig["BERTScore_F1_mean"], 2.80),
         bt["mean_latency_s"], 2000, "Original\n(0-shot)"),
        ("tf",   quality(tf1["BLEU"],  tf1["ROUGE-L"],  tf1["BERTScore_F1_mean"],  3.13),
         bt["mean_latency_s"], 2000, "Transformers\nLoRA v1"),
        ("u1",   quality(u1["BLEU"],   u1["ROUGE-L"],   u1["BERTScore_F1_mean"],   2.67),
         bu1["mean_latency_s"], 2000, "Unsloth\nLoRA v1"),
        ("u2",   quality(u2["BLEU"],   u2["ROUGE-L"],   u2["BERTScore_F1_mean"],   3.40),
         bu2["mean_latency_s"], 6000, "Unsloth\nLoRA v2"),
        ("ollama", quality(orig["BLEU"], orig["ROUGE-L"], orig["BERTScore_F1_mean"], 2.80),
         bol["mean_latency_s"], 2000, "Ollama\n(base, Metal)"),
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    for key, q, lat, samples, label in pts:
        size = 200 + samples / 15  # burbuja proporcional a muestras de entrenamiento
        ax.scatter(lat, q, s=size, color=C[key], alpha=0.85, zorder=5, edgecolors="white", linewidth=1.5)
        offset_x = 0.06
        offset_y = 0.012
        ax.annotate(label, (lat, q), xytext=(lat+offset_x, q+offset_y),
                    fontsize=9.5, color=C[key], fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=C[key], lw=0.8, alpha=0.5))

    ax.set_xlabel("Latency per request (seconds) — lower is better →", fontsize=11)
    ax.set_ylabel("Composite quality score (0–1) — higher is better ↑", fontsize=11)
    ax.set_title("Figure 8 — Quality vs. Speed Trade-off\n"
                 "(bubble size ∝ training samples; sweet spot = top-left)", fontsize=12)
    ax.invert_xaxis()
    ax.set_xlim(3.0, 0.5)

    ax.axhline(quality(u2["BLEU"], u2["ROUGE-L"], u2["BERTScore_F1_mean"], 3.40),
               color=C["u2"], linestyle=":", alpha=0.4)
    ax.axvline(bu2["mean_latency_s"], color=C["u2"], linestyle=":", alpha=0.4)
    ax.text(0.6, 0.02, "← Fast & High Quality\n     (ideal zone)", fontsize=9,
            color="gray", style="italic", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(OUT/"fig8_quality_vs_speed.png", dpi=DPI)
    plt.close()
    print("  fig8_quality_vs_speed.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 (NUEVA) — Journey: mejora progresiva paso a paso
# ═══════════════════════════════════════════════════════════════════════════════
def fig9_improvement_journey():
    steps  = ["Baseline\n(0-shot)", "TF LoRA\nv1", "Unsloth\nv1", "Unsloth\nv2"]
    colors = [C["orig"], C["tf"], C["u1"], C["u2"]]

    bleu   = [orig["BLEU"], tf1["BLEU"], u1["BLEU"], u2["BLEU"]]
    rouge  = [orig["ROUGE-L"], tf1["ROUGE-L"], u1["ROUGE-L"], u2["ROUGE-L"]]
    human  = [2.80, 3.13, 2.67, 3.40]

    x = np.arange(len(steps))
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)

    for ax, vals, title, ylab, color_line in zip(
            axes,
            [bleu, rouge, human],
            ["BLEU", "ROUGE-L", "Human Eval (overall mean)"],
            ["BLEU score", "ROUGE-L score", "Mean score (1–5)"],
            ["#555", "#555", "#555"]):

        ax.plot(x, vals, color="#ccc", linewidth=2.5, zorder=3)
        for i, (xi, v) in enumerate(zip(x, vals)):
            ax.scatter(xi, v, color=colors[i], s=160, zorder=5, edgecolors="white", linewidth=1.5)
            ax.text(xi, v + (max(vals)-min(vals))*0.06,
                    f"{v:.4f}" if title in ["BLEU","ROUGE-L"] else f"{v:.2f}",
                    ha="center", fontsize=9.5, fontweight="bold", color=colors[i])

        ax.set_xticks(x)
        ax.set_xticklabels(steps, fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(min(vals)*0.92, max(vals)*1.12)

        # Flecha de mejora total
        total = (vals[-1]-vals[0])/vals[0]*100
        ax.annotate("", xy=(3, vals[-1]), xytext=(0, vals[0]),
                    arrowprops=dict(arrowstyle="->", color=C["u2"], lw=1.8,
                                   connectionstyle="arc3,rad=-0.2"))
        ax.text(1.5, (vals[0]+vals[-1])/2 + (max(vals)-min(vals))*0.18,
                f"+{total:.1f}% total", ha="center", fontsize=9,
                color=C["u2"], style="italic")

    fig.suptitle("Figure 9 — Improvement Journey: Baseline → Best Model\n"
                 "(coloured dots follow the palette: gray / blue / orange / green)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT/"fig9_improvement_journey.png", dpi=DPI)
    plt.close()
    print("  fig9_improvement_journey.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 10 (NUEVA) — Unsloth v1 vs v2: prompt por prompt (human eval)
# ═══════════════════════════════════════════════════════════════════════════════
def fig10_v1_vs_v2_prompts():
    prompt_short = ["P1\nCapital", "P2\nHuevos", "P3\nSentim.", "P4\nPoema", "P5\nCafet."]
    dims   = ["Helpfulness", "Factuality", "Instr.-Following"]
    v1_mat = np.array(SCORES["unsloth"],    dtype=float)
    v2_mat = np.array(SCORES["unsloth_v2"], dtype=float)

    x = np.arange(5)
    w = 0.28
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for ax, col, dim in zip(axes, range(3), dims):
        bars1 = ax.bar(x - w/2, v1_mat[:, col], w, color=C["u1"], label="Unsloth v1",
                       zorder=3, edgecolor="white")
        bars2 = ax.bar(x + w/2, v2_mat[:, col], w, color=C["u2"], label="Unsloth v2",
                       zorder=3, edgecolor="white")
        for b, v in zip(bars1, v1_mat[:, col]):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                    str(int(v)), ha="center", fontsize=10, color=C["u1"], fontweight="bold")
        for b, v in zip(bars2, v2_mat[:, col]):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05,
                    str(int(v)), ha="center", fontsize=10, color=C["u2"], fontweight="bold")
        # delta arrows
        for i, (v1, v2) in enumerate(zip(v1_mat[:, col], v2_mat[:, col])):
            delta = v2 - v1
            if delta != 0:
                color = C["u2"] if delta > 0 else "#cc3333"
                ax.annotate("", xy=(i+w/2, v2+0.15), xytext=(i-w/2, v1+0.15),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

        ax.set_xticks(x)
        ax.set_xticklabels(prompt_short, fontsize=10)
        ax.set_ylim(0, 6.2)
        ax.set_title(dim, fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel("Score (1–5)", fontsize=10)
        ax.legend(fontsize=9)

    fig.suptitle("Figure 10 — Unsloth v1 vs v2: Human Scores per Prompt\n"
                 "(arrows show direction of change; green = improvement)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT/"fig10_v1_vs_v2_prompts.png", dpi=DPI)
    plt.close()
    print("  fig10_v1_vs_v2_prompts.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 11 (NUEVA) — Variabilidad de inferencia: mean ± std para latencia y throughput
# ═══════════════════════════════════════════════════════════════════════════════
def fig11_std_inference():
    import statistics as _st
    frameworks = ["Transformers\n(MPS)", "Unsloth v1\n(MLX, base)",
                  "Unsloth v2\n(MLX, +LoRA)", "Ollama\n(Metal)"]
    colors     = [C["tf"], C["u1"], C["u2"], C["ollama"]]
    all_b      = [bt, bu1, bu2, bol]

    lat_mean = [_st.mean(r["latency_s"] for r in b["trials"]) for b in all_b]
    lat_std  = [_st.stdev(r["latency_s"] for r in b["trials"]) for b in all_b]
    tok_mean = [_st.mean(r["tok_per_s"]  for r in b["trials"]) for b in all_b]
    tok_std  = [_st.stdev(r["tok_per_s"] for r in b["trials"]) for b in all_b]

    x  = np.arange(4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # — Latencia —
    bars1 = ax1.bar(x, lat_mean, color=colors, zorder=3, edgecolor="white", width=0.55)
    ax1.errorbar(x, lat_mean, yerr=lat_std, fmt="none", color="black",
                 capsize=7, capthick=2, linewidth=2, zorder=5)
    for xi, m, s in zip(x, lat_mean, lat_std):
        ax1.text(xi, m + s + 0.08, f"{m:.2f}s\n±{s:.2f}", ha="center",
                 fontsize=9, fontweight="bold")
        # coeficiente de variación
        cv = s / m * 100
        ax1.text(xi, 0.15, f"CV={cv:.0f}%", ha="center", fontsize=8,
                 color="white" if m > 1.0 else "black", style="italic")
    ax1.set_xticks(x)
    ax1.set_xticklabels(frameworks, fontsize=10)
    ax1.set_ylabel("Latency per request (s)", fontsize=11)
    ax1.set_title("Latency — mean ± 1 std\n(lower & narrower bar = better)", fontsize=11)
    ax1.set_ylim(0, max(m+s for m,s in zip(lat_mean,lat_std)) * 1.35)

    # — Throughput —
    bars2 = ax2.bar(x, tok_mean, color=colors, zorder=3, edgecolor="white", width=0.55)
    ax2.errorbar(x, tok_mean, yerr=tok_std, fmt="none", color="black",
                 capsize=7, capthick=2, linewidth=2, zorder=5)
    for xi, m, s in zip(x, tok_mean, tok_std):
        ax2.text(xi, m + s + 3, f"{m:.1f}\n±{s:.1f}", ha="center",
                 fontsize=9, fontweight="bold")
        cv = s / m * 100
        ax2.text(xi, 4, f"CV={cv:.0f}%", ha="center", fontsize=8,
                 color="white" if m > 80 else "black", style="italic")
    ax2.set_xticks(x)
    ax2.set_xticklabels(frameworks, fontsize=10)
    ax2.set_ylabel("Throughput (tokens / second)", fontsize=11)
    ax2.set_title("Throughput — mean ± 1 std\n(higher & narrower bar = better)", fontsize=11)
    ax2.set_ylim(0, max(m+s for m,s in zip(tok_mean,tok_std)) * 1.35)

    fig.suptitle("Figure 11 — Inference Variability: Mean ± Std across 15 Measurements\n"
                 "(CV = coefficient of variation; higher CV = less predictable latency)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT/"fig11_std_inference.png", dpi=DPI)
    plt.close()
    print("  fig11_std_inference.png")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generando figuras...")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6_radar()
    fig7()
    fig8_quality_vs_speed()
    fig9_improvement_journey()
    fig10_v1_vs_v2_prompts()
    fig11_std_inference()
    print(f"\nTodas las figuras guardadas en: {OUT}")
