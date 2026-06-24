"""
PASO 5 — Comparación de los tres modelos

Lee los resultados de evaluación de los tres modelos y genera:
  - results/comparison/full_comparison.json
  - results/comparison/summary_table.txt  (tabla ASCII lista para el informe)

Uso:
    python src/compare_models.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

MODELS = {
    "original":    RESULTS / "original_model"     / "metrics.json",
    "transformers": RESULTS / "transformers_lora"  / "eval" / "metrics.json",
    "unsloth":     RESULTS / "unsloth_lora"       / "eval" / "metrics.json",
}

HUMAN_EVALS = {
    "original":    RESULTS / "original_model"     / "human_eval_prompts.json",
    "transformers": RESULTS / "transformers_lora"  / "eval" / "human_eval_prompts.json",
    "unsloth":     RESULTS / "unsloth_lora"       / "eval" / "human_eval_prompts.json",
}

TRAINING_LOGS = {
    "transformers": RESULTS / "transformers_lora" / "training_log.json",
    "unsloth":      RESULTS / "unsloth_lora"      / "training_log.json",
}

LABELS = {
    "original":     "TinyLlama original",
    "transformers": "TinyLlama + Transformers LoRA",
    "unsloth":      "TinyLlama + Unsloth (MLX)",
}


def load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return {}


def delta_pct(base, new):
    if base and new and base != 0:
        return f"{(new - base) / abs(base) * 100:+.1f}%"
    return "—"


def main():
    out_dir = RESULTS / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cargar métricas automáticas
    metrics = {k: load_json(p) for k, p in MODELS.items()}
    logs    = {k: load_json(p) for k, p in TRAINING_LOGS.items()}

    # ── Tabla de métricas automáticas ─────────────────────────────────────
    auto_keys = ["BLEU", "ROUGE-L", "BERTScore_F1_mean", "avg_latency_s", "avg_throughput_tok_s"]
    comparison = {}
    for key in auto_keys:
        comparison[key] = {k: metrics[k].get(key) for k in MODELS}

    # ── Tabla de entrenamiento ─────────────────────────────────────────────
    training_comparison = {
        "training_time_min": {k: logs[k].get("results", {}).get("training_time_min") for k in logs},
        "training_loss":     {k: logs[k].get("results", {}).get("training_loss") for k in logs},
        "mlx_peak_mb":       {"unsloth": logs.get("unsloth", {}).get("memory", {}).get("mlx_peak_train_mb")},
        "mps_peak_mb":       {"transformers": logs.get("transformers", {}).get("memory", {}).get("mps_peak_mb")},
    }

    full = {
        "automatic_metrics": comparison,
        "training":          training_comparison,
        "labels":            LABELS,
    }
    (out_dir / "full_comparison.json").write_text(
        json.dumps(full, indent=2, ensure_ascii=False)
    )

    # ── Tabla ASCII ────────────────────────────────────────────────────────
    orig_bleu   = metrics["original"].get("BLEU")
    orig_rouge  = metrics["original"].get("ROUGE-L")
    orig_bert   = metrics["original"].get("BERTScore_F1_mean")
    orig_lat    = metrics["original"].get("avg_latency_s")
    orig_thr    = metrics["original"].get("avg_throughput_tok_s")

    rows = []
    for key, label in LABELS.items():
        m = metrics[key]
        b  = m.get("BLEU",               "—")
        r  = m.get("ROUGE-L",            "—")
        bs = m.get("BERTScore_F1_mean",  "—")
        lt = m.get("avg_latency_s",      "—")
        th = m.get("avg_throughput_tok_s","—")
        if key == "original":
            rows.append((label, f"{b:.4f}", f"{r:.4f}", f"{bs:.4f}", f"{lt:.3f}s", f"{th:.1f}"))
        else:
            rows.append((
                label,
                f"{b:.4f} ({delta_pct(orig_bleu, b)})",
                f"{r:.4f} ({delta_pct(orig_rouge, r)})",
                f"{bs:.4f} ({delta_pct(orig_bert, bs)})",
                f"{lt:.3f}s ({delta_pct(orig_lat, lt)})",
                f"{th:.1f} ({delta_pct(orig_thr, th)})",
            ))

    header = ["Modelo", "BLEU", "ROUGE-L", "BERTScore F1", "Latencia", "Throughput"]
    col_w  = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(header)]

    def fmt_row(row):
        return "| " + " | ".join(v.ljust(col_w[i]) for i, v in enumerate(row)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in col_w) + "-|"
    table_lines = [fmt_row(header), sep] + [fmt_row(r) for r in rows]
    table = "\n".join(table_lines)

    (out_dir / "summary_table.txt").write_text(table + "\n")

    print("\n" + "="*70)
    print("COMPARACIÓN DE LOS TRES MODELOS")
    print("="*70)
    print(table)
    print("\nGuardado en:", out_dir)


if __name__ == "__main__":
    main()
