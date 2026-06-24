"""
Rellena los scores nulos de los ficheros human_eval_prompts.json
con las puntuaciones asignadas en el análisis (analysis.md, sección 5.3).
También genera results/comparison/human_eval_comparison.json con la tabla
completa de evaluación humana de todos los modelos.
"""

import json
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# Puntuaciones asignadas en analysis.md (sección 5.3) y analysis2.md
# Orden: [Helpfulness, Factuality, Instruction-following]
SCORES = {
    "original": [
        {"helpfulness": 3, "factuality": 2, "instruction_following": 4},  # P1 Capital
        {"helpfulness": 1, "factuality": 2, "instruction_following": 1},  # P2 Huevos
        {"helpfulness": 4, "factuality": 5, "instruction_following": 2},  # P3 Sentimiento
        {"helpfulness": 3, "factuality": 4, "instruction_following": 1},  # P4 Poema
        {"helpfulness": 3, "factuality": 4, "instruction_following": 3},  # P5 Cafetería
    ],
    "transformers": [
        {"helpfulness": 2, "factuality": 1, "instruction_following": 4},
        {"helpfulness": 3, "factuality": 3, "instruction_following": 3},
        {"helpfulness": 4, "factuality": 5, "instruction_following": 4},
        {"helpfulness": 3, "factuality": 4, "instruction_following": 4},
        {"helpfulness": 1, "factuality": 3, "instruction_following": 3},
    ],
    "unsloth": [
        {"helpfulness": 2, "factuality": 1, "instruction_following": 4},
        {"helpfulness": 4, "factuality": 4, "instruction_following": 4},
        {"helpfulness": 3, "factuality": 3, "instruction_following": 3},
        {"helpfulness": 1, "factuality": 2, "instruction_following": 1},
        {"helpfulness": 2, "factuality": 3, "instruction_following": 3},
    ],
    # v2: Unsloth LoRA v2 (6k muestras, 2 épocas, max_seq_len=384)
    # P1 Capital: Canberra correcto pero razón de confusión débil ("same country").
    # P2 Huevos: 7 pasos claros y correctos.
    # P3 Sentimiento: perfecta clasificación de las 8 palabras en dos listas.
    # P4 Poema: 4 líneas pero "a sight to behold" y "diamonds" se repiten — poco creativo.
    # P5 Cafetería: 5 nombres tipo "The Coffee X" sin explicación del concepto (requisito omitido).
    "unsloth_v2": [
        {"helpfulness": 3, "factuality": 3, "instruction_following": 4},  # P1 Capital
        {"helpfulness": 4, "factuality": 4, "instruction_following": 4},  # P2 Huevos
        {"helpfulness": 5, "factuality": 5, "instruction_following": 5},  # P3 Sentimiento
        {"helpfulness": 2, "factuality": 4, "instruction_following": 3},  # P4 Poema
        {"helpfulness": 1, "factuality": 3, "instruction_following": 1},  # P5 Cafetería
    ],
}

PATHS = {
    "original":    RESULTS / "original_model"    / "human_eval_prompts.json",
    "transformers": RESULTS / "transformers_lora" / "eval" / "human_eval_prompts.json",
    "unsloth":     RESULTS / "unsloth_lora"      / "eval" / "human_eval_prompts.json",
    "unsloth_v2":  RESULTS / "unsloth_lora_v2"   / "eval" / "human_eval_prompts.json",
}

PROMPT_LABELS = [
    "Capital de Australia (open_qa — Factuality)",
    "Huevos revueltos (general_qa — Helpfulness)",
    "Clasificación sentimiento (classification — Instruction-following)",
    "Poema 4 líneas (creative_writing — Helpfulness + IF)",
    "Nombres cafetería (brainstorming — Helpfulness)",
]


def fill_scores():
    for model_key, path in PATHS.items():
        if not path.exists():
            print(f"  SKIP (no existe): {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for i, item in enumerate(data):
            item["scores"] = SCORES[model_key][i]
            item["scores"]["mean"] = round(
                sum(SCORES[model_key][i].values()) / 3, 2
            )
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"  Scores rellenados: {path.name} ({model_key})")


def generate_comparison_table():
    out_dir = RESULTS / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    available = {k: p for k, p in PATHS.items() if p.exists()}
    data = {k: json.loads(p.read_text()) for k, p in available.items()}

    comparison = []
    for i, label in enumerate(PROMPT_LABELS):
        row = {"prompt": label}
        for model_key in available:
            item = data[model_key][i]
            row[model_key] = {
                "response": item["prediction"][:120] + "..." if len(item["prediction"]) > 120 else item["prediction"],
                "scores": item["scores"],
            }
        comparison.append(row)

    summary = {}
    for model_key in available:
        h   = [SCORES[model_key][i]["helpfulness"]           for i in range(5)]
        f   = [SCORES[model_key][i]["factuality"]            for i in range(5)]
        if_ = [SCORES[model_key][i]["instruction_following"] for i in range(5)]
        summary[model_key] = {
            "helpfulness_mean":           round(sum(h)  / 5, 2),
            "factuality_mean":            round(sum(f)  / 5, 2),
            "instruction_following_mean": round(sum(if_) / 5, 2),
            "overall_mean":               round((sum(h) + sum(f) + sum(if_)) / 15, 2),
        }

    result = {"per_prompt": comparison, "summary": summary}
    out = out_dir / "human_eval_comparison.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    labels = {
        "original":    "TinyLlama original",
        "transformers": "Transformers LoRA v1",
        "unsloth":     "Unsloth (MLX) v1",
        "unsloth_v2":  "Unsloth (MLX) v2",
    }

    print("\n── Resumen evaluación humana (media sobre 5 prompts) ──────")
    print(f"{'Modelo':<28} {'Helpfulness':>12} {'Factuality':>11} {'Instr.-follow':>14} {'MEDIA':>7}")
    print("-" * 75)
    for k, v in summary.items():
        print(f"  {labels.get(k, k):<26} {v['helpfulness_mean']:>12.2f} "
              f"{v['factuality_mean']:>11.2f} "
              f"{v['instruction_following_mean']:>14.2f} "
              f"{v['overall_mean']:>7.2f}")

    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    fill_scores()
    generate_comparison_table()
    print("\nFicheros de evaluación humana actualizados.")
