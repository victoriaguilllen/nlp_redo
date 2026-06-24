"""
PASO 5 (parte 2) — Evaluación del modelo Unsloth fine-tuneado

Carga el modelo Unsloth via MLX y evalúa sobre:
  - Los mismos 100 ejemplos del test set
  - Los mismos 5 prompts de evaluación humana

Resultados en: results/unsloth_lora/eval/

Uso:
    python src/evaluate_unsloth.py
"""

import argparse
import json
import time
from pathlib import Path

SPLITS_DIR   = Path(__file__).resolve().parents[2] / "data" / "splits"
ADAPTER_DIR  = Path(__file__).resolve().parents[2] / "results" / "unsloth_lora"
RESULTS_DIR  = ADAPTER_DIR / "eval"
BASELINE_DIR = Path(__file__).resolve().parents[2] / "results" / "original_model"
MAX_NEW_TOKENS = 256

HUMAN_EVAL_PROMPTS = [
    {
        "id": 1, "category": "open_qa", "dimension": "Factuality",
        "prompt": (
            "### Instruction:\nWhat is the capital of Australia, and why is it often "
            "confused with Sydney?\n\n### Response:\n"
        ),
    },
    {
        "id": 2, "category": "general_qa", "dimension": "Helpfulness",
        "prompt": (
            "### Instruction:\nGive me clear step-by-step instructions to make "
            "perfect scrambled eggs.\n\n### Response:\n"
        ),
    },
    {
        "id": 3, "category": "classification", "dimension": "Instruction-following",
        "prompt": (
            "### Instruction:\nClassify each of the following words as either positive "
            "or negative sentiment: joy, anger, love, fear, hope, sadness, peace, rage. "
            "Present the result as two lists.\n\n### Response:\n"
        ),
    },
    {
        "id": 4, "category": "creative_writing", "dimension": "Helpfulness + Instruction-following",
        "prompt": (
            "### Instruction:\nWrite a four-line rhyming poem about the night sky "
            "and the stars.\n\n### Response:\n"
        ),
    },
    {
        "id": 5, "category": "brainstorming", "dimension": "Helpfulness",
        "prompt": (
            "### Instruction:\nSuggest 5 creative and memorable names for a small "
            "independent coffee shop, and explain the concept behind each name.\n\n"
            "### Response:\n"
        ),
    },
]


def load_model_and_tokenizer():
    """Carga el modelo Unsloth fine-tuneado via MLX."""
    from unsloth import FastLanguageModel
    print(f"Cargando modelo Unsloth fine-tuneado desde '{ADAPTER_DIR}'...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER_DIR),
        max_seq_length=MAX_NEW_TOKENS * 4,
        load_in_4bit=False,
    )
    return model, tokenizer


def generate_mlx(model, tokenizer, prompt: str) -> tuple[str, float]:
    """Genera texto con el modelo MLX (greedy, temp=0)."""
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.0)   # greedy decoding
    t0 = time.perf_counter()
    result = mlx_generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=MAX_NEW_TOKENS,
        sampler=sampler,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    if result.startswith(prompt):
        result = result[len(prompt):]
    return result.strip(), elapsed


def compute_bleu(predictions, references) -> float:
    import sacrebleu
    return sacrebleu.corpus_bleu(predictions, [references]).score / 100.0


def compute_rouge_l(predictions, references) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return sum(scorer.score(r, p)["rougeL"].fmeasure
               for p, r in zip(predictions, references)) / len(predictions)


def compute_bertscore(predictions, references) -> float:
    from bert_score import score as bscore
    _, _, F1 = bscore(predictions, references, lang="en", verbose=False)
    return F1.mean().item()


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(n_samples: int = 100):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer()

    # ── 1. Evaluación automática ───────────────────────────────────────────
    test_rows = [
        json.loads(l)
        for l in (SPLITS_DIR / "test.jsonl").open(encoding="utf-8")
    ][:n_samples]

    print(f"\nGenerando respuestas Unsloth ({n_samples} ejemplos)...")
    pred_rows, latencies = [], []
    for i, row in enumerate(test_rows):
        pred, elapsed = generate_mlx(model, tokenizer, row["prompt"])
        pred_rows.append({
            "prompt":     row["prompt"],
            "reference":  row["response"],
            "prediction": pred,
            "category":   row.get("category", ""),
            "latency_s":  round(elapsed, 3),
        })
        latencies.append(elapsed)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_samples}...")

    write_jsonl(RESULTS_DIR / "predictions.jsonl", pred_rows)

    preds = [r["prediction"] for r in pred_rows]
    refs  = [r["reference"]  for r in pred_rows]

    print("\nCalculando métricas...")
    bleu      = compute_bleu(preds, refs)
    rouge_l   = compute_rouge_l(preds, refs)
    print("  BERTScore...")
    bertscore = compute_bertscore(preds, refs)

    avg_lat    = sum(latencies) / len(latencies)
    total_chars = sum(len(r["prediction"]) for r in pred_rows)
    # Estimación de tokens: ~4 chars/token para inglés
    est_tokens  = total_chars / 4
    throughput  = est_tokens / sum(latencies)

    metrics = {
        "model":                "TinyLlama + Unsloth LoRA (MLX)",
        "device":               "Apple MLX",
        "n_samples":            n_samples,
        "BLEU":                 round(bleu, 6),
        "ROUGE-L":              round(rouge_l, 6),
        "BERTScore_F1_mean":    round(bertscore, 6),
        "avg_latency_s":        round(avg_lat, 3),
        "avg_throughput_tok_s": round(throughput, 1),
    }
    write_json(RESULTS_DIR / "metrics.json", metrics)

    print("\n── Métricas Unsloth ─────────────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ── 2. Evaluación humana ───────────────────────────────────────────────
    print("\nGenerando respuestas humanas (Unsloth)...")
    human_results = []
    for item in HUMAN_EVAL_PROMPTS:
        pred, elapsed = generate_mlx(model, tokenizer, item["prompt"])
        human_results.append({
            **item,
            "prediction": pred,
            "latency_s":  round(elapsed, 3),
            "scores": {
                "helpfulness":           None,
                "factuality":            None,
                "instruction_following": None,
            },
        })
        print(f"  Prompt {item['id']} ({item['category']}) generado.")

    write_json(RESULTS_DIR / "human_eval_prompts.json", human_results)
    print(f"\nResultados guardados en '{RESULTS_DIR}'")
    print("Evaluación Unsloth completada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()
    main(n_samples=args.n_samples)
