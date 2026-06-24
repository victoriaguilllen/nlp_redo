"""
EVALUACIÓN V2 — Modelo Unsloth LoRA v2

Evalúa results/unsloth_lora_v2/ sobre los mismos 100 ejemplos de test
y los mismos 5 prompts de evaluación humana que v1, para comparación directa.

Uso:
    python3.11 src/eval/evaluate_unsloth_v2.py
"""

import json
import time
from pathlib import Path

ROOT         = Path(__file__).resolve().parents[2]
SPLITS_DIR   = ROOT / "data" / "splits"
ADAPTER_DIR  = ROOT / "results" / "unsloth_lora_v2"
RESULTS_DIR  = ADAPTER_DIR / "eval"
MAX_NEW_TOKENS = 256

HUMAN_EVAL_PROMPTS = [
    {
        "id": 1, "category": "open_qa", "dimension": "Factuality",
        "prompt": "### Instruction:\nWhat is the capital of Australia, and why is it often confused with Sydney?\n\n### Response:\n",
    },
    {
        "id": 2, "category": "general_qa", "dimension": "Helpfulness",
        "prompt": "### Instruction:\nGive me clear step-by-step instructions to make perfect scrambled eggs.\n\n### Response:\n",
    },
    {
        "id": 3, "category": "classification", "dimension": "Instruction-following",
        "prompt": "### Instruction:\nClassify each of the following words as either positive or negative sentiment: joy, anger, love, fear, hope, sadness, peace, rage. Present the result as two lists.\n\n### Response:\n",
    },
    {
        "id": 4, "category": "creative_writing", "dimension": "Helpfulness + Instruction-following",
        "prompt": "### Instruction:\nWrite a four-line rhyming poem about the night sky and the stars.\n\n### Response:\n",
    },
    {
        "id": 5, "category": "brainstorming", "dimension": "Helpfulness",
        "prompt": "### Instruction:\nSuggest 5 creative and memorable names for a small independent coffee shop, and explain the concept behind each name.\n\n### Response:\n",
    },
]


def load_model():
    from unsloth import FastLanguageModel
    print(f"Cargando modelo desde '{ADAPTER_DIR}'...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER_DIR),
        max_seq_length=MAX_NEW_TOKENS * 4,
        load_in_4bit=False,
    )
    return model, tokenizer


def generate(model, tokenizer, prompt: str) -> tuple[str, float]:
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    t0 = time.perf_counter()
    result = mlx_generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=MAX_NEW_TOKENS,
        sampler=make_sampler(temp=0.0),
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    if result.startswith(prompt):
        result = result[len(prompt):]
    return result.strip(), elapsed


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


def main(n_samples: int = 100):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model()

    # ── Métricas automáticas ───────────────────────────────────────────────────
    test_rows = [json.loads(l) for l in (SPLITS_DIR / "test.jsonl").open()][:n_samples]

    print(f"\nGenerando {n_samples} predicciones...")
    pred_rows, latencies = [], []
    for i, row in enumerate(test_rows):
        pred, elapsed = generate(model, tokenizer, row["prompt"])
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
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score / 100.0

    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)) / len(preds)

    print("  BERTScore (puede tardar ~1 min)...")
    from bert_score import score as bscore
    _, _, F1 = bscore(preds, refs, lang="en", verbose=False)
    bertscore = F1.mean().item()

    avg_lat = sum(latencies) / len(latencies)
    throughput = (sum(len(p) for p in preds) / 4) / sum(latencies)

    metrics = {
        "experiment":           "unsloth_lora_v2",
        "model":                "TinyLlama + Unsloth LoRA v2 (MLX)",
        "device":               "Apple MLX",
        "n_samples":            n_samples,
        "BLEU":                 round(bleu, 6),
        "ROUGE-L":              round(rouge_l, 6),
        "BERTScore_F1_mean":    round(bertscore, 6),
        "avg_latency_s":        round(avg_lat, 3),
        "avg_throughput_tok_s": round(throughput, 1),
        "v1_reference": {
            "BLEU":              0.079252,
            "ROUGE-L":           0.253098,
            "BERTScore_F1_mean": 0.869745,
        },
    }
    write_json(RESULTS_DIR / "metrics.json", metrics)

    print("\n── Métricas v2 vs v1 ─────────────────────────────────────────")
    print(f"  BLEU      : {bleu:.6f}  (v1: 0.079252)")
    print(f"  ROUGE-L   : {rouge_l:.6f}  (v1: 0.253098)")
    print(f"  BERTScore : {bertscore:.6f}  (v1: 0.869745)")

    # ── Evaluación humana ──────────────────────────────────────────────────────
    print("\nGenerando respuestas para evaluación humana...")
    human_results = []
    for item in HUMAN_EVAL_PROMPTS:
        pred, elapsed = generate(model, tokenizer, item["prompt"])
        human_results.append({
            **item,
            "prediction": pred,
            "latency_s":  round(elapsed, 3),
            "scores": {"helpfulness": None, "factuality": None, "instruction_following": None},
        })
        print(f"  [{item['category']}] → {pred[:80]}...")

    write_json(RESULTS_DIR / "human_eval_prompts.json", human_results)
    print(f"\nResultados guardados en '{RESULTS_DIR}'")
    print("Evaluación Unsloth v2 completada.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=100)
    main(n_samples=p.parse_args().n_samples)
