"""
PASO 2 - Evaluación del modelo original (zero-shot, sin fine-tuning)

Carga TinyLlama-1.1B-Chat y evalúa sobre el test set con:
  - Métricas automáticas: BLEU, ROUGE-L, BERTScore
  - 5 prompts de evaluación humana (inventados para cubrir las dimensiones
    del enunciado: Helpfulness, Factuality, Instruction-following)

Resultados guardados en: results/original_model/

Uso:
    python src/evaluate_baseline.py
    python src/evaluate_baseline.py --n_samples 50   # más rápido
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SPLITS_DIR   = Path(__file__).resolve().parents[2] / "data" / "splits"
RESULTS_DIR  = Path(__file__).resolve().parents[2] / "results" / "original_model"
MAX_NEW_TOKENS = 256
TEMPERATURE    = 0.1   # baja para respuestas más deterministas
DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)

# ---------------------------------------------------------------------------
# 5 prompts de evaluación humana (cubren Helpfulness, Factuality,
# Instruction-following y Creative writing según el enunciado del proyecto)
# ---------------------------------------------------------------------------
HUMAN_EVAL_PROMPTS = [
    {
        "id": 1,
        "category": "open_qa",
        "dimension": "Factuality",
        "prompt": (
            "### Instruction:\n"
            "What is the capital of Australia, and why is it often confused "
            "with Sydney?\n\n"
            "### Response:\n"
        ),
    },
    {
        "id": 2,
        "category": "general_qa",
        "dimension": "Helpfulness",
        "prompt": (
            "### Instruction:\n"
            "Give me clear step-by-step instructions to make perfect "
            "scrambled eggs.\n\n"
            "### Response:\n"
        ),
    },
    {
        "id": 3,
        "category": "classification",
        "dimension": "Instruction-following",
        "prompt": (
            "### Instruction:\n"
            "Classify each of the following words as either positive or "
            "negative sentiment: joy, anger, love, fear, hope, sadness, "
            "peace, rage. Present the result as two lists.\n\n"
            "### Response:\n"
        ),
    },
    {
        "id": 4,
        "category": "creative_writing",
        "dimension": "Helpfulness + Instruction-following",
        "prompt": (
            "### Instruction:\n"
            "Write a four-line rhyming poem about the night sky and the "
            "stars.\n\n"
            "### Response:\n"
        ),
    },
    {
        "id": 5,
        "category": "brainstorming",
        "dimension": "Helpfulness",
        "prompt": (
            "### Instruction:\n"
            "Suggest 5 creative and memorable names for a small independent "
            "coffee shop, and explain the concept behind each name.\n\n"
            "### Response:\n"
        ),
    },
]


def load_model():
    print(f"Cargando modelo '{BASE_MODEL}' en {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
    ).to(DEVICE)
    model.eval()
    print("Modelo cargado.")
    return model, tokenizer


def generate(model, tokenizer, prompt: str) -> tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0
    # Decodificar solo los tokens nuevos
    generated = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, elapsed


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    import sacrebleu
    result = sacrebleu.corpus_bleu(predictions, [references])
    return result.score / 100.0


def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure
              for pred, ref in zip(predictions, references)]
    return sum(scores) / len(scores)


def compute_bertscore(predictions: list[str], references: list[str]) -> float:
    from bert_score import score as bscore
    _, _, F1 = bscore(predictions, references, lang="en", verbose=False)
    return F1.mean().item()


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(n_samples: int = 100):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model()

    # ------------------------------------------------------------------
    # 1. Evaluación automática sobre el test set
    # ------------------------------------------------------------------
    test_rows = [
        json.loads(l)
        for l in (SPLITS_DIR / "test.jsonl").open(encoding="utf-8")
    ][:n_samples]

    print(f"\nGenerando respuestas para {n_samples} ejemplos del test set...")
    predictions_rows = []
    latencies = []

    for i, row in enumerate(test_rows):
        pred, elapsed = generate(model, tokenizer, row["prompt"])
        predictions_rows.append({
            "prompt":    row["prompt"],
            "reference": row["response"],
            "prediction": pred,
            "category":   row.get("category", ""),
            "latency_s":  round(elapsed, 3),
        })
        latencies.append(elapsed)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_samples} generados...")

    write_jsonl(RESULTS_DIR / "predictions.jsonl", predictions_rows)

    preds = [r["prediction"] for r in predictions_rows]
    refs  = [r["reference"]  for r in predictions_rows]

    print("\nCalculando métricas automáticas...")
    bleu    = compute_bleu(preds, refs)
    rouge_l = compute_rouge_l(preds, refs)
    print("  Calculando BERTScore (puede tardar un momento)...")
    bertscore = compute_bertscore(preds, refs)

    avg_latency    = sum(latencies) / len(latencies)
    total_tokens   = sum(
        len(tokenizer.encode(r["prediction"])) for r in predictions_rows
    )
    avg_throughput = total_tokens / sum(latencies)

    metrics = {
        "model":             BASE_MODEL,
        "device":            DEVICE,
        "n_samples":         n_samples,
        "BLEU":              round(bleu, 6),
        "ROUGE-L":           round(rouge_l, 6),
        "BERTScore_F1_mean": round(bertscore, 6),
        "avg_latency_s":     round(avg_latency, 3),
        "avg_throughput_tok_s": round(avg_throughput, 1),
    }
    write_json(RESULTS_DIR / "metrics.json", metrics)
    print("\nMétricas automáticas:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # 2. Evaluación humana — 5 prompts inventados
    # ------------------------------------------------------------------
    print("\nGenerando respuestas para los 5 prompts de evaluación humana...")
    human_results = []
    for item in HUMAN_EVAL_PROMPTS:
        pred, elapsed = generate(model, tokenizer, item["prompt"])
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

    print(f"\nTodos los resultados guardados en '{RESULTS_DIR}'")
    print("PASO 2 completado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_samples", type=int, default=100,
        help="Nº de ejemplos del test set a evaluar (default: 100)"
    )
    args = parser.parse_args()
    main(n_samples=args.n_samples)
