"""
PASO C — Evaluación del modelo fine-tuneado (post fine-tuning)

Carga TinyLlama + adaptador LoRA y evalúa sobre:
  - Los mismos 100 ejemplos del test set que se usaron en el baseline
  - Los mismos 5 prompts de evaluación humana del Paso 2

Resultados guardados en: results/transformers_lora/eval/

Uso:
    python src/evaluate_finetuned.py
    python src/evaluate_finetuned.py --n_samples 50
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR  = Path(__file__).resolve().parents[2] / "results" / "transformers_lora"
SPLITS_DIR   = Path(__file__).resolve().parents[2] / "data" / "splits"
BASELINE_DIR = Path(__file__).resolve().parents[2] / "results" / "original_model"
RESULTS_DIR  = ADAPTER_DIR / "eval"

MAX_NEW_TOKENS = 256
DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)

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


def load_model():
    print(f"Cargando modelo base '{BASE_MODEL}' en {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float16
    ).to(DEVICE)
    print(f"Cargando adaptador LoRA desde '{ADAPTER_DIR}'...")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()
    print("Modelo fine-tuneado listo.")
    return model, tokenizer


def generate(model, tokenizer, prompt: str) -> tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - t0
    generated = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, elapsed


def compute_bleu(predictions, references) -> float:
    import sacrebleu
    return sacrebleu.corpus_bleu(predictions, [references]).score / 100.0


def compute_rouge_l(predictions, references) -> float:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)]
    return sum(scores) / len(scores)


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
    model, tokenizer = load_model()

    # ── 1. Evaluación automática ───────────────────────────────────────────
    test_rows = [
        json.loads(l)
        for l in (SPLITS_DIR / "test.jsonl").open(encoding="utf-8")
    ][:n_samples]

    print(f"\nGenerando respuestas del modelo fine-tuneado ({n_samples} ejemplos)...")
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

    print("\nCalculando métricas automáticas...")
    bleu      = compute_bleu(preds, refs)
    rouge_l   = compute_rouge_l(preds, refs)
    print("  BERTScore...")
    bertscore = compute_bertscore(preds, refs)

    avg_lat = sum(latencies) / len(latencies)
    total_tok = sum(len(tokenizer.encode(r["prediction"])) for r in pred_rows)
    throughput = total_tok / sum(latencies)

    metrics = {
        "model":                "TinyLlama + LoRA adapter",
        "adapter":              str(ADAPTER_DIR),
        "device":               DEVICE,
        "n_samples":            n_samples,
        "BLEU":                 round(bleu, 6),
        "ROUGE-L":              round(rouge_l, 6),
        "BERTScore_F1_mean":    round(bertscore, 6),
        "avg_latency_s":        round(avg_lat, 3),
        "avg_throughput_tok_s": round(throughput, 1),
    }
    write_json(RESULTS_DIR / "metrics.json", metrics)

    # ── 2. Cargar baseline para comparar ──────────────────────────────────
    baseline_path = BASELINE_DIR / "metrics.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}

    comparison = {
        "metric":       ["BLEU", "ROUGE-L", "BERTScore F1", "Latencia (s)", "Throughput (tok/s)"],
        "original":     [
            baseline.get("BLEU", "-"),
            baseline.get("ROUGE-L", "-"),
            baseline.get("BERTScore_F1_mean", "-"),
            baseline.get("avg_latency_s", "-"),
            baseline.get("avg_throughput_tok_s", "-"),
        ],
        "finetuned":    [
            metrics["BLEU"],
            metrics["ROUGE-L"],
            metrics["BERTScore_F1_mean"],
            metrics["avg_latency_s"],
            metrics["avg_throughput_tok_s"],
        ],
    }
    write_json(RESULTS_DIR / "comparison.json", comparison)

    print("\n── Comparación pre vs. post fine-tuning ────────────────────")
    for m, o, f in zip(comparison["metric"], comparison["original"], comparison["finetuned"]):
        delta = ""
        if isinstance(o, float) and isinstance(f, float):
            delta = f"  ({f-o:+.4f})"
        print(f"  {m:<22} original={o}  finetuned={f}{delta}")

    # ── 3. Evaluación humana — mismos 5 prompts ────────────────────────────
    print("\nGenerando respuestas humanas del modelo fine-tuneado...")
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
    print(f"\nResultados guardados en '{RESULTS_DIR}'")
    print("Evaluación del modelo fine-tuneado completada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()
    main(n_samples=args.n_samples)
