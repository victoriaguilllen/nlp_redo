"""
Inferencia interactiva — Unsloth LoRA v2 (MLX backend)

Carga el modelo fine-tuned v2 y genera respuestas para un conjunto
de prompts de demostración, mostrando comparativa con v1.

Uso:
    python3.11 src/eval/infer_unsloth_v2.py
"""

import json
import time
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[2]
ADAPTER_V2 = ROOT / "results" / "unsloth_lora_v2"
ADAPTER_V1 = ROOT / "results" / "unsloth_lora"

DEMO_PROMPTS = [
    {
        "label": "Factualidad — Capital Australia",
        "prompt": "### Instruction:\nWhat is the capital of Australia, and why is it often confused with Sydney?\n\n### Response:\n",
    },
    {
        "label": "Instruction-following — Clasificación sentimiento",
        "prompt": "### Instruction:\nClassify each of the following words as either positive or negative sentiment: joy, anger, love, fear, hope, sadness, peace, rage. Present the result as two lists.\n\n### Response:\n",
    },
    {
        "label": "Helpfulness — Huevos revueltos",
        "prompt": "### Instruction:\nGive me clear step-by-step instructions to make perfect scrambled eggs.\n\n### Response:\n",
    },
    {
        "label": "Escritura creativa — Poema noche",
        "prompt": "### Instruction:\nWrite a four-line rhyming poem about the night sky and the stars.\n\n### Response:\n",
    },
    {
        "label": "Brainstorming — Nombres cafetería",
        "prompt": "### Instruction:\nSuggest 5 creative and memorable names for a small independent coffee shop, and explain the concept behind each name.\n\n### Response:\n",
    },
    {
        "label": "EXTRA — Explicación IA",
        "prompt": "### Instruction:\nExplain what a neural network is in simple terms, as if talking to a 10-year-old.\n\n### Response:\n",
    },
    {
        "label": "EXTRA — Receta española",
        "prompt": "### Instruction:\nHow do you make a traditional Spanish tortilla?\n\n### Response:\n",
    },
]


def load_model(adapter_dir: Path):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=512,
        load_in_4bit=False,
    )
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_tokens: int = 256) -> tuple[str, float]:
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    t0 = time.perf_counter()
    result = mlx_generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=0.0),
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    if result.startswith(prompt):
        result = result[len(prompt):]
    return result.strip(), elapsed


def main():
    print("\n" + "="*60)
    print("  INFERENCIA — Unsloth LoRA v2 (MLX, Apple Silicon)")
    print("="*60)

    print("\nCargando modelo v2...")
    model, tokenizer = load_model(ADAPTER_V2)
    print("Listo.\n")

    results = []

    for item in DEMO_PROMPTS:
        print(f"\n{'─'*60}")
        print(f"  {item['label']}")
        print(f"{'─'*60}")

        response, elapsed = generate(model, tokenizer, item["prompt"])
        tok_s = round(len(response) / 4 / elapsed, 1)

        print(f"RESPUESTA ({elapsed:.1f}s | ~{tok_s} tok/s):\n{response}\n")

        results.append({
            "label":    item["label"],
            "prompt":   item["prompt"],
            "response": response,
            "latency_s": round(elapsed, 3),
            "tok_s":    tok_s,
        })

    out_path = ADAPTER_V2 / "eval" / "inference_demo.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"  Guardado en: {out_path}")
    print(f"  Prompts evaluados: {len(results)}")
    avg_lat = sum(r['latency_s'] for r in results) / len(results)
    print(f"  Latencia media:   {avg_lat:.2f}s")
    print("="*60)


if __name__ == "__main__":
    main()
