"""
Benchmark de inferencia — Unsloth LoRA v2 / MLX (Apple Silicon Metal)
Modelo: TinyLlama-1.1B-Chat-v1.0 + adaptador LoRA v2
Métricas: latencia, tokens/s, memoria MLX pico
"""

import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

ROOT       = Path(__file__).resolve().parents[2]
OUT_DIR    = ROOT / "results" / "benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_DIR  = str(ROOT / "results" / "unsloth_lora_v2")
MAX_TOKENS   = 256
N_TRIALS     = 3
SAMPLER      = make_sampler(temp=0.0)

PROMPTS = [
    "### Instruction:\nWhat is the capital of Australia, and why is it often confused with Sydney?\n\n### Response:\n",
    "### Instruction:\nGive me clear step-by-step instructions to make perfect scrambled eggs.\n\n### Response:\n",
    "### Instruction:\nClassify each of the following words as either positive or negative sentiment: joy, anger, love, fear, hope, sadness, peace, rage. Present the result as two lists.\n\n### Response:\n",
    "### Instruction:\nWrite a four-line rhyming poem about the night sky and the stars.\n\n### Response:\n",
    "### Instruction:\nSuggest 5 creative and memorable names for a small independent coffee shop, and explain the concept behind each name.\n\n### Response:\n",
]


def count_tokens(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text)
    return len(ids) if isinstance(ids, list) else ids.shape[-1]


def main():
    print(f"[Unsloth/MLX v2] Cargando {BASE_MODEL} + adaptador v2 ...")
    model, tokenizer = load(BASE_MODEL, adapter_path=ADAPTER_DIR)
    mx.eval(model.parameters())

    mem_after_load = mx.metal.get_active_memory() / 1e6 if hasattr(mx, "metal") else 0
    print(f"  Modelo cargado — memoria activa: {mem_after_load:.0f} MB")

    print("  Warm-up ...")
    generate(model, tokenizer, prompt=PROMPTS[0], max_tokens=32,
             sampler=SAMPLER, verbose=False)

    trials = []
    for trial in range(N_TRIALS):
        for pid, prompt in enumerate(PROMPTS):
            if hasattr(mx, "metal"):
                mx.metal.reset_peak_memory()

            t0 = time.perf_counter()
            response = generate(model, tokenizer, prompt=prompt,
                                max_tokens=MAX_TOKENS, sampler=SAMPLER,
                                verbose=False)
            mx.eval()
            t1 = time.perf_counter()

            n_output  = count_tokens(tokenizer, response)
            latency_s = t1 - t0
            tok_per_s = n_output / latency_s if latency_s > 0 else 0

            peak_mb = (mx.metal.get_peak_memory() / 1e6
                       if hasattr(mx, "metal") else 0)

            trials.append({
                "trial":           trial,
                "prompt_id":       pid + 1,
                "latency_s":       round(latency_s, 4),
                "n_output_tokens": int(n_output),
                "tok_per_s":       round(tok_per_s, 2),
                "peak_memory_mb":  round(peak_mb, 1),
            })
            print(f"  trial={trial} p={pid+1}  {latency_s:.2f}s  "
                  f"{tok_per_s:.1f} tok/s  {peak_mb:.0f} MB")

    latencies = [r["latency_s"]      for r in trials]
    toks      = [r["tok_per_s"]      for r in trials]
    mems      = [r["peak_memory_mb"] for r in trials]

    summary = {
        "framework":           "Unsloth / mlx_lm v2 (Metal + LoRA adapter)",
        "model":               BASE_MODEL,
        "adapter":             ADAPTER_DIR,
        "device":              "Apple Metal (MLX)",
        "max_new_tokens":      MAX_TOKENS,
        "temperature":         0.0,
        "n_trials":            N_TRIALS,
        "n_prompts":           len(PROMPTS),
        "model_load_mb":       round(mem_after_load, 1),
        "mean_latency_s":      round(statistics.mean(latencies), 4),
        "std_latency_s":       round(statistics.stdev(latencies), 4),
        "mean_tok_per_s":      round(statistics.mean(toks), 2),
        "mean_peak_memory_mb": round(statistics.mean(mems), 1),
        "trials":              trials,
    }

    out_file = OUT_DIR / "benchmark_unsloth_v2.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\n[Unsloth/MLX v2] RESUMEN")
    print(f"  Latencia media : {summary['mean_latency_s']:.3f} s")
    print(f"  Throughput     : {summary['mean_tok_per_s']:.1f} tok/s")
    print(f"  Memoria pico   : {summary['mean_peak_memory_mb']:.0f} MB")
    print(f"  Guardado       : {out_file}")


if __name__ == "__main__":
    main()
