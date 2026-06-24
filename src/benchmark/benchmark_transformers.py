"""
Benchmark de inferencia — HuggingFace Transformers (PyTorch MPS)
Modelo: TinyLlama-1.1B-Chat-v1.0 (sin adapter LoRA, inferencia pura)
Métricas: latencia, tokens/s, memoria MPS pico
"""

import json
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT       = Path(__file__).resolve().parents[2]
OUT_DIR    = ROOT / "results" / "benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_NEW_TOKENS = 256
N_TRIALS       = 3
DEVICE         = "mps" if torch.backends.mps.is_available() else "cpu"

PROMPTS = [
    "### Instruction:\nWhat is the capital of Australia, and why is it often confused with Sydney?\n\n### Response:\n",
    "### Instruction:\nGive me clear step-by-step instructions to make perfect scrambled eggs.\n\n### Response:\n",
    "### Instruction:\nClassify each of the following words as either positive or negative sentiment: joy, anger, love, fear, hope, sadness, peace, rage. Present the result as two lists.\n\n### Response:\n",
    "### Instruction:\nWrite a four-line rhyming poem about the night sky and the stars.\n\n### Response:\n",
    "### Instruction:\nSuggest 5 creative and memorable names for a small independent coffee shop, and explain the concept behind each name.\n\n### Response:\n",
]


def main():
    print(f"[Transformers] Cargando {MODEL_NAME} en {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16
    ).to(DEVICE)
    model.eval()

    # memoria del modelo recién cargado
    if DEVICE == "mps":
        model_mem_mb = torch.mps.current_allocated_memory() / 1e6
    else:
        model_mem_mb = psutil.Process().memory_info().rss / 1e6

    print(f"  Modelo cargado — memoria base: {model_mem_mb:.0f} MB")

    # warm-up (1 pase descartado para compilar kernels MPS)
    print("  Warm-up ...")
    with torch.no_grad():
        inp = tokenizer(PROMPTS[0], return_tensors="pt").to(DEVICE)
        model.generate(**inp, max_new_tokens=32, do_sample=False,
                       pad_token_id=tokenizer.eos_token_id)

    trials = []
    for trial in range(N_TRIALS):
        for pid, prompt in enumerate(PROMPTS):
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            n_input = inputs["input_ids"].shape[1]

            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            if DEVICE == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()

            n_output  = out.shape[1] - n_input
            latency_s = t1 - t0
            tok_per_s = n_output / latency_s if latency_s > 0 else 0

            peak_mb = (torch.mps.driver_allocated_memory() / 1e6
                       if DEVICE == "mps" else
                       psutil.Process().memory_info().rss / 1e6)

            trials.append({
                "trial":         trial,
                "prompt_id":     pid + 1,
                "latency_s":     round(latency_s, 4),
                "n_output_tokens": int(n_output),
                "tok_per_s":     round(tok_per_s, 2),
                "peak_memory_mb": round(peak_mb, 1),
            })
            print(f"  trial={trial} p={pid+1}  {latency_s:.2f}s  "
                  f"{tok_per_s:.1f} tok/s  {peak_mb:.0f} MB")

    latencies = [r["latency_s"]     for r in trials]
    toks      = [r["tok_per_s"]     for r in trials]
    mems      = [r["peak_memory_mb"] for r in trials]
    import statistics
    summary = {
        "framework":           "Transformers (PyTorch MPS)",
        "model":               MODEL_NAME,
        "device":              DEVICE,
        "max_new_tokens":      MAX_NEW_TOKENS,
        "temperature":         0.0,
        "n_trials":            N_TRIALS,
        "n_prompts":           len(PROMPTS),
        "model_load_mb":       round(model_mem_mb, 1),
        "mean_latency_s":      round(statistics.mean(latencies), 4),
        "std_latency_s":       round(statistics.stdev(latencies), 4),
        "mean_tok_per_s":      round(statistics.mean(toks), 2),
        "mean_peak_memory_mb": round(statistics.mean(mems), 1),
        "trials":              trials,
    }

    out_file = OUT_DIR / "benchmark_transformers.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\n[Transformers] RESUMEN")
    print(f"  Latencia media : {summary['mean_latency_s']:.3f} s")
    print(f"  Throughput     : {summary['mean_tok_per_s']:.1f} tok/s")
    print(f"  Memoria pico   : {summary['mean_peak_memory_mb']:.0f} MB")
    print(f"  Guardado       : {out_file}")


if __name__ == "__main__":
    main()
