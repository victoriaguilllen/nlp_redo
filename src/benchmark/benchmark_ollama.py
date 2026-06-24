"""
Benchmark de inferencia — Ollama (API HTTP local, Metal backend)
Modelo: tinyllama (descargado con `ollama pull tinyllama`)
Métricas: latencia, tokens/s (vía eval_count / eval_duration de Ollama)

Requisito: `ollama serve` corriendo en 127.0.0.1:11434
"""

import json
import statistics
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL     = "http://127.0.0.1:11434"
MODEL_NAME     = "tinyllama"
MAX_TOKENS     = 256
N_TRIALS       = 3

PROMPTS = [
    "### Instruction:\nWhat is the capital of Australia, and why is it often confused with Sydney?\n\n### Response:\n",
    "### Instruction:\nGive me clear step-by-step instructions to make perfect scrambled eggs.\n\n### Response:\n",
    "### Instruction:\nClassify each of the following words as either positive or negative sentiment: joy, anger, love, fear, hope, sadness, peace, rage. Present the result as two lists.\n\n### Response:\n",
    "### Instruction:\nWrite a four-line rhyming poem about the night sky and the stars.\n\n### Response:\n",
    "### Instruction:\nSuggest 5 creative and memorable names for a small independent coffee shop, and explain the concept behind each name.\n\n### Response:\n",
]


def check_ollama():
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def generate_ollama(prompt: str) -> dict:
    payload = json.dumps({
        "model":   MODEL_NAME,
        "prompt":  prompt,
        "stream":  False,
        "options": {
            "temperature":   0.0,
            "num_predict":   MAX_TOKENS,
            "seed":          42,
        },
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    t1 = time.perf_counter()

    data = json.loads(raw)
    wall_latency = t1 - t0

    # Ollama reporta duración en nanosegundos
    eval_count    = data.get("eval_count", 0)
    eval_ns       = data.get("eval_duration", 1)
    tok_per_s_api = eval_count / (eval_ns / 1e9) if eval_ns > 0 else 0

    return {
        "wall_latency_s":  wall_latency,
        "eval_count":      eval_count,
        "tok_per_s":       tok_per_s_api,
        "total_duration_s": data.get("total_duration", 0) / 1e9,
        "load_duration_s":  data.get("load_duration", 0) / 1e9,
    }


def main():
    if not check_ollama():
        print("[Ollama] ERROR: el servidor no responde en 127.0.0.1:11434")
        print("  Arranca Ollama con: ollama serve")
        return

    print(f"[Ollama] Servidor OK — modelo: {MODEL_NAME}")
    print(f"  Configuración: temp=0.0, max_tokens={MAX_TOKENS}, trials={N_TRIALS}")

    # warm-up
    print("  Warm-up (primer prompt, carga modelo en Metal) ...")
    generate_ollama(PROMPTS[0])
    print("  Warm-up completado.")

    trials = []
    for trial in range(N_TRIALS):
        for pid, prompt in enumerate(PROMPTS):
            res = generate_ollama(prompt)
            trials.append({
                "trial":          trial,
                "prompt_id":      pid + 1,
                "latency_s":      round(res["wall_latency_s"], 4),
                "n_output_tokens": int(res["eval_count"]),
                "tok_per_s":      round(res["tok_per_s"], 2),
            })
            print(f"  trial={trial} p={pid+1}  {res['wall_latency_s']:.2f}s  "
                  f"{res['tok_per_s']:.1f} tok/s  ({res['eval_count']} tokens)")

    latencies = [r["latency_s"] for r in trials]
    toks      = [r["tok_per_s"] for r in trials]

    summary = {
        "framework":       "Ollama (Metal backend)",
        "model":           MODEL_NAME,
        "device":          "Apple Metal (Ollama)",
        "max_new_tokens":  MAX_TOKENS,
        "temperature":     0.0,
        "n_trials":        N_TRIALS,
        "n_prompts":       len(PROMPTS),
        "note":            "Memoria gestionada internamente por Ollama (no accesible vía API)",
        "mean_latency_s":  round(statistics.mean(latencies), 4),
        "std_latency_s":   round(statistics.stdev(latencies), 4),
        "mean_tok_per_s":  round(statistics.mean(toks), 2),
        "trials":          trials,
    }

    out_file = OUT_DIR / "benchmark_ollama.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\n[Ollama] RESUMEN")
    print(f"  Latencia media : {summary['mean_latency_s']:.3f} s")
    print(f"  Throughput     : {summary['mean_tok_per_s']:.1f} tok/s")
    print(f"  Guardado       : {out_file}")


if __name__ == "__main__":
    main()
