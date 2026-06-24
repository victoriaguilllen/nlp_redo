"""
Benchmark de inferencia — vLLM
Modelo: TinyLlama-1.1B-Chat-v1.0

NOTA: vLLM requiere CUDA (Linux + GPU NVIDIA) para obtener su máximo rendimiento.
En macOS puede ejecutarse en modo CPU pero con rendimiento muy degradado.
Si vLLM no está instalado o no puede ejecutarse, este script documenta
la limitación y genera un resultado explicativo.
"""

import json
import statistics
import time
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_TOKENS     = 256
N_TRIALS       = 3

PROMPTS = [
    "### Instruction:\nWhat is the capital of Australia, and why is it often confused with Sydney?\n\n### Response:\n",
    "### Instruction:\nGive me clear step-by-step instructions to make perfect scrambled eggs.\n\n### Response:\n",
    "### Instruction:\nClassify each of the following words as either positive or negative sentiment: joy, anger, love, fear, hope, sadness, peace, rage. Present the result as two lists.\n\n### Response:\n",
    "### Instruction:\nWrite a four-line rhyming poem about the night sky and the stars.\n\n### Response:\n",
    "### Instruction:\nSuggest 5 creative and memorable names for a small independent coffee shop, and explain the concept behind each name.\n\n### Response:\n",
]


def run_vllm_benchmark():
    from vllm import LLM, SamplingParams

    print(f"[vLLM] Cargando {MODEL_NAME} (modo CPU en macOS) ...")
    llm = LLM(model=MODEL_NAME, dtype="float32", max_model_len=1024)
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, seed=42)

    print("  Warm-up ...")
    llm.generate([PROMPTS[0]], sampling)

    trials = []
    for trial in range(N_TRIALS):
        for pid, prompt in enumerate(PROMPTS):
            t0 = time.perf_counter()
            outputs = llm.generate([prompt], sampling)
            t1 = time.perf_counter()

            n_output  = len(outputs[0].outputs[0].token_ids)
            latency_s = t1 - t0
            tok_per_s = n_output / latency_s if latency_s > 0 else 0

            trials.append({
                "trial":          trial,
                "prompt_id":      pid + 1,
                "latency_s":      round(latency_s, 4),
                "n_output_tokens": int(n_output),
                "tok_per_s":      round(tok_per_s, 2),
            })
            print(f"  trial={trial} p={pid+1}  {latency_s:.2f}s  {tok_per_s:.1f} tok/s")

    latencies = [r["latency_s"] for r in trials]
    toks      = [r["tok_per_s"] for r in trials]
    return {
        "framework":       "vLLM (CPU mode — macOS)",
        "model":           MODEL_NAME,
        "device":          "CPU (vLLM no soporta MPS; CUDA requerido para GPU)",
        "max_new_tokens":  MAX_TOKENS,
        "temperature":     0.0,
        "n_trials":        N_TRIALS,
        "n_prompts":       len(PROMPTS),
        "mean_latency_s":  round(statistics.mean(latencies), 4),
        "std_latency_s":   round(statistics.stdev(latencies), 4),
        "mean_tok_per_s":  round(statistics.mean(toks), 2),
        "trials":          trials,
    }


def not_available_result(reason: str) -> dict:
    return {
        "framework":      "vLLM",
        "model":          MODEL_NAME,
        "device":         "N/A",
        "max_new_tokens": MAX_TOKENS,
        "temperature":    0.0,
        "available":      False,
        "reason":         reason,
        "note": (
            "vLLM está diseñado para servidores Linux con GPU NVIDIA (CUDA). "
            "En macOS no hay soporte oficial para MPS/Metal. "
            "En producción con hardware NVIDIA, vLLM alcanza >1000 tok/s "
            "gracias a PagedAttention y continuous batching."
        ),
        "reference_throughput_linux_a100": ">2000 tok/s (TinyLlama en A100 con vLLM)",
    }


def main():
    try:
        summary = run_vllm_benchmark()
        print(f"\n[vLLM] Latencia media : {summary['mean_latency_s']:.3f} s")
        print(f"[vLLM] Throughput     : {summary['mean_tok_per_s']:.1f} tok/s")
    except ImportError:
        print("[vLLM] No instalado en este entorno.")
        print("  vLLM requiere Linux + CUDA para funcionamiento completo.")
        summary = not_available_result(
            "vLLM no está instalado. Requiere pip install vllm en entorno Linux/CUDA."
        )
    except Exception as e:
        print(f"[vLLM] Error al ejecutar: {e}")
        summary = not_available_result(f"Error de ejecución en macOS: {e}")

    out_file = OUT_DIR / "benchmark_vllm.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"  Guardado: {out_file}")


if __name__ == "__main__":
    main()
