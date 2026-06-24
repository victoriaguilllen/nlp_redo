"""
Ejecuta los 4 benchmarks de inferencia y genera la tabla comparativa final.
Uso: python src/benchmark/run_all_benchmarks.py
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[2]
BENCH_DIR  = ROOT / "src" / "benchmark"
OUT_DIR    = ROOT / "results" / "benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    ("Transformers (MPS)",  BENCH_DIR / "benchmark_transformers.py"),
    ("Unsloth / MLX",       BENCH_DIR / "benchmark_unsloth.py"),
    ("vLLM",                BENCH_DIR / "benchmark_vllm.py"),
    ("Ollama (Metal)",      BENCH_DIR / "benchmark_ollama.py"),
]

RESULT_FILES = {
    "Transformers (MPS)": OUT_DIR / "benchmark_transformers.json",
    "Unsloth / MLX":      OUT_DIR / "benchmark_unsloth.json",
    "vLLM":               OUT_DIR / "benchmark_vllm.json",
    "Ollama (Metal)":     OUT_DIR / "benchmark_ollama.json",
}


def run_script(name: str, path: Path):
    print(f"\n{'='*60}")
    print(f"  Ejecutando: {name}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  [AVISO] {name} terminó con código {result.returncode}")


def print_summary():
    print(f"\n{'='*70}")
    print("  TABLA COMPARATIVA — Benchmark de inferencia (Paso 6)")
    print(f"{'='*70}")
    fmt = "{:<28} {:>12} {:>12} {:>14}"
    print(fmt.format("Framework", "Latencia (s)", "Tok/s", "Memoria (MB)"))
    print("-" * 70)

    rows = []
    for label, fpath in RESULT_FILES.items():
        if not fpath.exists():
            print(fmt.format(label, "—", "—", "—"))
            continue
        d = json.loads(fpath.read_text())
        available = d.get("available", True)
        if not available:
            lat  = "N/A (CUDA)"
            toks = "N/A (CUDA)"
            mem  = "N/A"
        else:
            lat  = f"{d.get('mean_latency_s', 0):.3f}"
            toks = f"{d.get('mean_tok_per_s', 0):.1f}"
            mem  = str(int(d.get('mean_peak_memory_mb', 0))) if d.get('mean_peak_memory_mb') else "—"
        print(fmt.format(label, lat, toks, mem))
        rows.append({"framework": label, "latency_s": lat, "tok_per_s": toks, "memory_mb": mem})

    out = OUT_DIR / "benchmark_summary.json"
    out.write_text(json.dumps({"results": rows}, indent=2))
    print(f"\nResumen guardado: {out}")


if __name__ == "__main__":
    for name, script in SCRIPTS:
        run_script(name, script)
    print_summary()
