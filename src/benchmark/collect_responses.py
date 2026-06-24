"""
Captura las respuestas textuales de cada framework para los 5 prompts.
Uso: python src/collect_responses.py --framework [transformers|unsloth|ollama|vllm|all]
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_NEW_TOKENS = 256

PROMPT_LABELS = [
    "Capital de Australia",
    "Huevos revueltos",
    "Clasificación sentimiento",
    "Poema 4 líneas",
    "Nombres cafetería",
]

PROMPTS = [
    "### Instruction:\nWhat is the capital of Australia, and why is it often confused with Sydney?\n\n### Response:\n",
    "### Instruction:\nGive me clear step-by-step instructions to make perfect scrambled eggs.\n\n### Response:\n",
    "### Instruction:\nClassify each of the following words as either positive or negative sentiment: joy, anger, love, fear, hope, sadness, peace, rage. Present the result as two lists.\n\n### Response:\n",
    "### Instruction:\nWrite a four-line rhyming poem about the night sky and the stars.\n\n### Response:\n",
    "### Instruction:\nSuggest 5 creative and memorable names for a small independent coffee shop, and explain the concept behind each name.\n\n### Response:\n",
]


# ── Transformers ──────────────────────────────────────────────────────────────
def run_transformers():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Cargando modelo en {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(DEVICE)
    model.eval()

    responses = []
    for i, (label, prompt) in enumerate(zip(PROMPT_LABELS, PROMPTS)):
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        n_in = inputs["input_ids"].shape[1]
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        if DEVICE == "mps":
            torch.mps.synchronize()
        latency = time.perf_counter() - t0
        text = tokenizer.decode(out[0][n_in:], skip_special_tokens=True).strip()
        responses.append({"prompt_id": i+1, "label": label, "response": text, "latency_s": round(latency,3)})
        print(f"  [{i+1}/5] {label}: {latency:.2f}s")
    return responses


# ── Unsloth / MLX ─────────────────────────────────────────────────────────────
def run_unsloth():
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print("  Cargando modelo MLX...")
    model, tokenizer = load(MODEL)
    sampler = make_sampler(temp=0.0)

    responses = []
    for i, (label, prompt) in enumerate(zip(PROMPT_LABELS, PROMPTS)):
        t0 = time.perf_counter()
        text = generate(model, tokenizer, prompt=prompt,
                        max_tokens=MAX_NEW_TOKENS, sampler=sampler, verbose=False)
        latency = time.perf_counter() - t0
        responses.append({"prompt_id": i+1, "label": label, "response": text.strip(), "latency_s": round(latency,3)})
        print(f"  [{i+1}/5] {label}: {latency:.2f}s")
    return responses


# ── Ollama ────────────────────────────────────────────────────────────────────
def run_ollama():
    OLLAMA_URL = "http://127.0.0.1:11434"
    MODEL = "tinyllama"

    responses = []
    for i, (label, prompt) in enumerate(zip(PROMPT_LABELS, PROMPTS)):
        payload = json.dumps({
            "model": MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.0, "num_predict": MAX_NEW_TOKENS, "seed": 42},
        }).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        latency = time.perf_counter() - t0
        text = data.get("response", "").strip()
        responses.append({"prompt_id": i+1, "label": label, "response": text, "latency_s": round(latency,3)})
        print(f"  [{i+1}/5] {label}: {latency:.2f}s")
    return responses


# ── vLLM ──────────────────────────────────────────────────────────────────────
def run_vllm():
    from vllm import LLM, SamplingParams
    MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print("  Cargando motor vLLM (CPU)...")
    llm = LLM(model=MODEL, dtype="float32", max_model_len=1024)
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, seed=42)

    responses = []
    for i, (label, prompt) in enumerate(zip(PROMPT_LABELS, PROMPTS)):
        t0 = time.perf_counter()
        out = llm.generate([prompt], sampling)
        latency = time.perf_counter() - t0
        text = out[0].outputs[0].text.strip()
        responses.append({"prompt_id": i+1, "label": label, "response": text, "latency_s": round(latency,3)})
        print(f"  [{i+1}/5] {label}: {latency:.2f}s")
    return responses


RUNNERS = {
    "transformers": run_transformers,
    "unsloth":      run_unsloth,
    "ollama":       run_ollama,
    "vllm":         run_vllm,
}

FRAMEWORK_LABELS = {
    "transformers": "Transformers (PyTorch MPS)",
    "unsloth":      "Unsloth / MLX",
    "ollama":       "Ollama (Metal)",
    "vllm":         "vLLM (CPU)",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", default="all",
                        choices=list(RUNNERS.keys()) + ["all"])
    args = parser.parse_args()

    targets = list(RUNNERS.keys()) if args.framework == "all" else [args.framework]

    all_results = {}
    # Si ya hay resultados guardados, los cargamos
    out_file = OUT_DIR / "responses_by_framework.json"
    if out_file.exists():
        all_results = json.loads(out_file.read_text())

    for fw in targets:
        print(f"\n{'='*55}")
        print(f"  Framework: {FRAMEWORK_LABELS[fw]}")
        print(f"{'='*55}")
        try:
            responses = RUNNERS[fw]()
            all_results[fw] = {
                "framework_label": FRAMEWORK_LABELS[fw],
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0.0,
                "responses": responses,
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[fw] = {"framework_label": FRAMEWORK_LABELS[fw], "error": str(e)}

        # Guardamos tras cada framework (por si se interrumpe)
        out_file.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        print(f"  Guardado parcial: {out_file.name}")

    # Tabla resumen por pantalla
    print(f"\n{'='*70}")
    print("  RESPUESTAS POR PROMPT Y FRAMEWORK")
    print(f"{'='*70}")
    for pid, label in enumerate(PROMPT_LABELS):
        print(f"\n── Prompt {pid+1}: {label} {'─'*30}")
        print(f"  Instrucción: {PROMPTS[pid].split(chr(10))[1]}")
        for fw in targets:
            fw_data = all_results.get(fw, {})
            if "error" in fw_data:
                print(f"\n  [{FRAMEWORK_LABELS.get(fw, fw)}] ERROR: {fw_data['error']}")
                continue
            resp = next((r for r in fw_data.get("responses", []) if r["prompt_id"] == pid+1), None)
            if resp:
                preview = resp["response"][:300] + ("..." if len(resp["response"]) > 300 else "")
                print(f"\n  [{FRAMEWORK_LABELS.get(fw, fw)}] ({resp['latency_s']}s)")
                print(f"  {preview}")

    print(f"\nGuardado completo: {out_file}")


if __name__ == "__main__":
    main()
