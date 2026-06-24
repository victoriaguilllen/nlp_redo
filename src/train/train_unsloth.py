"""
PASO 4 — Fine-tuning con Unsloth (MLX backend en Apple Silicon)

En macOS con Apple Silicon, Unsloth enruta automáticamente a través de MLX
(el framework ML nativo de Apple) en lugar de CUDA. Esto es Unsloth oficial
para Mac: mismos hiperparámetros que el experimento Transformers para comparación justa.

Diferencias clave respecto al experimento Transformers:
  - Backend: MLX (nativo Apple Silicon) vs. PyTorch + MPS
  - Trainer: MLXTrainer (Unsloth) vs. HF Trainer
  - Compilación de kernels: MLX JIT vs. PyTorch MPS JIT
  - Precisión: bfloat16 MLX vs. float16 PyTorch

Hiperparámetros idénticos al experimento Transformers:
  - LoRA rank=16, alpha=32, dropout=0.05
  - Mismos módulos objetivo (q,k,v,o,gate,up,down proj)
  - lr=2e-4, cosine scheduler, warmup ~5%, 1 época, 2000 muestras

Uso:
    python src/train_unsloth.py
    python src/train_unsloth.py --max_train_samples 500
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import psutil
from datasets import Dataset
from unsloth import FastLanguageModel, MLXTrainer, MLXTrainingConfig

SPLITS_DIR  = Path(__file__).resolve().parents[2] / "data" / "splits"
OUTPUT_DIR  = Path(__file__).resolve().parents[2] / "results" / "unsloth_lora"
MAX_SEQ_LEN = 512

LORA_CONFIG = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
)


def get_ram_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 ** 2


def get_mlx_mb() -> float:
    return mx.get_active_memory() / 1024 ** 2


def get_mlx_peak_mb() -> float:
    return mx.get_peak_memory() / 1024 ** 2


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def format_as_text(ex: dict) -> dict:
    """Unsloth MLX espera un campo 'text' con prompt+response completo."""
    return {"text": ex["prompt"] + ex["response"]}


def main(max_train_samples: int = 2000):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mx.reset_peak_memory()
    ram_start = get_ram_mb()

    print(f"Backend: Unsloth MLX (Apple Silicon)")
    print(f"MLX version: {mx.__version__}")
    print(f"Muestras de entrenamiento: {max_train_samples}")

    # ── 1. Cargar modelo con Unsloth ──────────────────────────────────────
    print("\nCargando modelo con FastLanguageModel (Unsloth MLX)...")
    mlx_before = get_mlx_mb()
    t_load_start = time.perf_counter()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
    )

    t_load_end = time.perf_counter()
    mlx_after_load = get_mlx_mb()
    print(f"  Modelo cargado en {t_load_end - t_load_start:.1f}s")
    print(f"  MLX memoria tras carga: {mlx_after_load:.0f} MB")

    # ── 2. Aplicar LoRA con Unsloth ───────────────────────────────────────
    print("Aplicando LoRA con FastLanguageModel.get_peft_model...")
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)

    # ── 3. Dataset ────────────────────────────────────────────────────────
    print(f"\nCargando {max_train_samples} ejemplos de train...")
    raw = load_jsonl(SPLITS_DIR / "train.jsonl")[:max_train_samples]
    ds = Dataset.from_list([format_as_text(r) for r in raw])

    # Calcular warmup_steps ≈ 5% del total de pasos optimizer
    steps_per_epoch = max_train_samples // (2 * 4)   # batch=2, grad_accum=4
    warmup_steps = max(1, int(steps_per_epoch * 0.05))

    # ── 4. Configurar entrenamiento ───────────────────────────────────────
    config = MLXTrainingConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,       # batch efectivo = 8
        num_train_epochs=1,
        max_steps=-1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        logging_steps=25,
        report_to="none",
        seed=42,
    )

    trainer = MLXTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=config,
    )

    # ── 5. Entrenar ───────────────────────────────────────────────────────
    print("\nIniciando entrenamiento con Unsloth MLXTrainer...")
    mx.reset_peak_memory()
    t_start = time.perf_counter()

    train_result = trainer.train()

    t_end = time.perf_counter()
    training_time_s = t_end - t_start
    mlx_peak = get_mlx_peak_mb()
    ram_peak  = get_ram_mb()

    # ── 6. Guardar modelo ─────────────────────────────────────────────────
    print("\nGuardando modelo Unsloth...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # ── 7. Registrar experimento ──────────────────────────────────────────
    # Extraer loss del resultado
    if hasattr(train_result, 'training_loss'):
        final_loss = train_result.training_loss
    elif isinstance(train_result, dict):
        final_loss = train_result.get('train_loss', train_result.get('loss', None))
    else:
        final_loss = None

    log = {
        "model":             "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "framework":         "Unsloth (MLX backend, Apple Silicon)",
        "mlx_version":       str(mx.__version__),
        "max_train_samples": max_train_samples,
        "lora":              LORA_CONFIG,
        "training": {
            "epochs":                1,
            "batch_size_device":     2,
            "gradient_accumulation": 4,
            "effective_batch_size":  8,
            "learning_rate":         2e-4,
            "scheduler":             "cosine",
            "warmup_steps":          warmup_steps,
            "max_seq_len":           MAX_SEQ_LEN,
        },
        "results": {
            "training_loss":     round(final_loss, 4) if final_loss else None,
            "training_time_s":   round(training_time_s, 1),
            "training_time_min": round(training_time_s / 60, 2),
        },
        "memory": {
            "mlx_after_load_mb":  round(mlx_after_load, 1),
            "mlx_peak_train_mb":  round(mlx_peak, 1),
            "ram_start_mb":       round(ram_start, 1),
            "ram_peak_mb":        round(ram_peak, 1),
            "ram_delta_mb":       round(ram_peak - ram_start, 1),
        },
    }

    log_path = OUTPUT_DIR / "training_log.json"
    with log_path.open("w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print("\n── Resumen del entrenamiento Unsloth ──────────────────────")
    print(f"  Loss final         : {final_loss}")
    print(f"  Tiempo total       : {training_time_s/60:.2f} min")
    print(f"  MLX peak memoria   : {mlx_peak:.0f} MB")
    print(f"  RAM delta          : {ram_peak - ram_start:.0f} MB")
    print(f"\nModelo guardado en '{OUTPUT_DIR}'")
    print("PASO 4 completado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_samples", type=int, default=2000)
    args = parser.parse_args()
    main(max_train_samples=args.max_train_samples)
