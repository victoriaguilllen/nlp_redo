"""
EXPERIMENTO V2 — Fine-tuning con Unsloth (MLX backend en Apple Silicon)

Mejoras respecto a v1 (train_unsloth.py):
  1. max_train_samples = 6000 (vs 2000) → 3x más datos
  2. num_train_epochs  = 2    (vs 1)    → el modelo ve cada ejemplo 2 veces
  3. logging_steps     = 5   (vs 25)   → curva de loss granular
  4. Output en results/unsloth_lora_v2/ → no sobreescribe v1

Tiempo estimado: ~40 min en Apple M4 Pro MLX (2.56× más rápido que Transformers).

Uso:
    python3.11 src/train/train_unsloth_v2.py
    python3.11 src/train/train_unsloth_v2.py --max_train_samples 3000 --epochs 1
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import psutil
from datasets import Dataset
from unsloth import FastLanguageModel, MLXTrainer, MLXTrainingConfig

ROOT        = Path(__file__).resolve().parents[2]
SPLITS_DIR  = ROOT / "data" / "splits"
OUTPUT_DIR  = ROOT / "results" / "unsloth_lora_v2"
MAX_SEQ_LEN = 384

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


def load_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def format_as_text(ex: dict) -> dict:
    return {"text": ex["prompt"] + ex["response"]}


def main(max_train_samples: int = 6000, num_epochs: int = 2):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mx.reset_peak_memory()
    ram_start = get_ram_mb()

    print(f"\n{'='*60}")
    print(f"  Unsloth LoRA v2 — experimento mejorado")
    print(f"{'='*60}")
    print(f"  Backend            : Unsloth MLX (Apple Silicon)")
    print(f"  MLX version        : {mx.__version__}")
    print(f"  Muestras           : {max_train_samples}  (v1 tenía 2000)")
    print(f"  Épocas             : {num_epochs}         (v1 tenía 1)")
    print(f"  Output             : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # ── Cargar modelo ──────────────────────────────────────────────────────────
    print("Cargando modelo con FastLanguageModel (Unsloth MLX)...")
    mlx_before = get_mlx_mb()
    t0 = time.perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
    )
    mlx_after_load = get_mlx_mb()
    print(f"  Cargado en {time.perf_counter()-t0:.1f}s — MLX: {mlx_after_load:.0f} MB")

    # ── LoRA ───────────────────────────────────────────────────────────────────
    print("Aplicando LoRA (misma config que v1)...")
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)

    # ── Dataset ────────────────────────────────────────────────────────────────
    print(f"\nCargando {max_train_samples} ejemplos...")
    raw = load_jsonl(SPLITS_DIR / "train.jsonl")[:max_train_samples]
    ds = Dataset.from_list([format_as_text(r) for r in raw])

    # Warmup proporcional: ~5% de los steps totales
    # steps_per_epoch = max_train_samples / (batch=2 × grad_accum=4)
    steps_per_epoch = max_train_samples // 8
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(1, int(total_steps * 0.05))

    # ── Entrenamiento ──────────────────────────────────────────────────────────
    # max_steps explícito: MLXTrainer ignora num_train_epochs cuando max_steps=-1
    # y puede computar 0 steps en ciertos casos (bug con epochs>1).
    config = MLXTrainingConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,     # batch efectivo = 8 (igual que v1)
        num_train_epochs=num_epochs,
        max_steps=total_steps,             # explícito: 6000/8 × 2 = 1500 steps
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        logging_steps=5,                   # << MEJORA: granular (v1=25)
        report_to="none",
        seed=42,
    )

    trainer = MLXTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=config,
    )

    print(f"\nIniciando entrenamiento ({total_steps} steps, warmup {warmup_steps})...")
    mx.reset_peak_memory()
    t_start = time.perf_counter()
    train_result = trainer.train()
    t_end = time.perf_counter()
    elapsed = t_end - t_start

    mlx_peak = get_mlx_peak_mb()
    ram_peak  = get_ram_mb()

    # ── Guardar modelo ─────────────────────────────────────────────────────────
    print("\nGuardando modelo Unsloth v2...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # ── Log history del trainer ────────────────────────────────────────────────
    log_history = []
    if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
        log_history = trainer.state.log_history

    if hasattr(train_result, 'training_loss'):
        final_loss = train_result.training_loss
    elif isinstance(train_result, dict):
        final_loss = train_result.get('train_loss', train_result.get('loss'))
    else:
        final_loss = None

    log = {
        "experiment":         "unsloth_lora_v2",
        "model":              "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "framework":          "Unsloth (MLX backend, Apple Silicon)",
        "mlx_version":        str(mx.__version__),
        "max_train_samples":  max_train_samples,
        "lora":               LORA_CONFIG,
        "training": {
            "epochs":                num_epochs,
            "batch_size_device":     2,
            "gradient_accumulation": 4,
            "effective_batch_size":  8,
            "learning_rate":         2e-4,
            "scheduler":             "cosine",
            "warmup_steps":          warmup_steps,
            "total_steps":           total_steps,
            "max_seq_len":           MAX_SEQ_LEN,
            "logging_steps":         5,
        },
        "results": {
            "training_loss":     round(final_loss, 4) if final_loss else None,
            "training_time_s":   round(elapsed, 1),
            "training_time_min": round(elapsed / 60, 2),
        },
        "memory": {
            "mlx_after_load_mb":  round(mlx_after_load, 1),
            "mlx_peak_train_mb":  round(mlx_peak, 1),
            "ram_start_mb":       round(ram_start, 1),
            "ram_peak_mb":        round(ram_peak, 1),
            "ram_delta_mb":       round(ram_peak - ram_start, 1),
        },
        "log_history": log_history,
        "v1_comparison": {
            "v1_max_train_samples": 2000,
            "v1_epochs":            1,
            "v1_training_loss":     1.7091,
            "v1_training_time_min": 11.20,
        },
    }

    (OUTPUT_DIR / "training_log.json").write_text(json.dumps(log, indent=2))

    print("\n── Resumen v2 vs v1 ──────────────────────────────────────")
    print(f"  Loss final v2   : {final_loss}  (v1: 1.7091)")
    print(f"  Tiempo          : {elapsed/60:.2f} min       (v1: 11.20 min)")
    print(f"  MLX peak        : {mlx_peak:.0f} MB          (v1: 3511 MB)")
    print(f"\nGuardado en: {OUTPUT_DIR}")
    print("PASO 4-v2 completado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_samples", type=int, default=6000)
    parser.add_argument("--epochs",            type=int, default=2)
    args = parser.parse_args()
    main(max_train_samples=args.max_train_samples, num_epochs=args.epochs)
