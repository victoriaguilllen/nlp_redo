"""
EXPERIMENTO V2 — Fine-tuning con Hugging Face Transformers + PEFT (LoRA)

Intento original (FALLIDO — exploding gradients):
  - max_train_samples = 6000, num_epochs = 2  → 750 steps
  - dtype = torch.float16 sin AMP/gradient scaler
  - Resultado: loss → 0.0 exacto en step 40, NaN en las 308 capas LoRA
  - Causa: overflow fp16 en batch con secuencias largas → gradientes NaN
    propagados a todos los adaptadores float32 (ver analysis2.md)

Configuración actual (fix aplicado):
  1. dtype = torch.float32  → elimina el overflow fp16
  2. max_grad_norm = 0.3    → gradient clipping como capa extra de seguridad
  3. max_train_samples = 3000 (vs 6000 original)  → ~45 min en lugar de 2.8 h
  4. num_train_epochs = 1   (vs 2 original)
  5. logging_steps = 5      → curva de loss granular
  6. Output en results/transformers_lora_v2/  → no sobreescribe v1

Tiempo estimado: ~45 min en Apple M4 Pro MPS.

Uso:
    python3.11 src/train/train_transformers_lora_v2.py
    python3.11 src/train/train_transformers_lora_v2.py --max_train_samples 6000 --epochs 2  # versión original
"""

import argparse
import json
import time
from pathlib import Path

import psutil
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

BASE_MODEL  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ROOT        = Path(__file__).resolve().parents[2]
SPLITS_DIR  = ROOT / "data" / "splits"
OUTPUT_DIR  = ROOT / "results" / "transformers_lora_v2"
MAX_SEQ_LEN = 512

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)

# ── LoRA (mismos hiperparámetros que v1 para que la comparación sea justa) ────
LORA_CONFIG = dict(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)


def get_ram_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 ** 2


def get_mps_mb() -> float:
    return torch.mps.current_allocated_memory() / 1024 ** 2 if DEVICE == "mps" else 0.0


def load_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def tokenize_example(ex: dict, tokenizer) -> dict:
    full_text = ex["prompt"] + ex["response"] + tokenizer.eos_token
    enc = tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LEN, padding=False)
    prompt_enc = tokenizer(ex["prompt"], truncation=True, max_length=MAX_SEQ_LEN)
    prompt_len = len(prompt_enc["input_ids"])
    enc["labels"] = [-100] * prompt_len + enc["input_ids"][prompt_len:]
    return enc


def main(max_train_samples: int = 6000, num_epochs: int = 2):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Transformers LoRA v2 — experimento mejorado")
    print(f"{'='*60}")
    print(f"  Dispositivo        : {DEVICE}")
    print(f"  Muestras           : {max_train_samples}  (v1 tenía 2000)")
    print(f"  Épocas             : {num_epochs}         (v1 tenía 1)")
    print(f"  Output             : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    ram_start = get_ram_mb()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Modelo base en float32 (fix NaN/exploding grads) ─────────────────────
    # NOTA: v1 usaba float16 sin AMP (fp16=False en TrainingArguments). En v2
    # con 750 steps, un batch con secuencias largas causó overflow en activaciones
    # fp16 → gradientes NaN → todos los pesos LoRA se corrompieron (exploding
    # gradients). Fix: cargar en float32 elimina el riesgo de overflow fp16.
    # Coste: +~2 GB MPS; el tiempo de entrenamiento es similar en MPS.
    print("Cargando modelo base en float32 (estable en MPS sin AMP)...")
    mps_before = get_mps_mb()
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32).to(DEVICE)
    mps_after_load = get_mps_mb()
    print(f"  MPS tras carga: {mps_after_load:.0f} MB")

    # ── LoRA ──────────────────────────────────────────────────────────────────
    print("Aplicando LoRA (misma config que v1)...")
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    model.print_trainable_parameters()
    # No es necesario castear adaptadores: base ya es float32, PEFT los crea en float32

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"\nCargando {max_train_samples} ejemplos...")
    raw = load_jsonl(SPLITS_DIR / "train.jsonl")[:max_train_samples]
    ds = Dataset.from_list(raw).map(
        lambda ex: tokenize_example(ex, tokenizer),
        remove_columns=Dataset.from_list(raw[:1]).column_names,
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, pad_to_multiple_of=8)

    # ── Training ──────────────────────────────────────────────────────────────
    # warmup_steps fijo para que sea comparable al warmup_ratio=0.05 de v1:
    # v1: 125 steps × 0.05 = 6 steps → aquí: 6000/16 × 2 ép × 0.05 = ~37 steps
    total_steps = (max_train_samples // 16) * num_epochs
    warmup_steps = max(1, int(total_steps * 0.05))

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,      # batch efectivo = 16 (igual que v1)
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        optim="adamw_torch",
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,                  # gradient clipping: previene NaN en MPS con fp16 base
        logging_steps=5,                    # << MEJORA: logs cada 5 steps (v1=25)
        save_strategy="epoch",
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print(f"\nIniciando entrenamiento ({total_steps} steps, warmup {warmup_steps} steps)...")
    t0 = time.perf_counter()
    result = trainer.train()
    t1 = time.perf_counter()
    elapsed = t1 - t0

    mps_peak = get_mps_mb()
    ram_peak = get_ram_mb()

    # ── Guardar adaptador ─────────────────────────────────────────────────────
    print("\nGuardando adaptador LoRA v2...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # ── Extraer log_history completo del trainer ──────────────────────────────
    log_history = trainer.state.log_history  # lista de dicts con step y loss

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())

    log = {
        "experiment":         "transformers_lora_v2",
        "model":              BASE_MODEL,
        "method":             "LoRA (float16 base + float32 adapters)",
        "device":             DEVICE,
        "max_train_samples":  max_train_samples,
        "lora": {
            "rank":           LORA_CONFIG["r"],
            "alpha":          LORA_CONFIG["lora_alpha"],
            "dropout":        LORA_CONFIG["lora_dropout"],
            "target_modules": LORA_CONFIG["target_modules"],
        },
        "training": {
            "epochs":                num_epochs,
            "batch_size_device":     4,
            "gradient_accumulation": 4,
            "effective_batch_size":  16,
            "learning_rate":         2e-4,
            "scheduler":             "cosine",
            "warmup_steps":          warmup_steps,
            "optimizer":             "adamw_torch",
            "max_seq_len":           MAX_SEQ_LEN,
            "logging_steps":         5,
        },
        "results": {
            "global_steps":        result.global_step,
            "training_loss":       round(result.training_loss, 4),
            "training_time_s":     round(elapsed, 1),
            "training_time_min":   round(elapsed / 60, 2),
            "trainable_params":    trainable,
            "total_params":        total,
            "trainable_pct":       round(trainable / total * 100, 2),
        },
        "memory": {
            "mps_model_load_mb":   round(mps_after_load - mps_before, 1),
            "mps_peak_mb":         round(mps_peak, 1),
            "ram_start_mb":        round(ram_start, 1),
            "ram_peak_mb":         round(ram_peak, 1),
            "ram_delta_mb":        round(ram_peak - ram_start, 1),
        },
        "log_history":  log_history,     # << curva de loss completa (logging_steps=5)
        "v1_comparison": {
            "v1_max_train_samples": 2000,
            "v1_epochs":            1,
            "v1_training_loss":     1.6895,
            "v1_training_time_min": 28.62,
        },
    }

    log_path = OUTPUT_DIR / "training_log.json"
    log_path.write_text(json.dumps(log, indent=2))

    print("\n── Resumen v2 vs v1 ──────────────────────────────────────")
    print(f"  Loss final v2   : {result.training_loss:.4f}  (v1: 1.6895)")
    print(f"  Steps           : {result.global_step}       (v1: 125)")
    print(f"  Tiempo          : {elapsed/60:.2f} min      (v1: 28.62 min)")
    print(f"  MPS peak        : {mps_peak:.0f} MB         (v1: 2364 MB)")
    print(f"\nGuardado en: {OUTPUT_DIR}")
    print("PASO 3-v2 completado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_samples", type=int, default=3000)
    parser.add_argument("--epochs",            type=int, default=1)
    args = parser.parse_args()
    main(max_train_samples=args.max_train_samples, num_epochs=args.epochs)
