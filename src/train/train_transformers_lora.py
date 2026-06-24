"""
PASO 3 — Fine-tuning con Hugging Face Transformers + PEFT (LoRA)

Técnica elegida: LoRA (Low-Rank Adaptation) en float16 sobre MPS.
Se usa LoRA en lugar de QLoRA (4-bit) porque, aunque bitsandbytes
carga modelos cuantizados en MPS, los kernels de entrenamiento 4-bit
requieren CUDA para el backward pass. LoRA en float16 es la alternativa
estable y recomendada para Apple Silicon.

Hiperparámetros basados en:
  - Hu et al. 2021 (LoRA paper): rank 16, alpha 32 como punto de partida
  - TinyLlama finetuning guides: lr 2e-4, cosine scheduler, warmup 5%
  - Hardware MPS: batch 4 × grad_accum 4 = 16 efectivo (limitado por RAM)

Uso:
    python src/train_transformers_lora.py
    python src/train_transformers_lora.py --max_train_samples 500   # prueba rápida
    python src/train_transformers_lora.py --max_train_samples 12000  # dataset completo
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

BASE_MODEL   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SPLITS_DIR   = Path(__file__).resolve().parents[2] / "data" / "splits"
OUTPUT_DIR   = Path(__file__).resolve().parents[2] / "results" / "transformers_lora"
MAX_SEQ_LEN  = 512
DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)

# ── Hiperparámetros ───────────────────────────────────────────────────────────
LORA_CONFIG = dict(
    r=16,           # rank: compromiso capacidad/memoria (Hu et al. 2021)
    lora_alpha=32,  # alpha = 2*r → escala efectiva de 1.0
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # atención
        "gate_proj", "up_proj", "down_proj",        # MLP
    ],
)

TRAIN_CONFIG = dict(
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # batch efectivo = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    optim="adamw_torch",
    fp16=False,                      # MPS no usa fp16 de CUDA; el dtype lo fija el modelo
    bf16=False,
    logging_steps=25,
    save_strategy="epoch",
    report_to="none",
    dataloader_num_workers=0,        # MPS: sin workers para evitar fork issues
)


def get_ram_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 ** 2


def get_mps_mb() -> float:
    if DEVICE == "mps":
        return torch.mps.current_allocated_memory() / 1024 ** 2
    return 0.0


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def tokenize_example(ex: dict, tokenizer) -> dict:
    full_text = ex["prompt"] + ex["response"] + tokenizer.eos_token
    enc = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    )
    # Solo calcular loss sobre la respuesta (enmascarar el prompt)
    prompt_enc = tokenizer(ex["prompt"], truncation=True, max_length=MAX_SEQ_LEN)
    prompt_len = len(prompt_enc["input_ids"])
    labels = [-100] * prompt_len + enc["input_ids"][prompt_len:]
    enc["labels"] = labels
    return enc


def main(max_train_samples: int = 2000):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dispositivo: {DEVICE}")
    print(f"Muestras de entrenamiento: {max_train_samples}")

    ram_start = get_ram_mb()

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    print("\nCargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 2. Modelo base en float16 ─────────────────────────────────────────────
    print("Cargando modelo base en float16...")
    mps_before_model = get_mps_mb()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
    ).to(DEVICE)
    mps_after_model = get_mps_mb()
    print(f"  Memoria MPS tras cargar modelo: {mps_after_model:.0f} MB")

    # ── 3. LoRA ───────────────────────────────────────────────────────────────
    print("Aplicando LoRA...")
    lora_cfg = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    # Asegurar que los parámetros LoRA están en float32 para estabilidad en MPS
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    # ── 4. Dataset ────────────────────────────────────────────────────────────
    print(f"\nCargando {max_train_samples} ejemplos de train...")
    raw = load_jsonl(SPLITS_DIR / "train.jsonl")[:max_train_samples]
    ds = Dataset.from_list(raw)
    ds = ds.map(
        lambda ex: tokenize_example(ex, tokenizer),
        remove_columns=ds.column_names,
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, padding=True, pad_to_multiple_of=8
    )

    # ── 5. Entrenamiento ──────────────────────────────────────────────────────
    print("\nIniciando entrenamiento...")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        **TRAIN_CONFIG,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    t_start = time.perf_counter()
    train_result = trainer.train()
    t_end = time.perf_counter()
    training_time_s = t_end - t_start

    mps_peak = get_mps_mb()
    ram_peak = get_ram_mb()

    # ── 6. Guardar adaptador ──────────────────────────────────────────────────
    print("\nGuardando adaptador LoRA...")
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # ── 7. Registro de experimento ────────────────────────────────────────────
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    steps     = train_result.global_step
    loss      = train_result.training_loss

    log = {
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
            "epochs":                TRAIN_CONFIG["num_train_epochs"],
            "batch_size_device":     TRAIN_CONFIG["per_device_train_batch_size"],
            "gradient_accumulation": TRAIN_CONFIG["gradient_accumulation_steps"],
            "effective_batch_size":  TRAIN_CONFIG["per_device_train_batch_size"] * TRAIN_CONFIG["gradient_accumulation_steps"],
            "learning_rate":         TRAIN_CONFIG["learning_rate"],
            "scheduler":             TRAIN_CONFIG["lr_scheduler_type"],
            "warmup_ratio":          TRAIN_CONFIG["warmup_ratio"],
            "optimizer":             TRAIN_CONFIG["optim"],
            "max_seq_len":           MAX_SEQ_LEN,
        },
        "results": {
            "global_steps":    steps,
            "training_loss":   round(loss, 4),
            "training_time_s": round(training_time_s, 1),
            "training_time_min": round(training_time_s / 60, 2),
            "trainable_params":  trainable,
            "total_params":      total,
            "trainable_pct":     round(trainable / total * 100, 2),
        },
        "memory": {
            "mps_model_load_mb":   round(mps_after_model - mps_before_model, 1),
            "mps_peak_mb":         round(mps_peak, 1),
            "ram_start_mb":        round(ram_start, 1),
            "ram_peak_mb":         round(ram_peak, 1),
            "ram_delta_mb":        round(ram_peak - ram_start, 1),
        },
    }

    log_path = OUTPUT_DIR / "training_log.json"
    with log_path.open("w") as f:
        json.dump(log, f, indent=2)

    print("\n── Resumen del entrenamiento ──────────────────────────────")
    print(f"  Parámetros entrenables : {trainable:,} ({log['results']['trainable_pct']}% del total)")
    print(f"  Steps                  : {steps}")
    print(f"  Loss final             : {loss:.4f}")
    print(f"  Tiempo total           : {training_time_s/60:.2f} min")
    print(f"  Memoria MPS peak       : {mps_peak:.0f} MB")
    print(f"  RAM delta              : {ram_peak - ram_start:.0f} MB")
    print(f"\nModelo guardado en '{OUTPUT_DIR}'")
    print(f"Log guardado en '{log_path}'")
    print("\nPASO 3 completado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_train_samples", type=int, default=2000,
        help="Nº de ejemplos de train a usar (default: 2000 para MPS)"
    )
    args = parser.parse_args()
    main(max_train_samples=args.max_train_samples)
