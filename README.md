# Fine-tune Once, Serve Anywhere: Transformers, Unsloth, vLLM & Ollama

Proyecto final de NLP II — Curso 2025/2026  
Universidad Pontificia Comillas (ICAI)

## Descripción

Proyecto de fine-tuning e inferencia de LLMs usando el dataset **Databricks Dolly 15k** y el modelo base **TinyLlama-1.1B-Chat**. Se comparan cuatro frameworks:

- **Hugging Face Transformers + PEFT** (LoRA, float32 base — QLoRA requiere CUDA)
- **Unsloth** (MLX backend, Apple Silicon) — modelo principal
- **vLLM** (excluido — requiere CUDA; sin backend MPS/Metal)
- **Ollama** (Metal backend, llama.cpp Q4)

**Modelos finales seleccionados:** Transformers LoRA v1 y Unsloth LoRA v2.

---

## Estructura del proyecto

```
ProyectoFinal/
├── src/
│   ├── data/
│   │   └── prepare_data.py               # PASO 1 — Descarga y split del dataset
│   ├── train/
│   │   ├── train_transformers_lora.py    # Fine-tuning Transformers LoRA v1 (2k, 1 epoch)
│   │   ├── train_transformers_lora_v2.py # Fine-tuning Transformers LoRA v2 (fallido — NaN)
│   │   ├── train_unsloth.py              # Fine-tuning Unsloth MLX v1 (2k, 1 epoch)
│   │   └── train_unsloth_v2.py           # Fine-tuning Unsloth MLX v2 (6k, 2 epochs) ← principal
│   ├── eval/
│   │   ├── evaluate_baseline.py          # Evaluación zero-shot (modelo original)
│   │   ├── evaluate_finetuned.py         # Evaluación Transformers LoRA v1
│   │   ├── evaluate_unsloth.py           # Evaluación Unsloth v1
│   │   ├── evaluate_unsloth_v2.py        # Evaluación Unsloth v2
│   │   ├── infer_unsloth_v2.py           # Demo de inferencia con 7 prompts (Unsloth v2)
│   │   ├── compare_models.py             # Comparación automática entre modelos
│   │   └── fill_human_scores.py          # Rellena scores de evaluación humana
│   ├── benchmark/
│   │   ├── benchmark_transformers.py     # Benchmark Transformers (MPS)
│   │   ├── benchmark_unsloth.py          # Benchmark Unsloth MLX v1 (base)
│   │   ├── benchmark_unsloth_v2.py       # Benchmark Unsloth MLX v2 (con LoRA)
│   │   ├── benchmark_vllm.py             # Benchmark vLLM (CPU en macOS — excluido)
│   │   ├── benchmark_ollama.py           # Benchmark Ollama (Metal)
│   │   ├── collect_responses.py          # Captura respuestas de texto por framework
│   │   └── run_all_benchmarks.py         # Ejecuta benchmarks y genera tabla resumen
│   └── generate_figures_v2.py            # Genera las 11 figuras comparativas
│
├── data/
│   └── splits/
│       ├── train.jsonl                   # 12.000 ejemplos de entrenamiento
│       ├── dev.jsonl                     #  2.000 ejemplos de validación
│       └── test.jsonl                    #  1.000 ejemplos de prueba
│
├── results/
│   ├── original_model/                   # Baseline sin fine-tuning
│   │   ├── metrics.json
│   │   ├── predictions.jsonl
│   │   └── human_eval_prompts.json
│   ├── transformers_lora/                # Transformers LoRA v1 ← modelo final
│   │   ├── adapter_model.safetensors
│   │   ├── training_log.json
│   │   └── eval/
│   ├── transformers_lora_v2/             # Transformers LoRA v2 (fallido — NaN en step 40)
│   │   └── training_log.json             # Solo log; pesos NaN descartados
│   ├── unsloth_lora/                     # Unsloth MLX v1
│   │   ├── adapters.safetensors
│   │   ├── training_log.json
│   │   └── eval/
│   ├── unsloth_lora_v2/                  # Unsloth MLX v2 ← modelo principal
│   │   ├── adapters.safetensors
│   │   ├── training_log.json
│   │   └── eval/
│   │       ├── metrics.json
│   │       ├── human_eval_prompts.json
│   │       └── inference_demo.json
│   ├── comparison/
│   │   ├── human_eval_comparison.json
│   │   └── summary_table.txt
│   └── benchmarks/
│       ├── benchmark_transformers.json
│       ├── benchmark_unsloth.json
│       ├── benchmark_unsloth_v2.json
│       ├── benchmark_ollama.json
│       └── responses_by_framework.json
│
├── figures/                              # 11 figuras generadas por generate_figures_v2.py
│   ├── fig1_metrics.png                  # Mejora relativa por modelo vs baseline
│   ├── fig2_human_eval.png               # Evaluación humana agrupada (4 modelos, 3 criterios)
│   ├── fig3_throughput.png               # Throughput con barras de error ±1σ
│   ├── fig4_latency_boxplot.png          # Distribución de latencia (boxplots)
│   ├── fig5_human_heatmap.png            # Heatmap de scores por prompt y modelo
│   ├── fig6_radar.png                    # Radar chart (7 dimensiones, 4 modelos)
│   ├── fig7_training_comparison.png      # Comparativa de entrenamiento (2×2 grid)
│   ├── fig8_quality_vs_speed.png         # Calidad vs velocidad (scatter)
│   ├── fig9_improvement_journey.png      # Evolución baseline → Unsloth v2
│   ├── fig10_v1_vs_v2_prompts.png        # Unsloth v1 vs v2 por prompt con deltas
│   └── fig11_std_inference.png           # Variabilidad de inferencia con CV por framework
│
├── analysis.md                           # Análisis v1 (Transformers y Unsloth v1)
├── analysis2.md                          # Análisis v2 (Unsloth v2, benchmarks, conclusiones)
├── promps.md                             # Prompts y respuestas completas (eval humana + inferencia)
├── README.md
└── doc.pdf                               # Enunciado del proyecto
```

---

## Hardware y entorno

| Parámetro | Valor |
|-----------|-------|
| Sistema operativo | macOS (Darwin 24.x, Apple Silicon M4 Pro) |
| Python | 3.11.9 |
| PyTorch | 2.9.0 (backend MPS) |
| MLX | 0.26.x (backend Unsloth) |
| RAM unificada | 16 GB |
| Modelo base | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |

> **Nota:** vLLM excluido (requiere CUDA; sin backend MPS/Metal). Se usan MPS y MLX como aceleradores en macOS.

---

## Instalación

```bash
# Transformers + PEFT
pip install datasets transformers accelerate peft sentencepiece \
            sacrebleu rouge-score bert-score torch psutil

# Unsloth / MLX
pip install unsloth mlx mlx-lm

# Ollama — instalar desde https://ollama.com
ollama pull tinyllama
```

---

## Uso

### PASO 1 — Preparar datos

```bash
python src/data/prepare_data.py
# → data/splits/ (train 12k / dev 2k / test 1k)
```

### PASO 2 — Evaluar modelo base

```bash
python src/eval/evaluate_baseline.py
```

### PASO 3 — Fine-tuning

```bash
# Unsloth v2 — modelo principal (Apple Silicon)
python3.11 src/train/train_unsloth_v2.py
# → results/unsloth_lora_v2/  (~62 min en M4 Pro)

# Transformers v1
python3.11 src/train/train_transformers_lora.py
# → results/transformers_lora/  (~29 min en M4 Pro)
```

### PASO 4 — Evaluación

```bash
python3.11 src/eval/evaluate_unsloth_v2.py
python3.11 src/eval/infer_unsloth_v2.py
python3.11 src/eval/evaluate_finetuned.py
python3.11 src/eval/fill_human_scores.py   # rellena scores humanos
```

### PASO 5 — Benchmark de inferencia

```bash
ollama serve                                    # en otra terminal
python src/benchmark/benchmark_ollama.py
python3.11 src/benchmark/benchmark_transformers.py
python3.11 src/benchmark/benchmark_unsloth_v2.py
```

### PASO 6 — Figuras

```bash
python3.11 src/generate_figures_v2.py
# → figures/fig1_*.png … fig11_*.png
```

---

## Resultados

### Evaluación automática — 100 ejemplos del test set

| Métrica | Original | TF LoRA v1 | Unsloth v1 | Unsloth v2 |
|---------|:--------:|:----------:|:----------:|:----------:|
| **BLEU** | 0.0625 | 0.0871 (+39%) | 0.0793 (+27%) | **0.1001 (+60%)** |
| **ROUGE-L** | 0.2174 | 0.2536 (+17%) | 0.2531 (+16%) | **0.2703 (+24%)** |
| **BERTScore F1** | 0.8657 | 0.8699 (+0.5%) | 0.8697 (+0.5%) | **0.8745 (+1.0%)** |

### Evaluación humana — 5 prompts (escala 1–5)

| Modelo | Helpfulness | Factuality | Instr.-follow | **Media** |
|--------|:-----------:|:----------:|:-------------:|:---------:|
| TinyLlama original | 2.80 | 3.40 | 2.20 | 2.80 |
| Transformers LoRA v1 | 2.60 | 3.20 | **3.60** | 3.13 |
| Unsloth v1 | 2.40 | 2.60 | 3.00 | 2.67 |
| **Unsloth v2** | **3.00** | **3.80** | 3.40 | **3.40** |

### Benchmark de inferencia — modelo base TinyLlama

| Framework | Latencia (s) | Throughput (tok/s) | Memoria (MB) |
|-----------|:------------:|:------------------:|:------------:|
| **Ollama** (Metal Q4) | **0.88 ± 0.20** | **268.4 ± 1.0** | interno |
| Unsloth v1 (MLX base) | 1.94 ± 0.75 | 106.7 ± 3.5 | 2.281 |
| Transformers (MPS) | 2.53 ± 1.53 | 66.6 ± 2.7 | 3.534 |
| Unsloth v2 (MLX + LoRA) | 1.29 ± 1.21 | 54.9 | 2.354 |

> Unsloth v2 carga el adaptador LoRA completo; el resto usan el modelo base. Las comparaciones de velocidad son orientativas.

### Entrenamiento

| | Transformers LoRA v1 | Unsloth v1 | Unsloth v2 |
|-|:-------------------:|:----------:|:----------:|
| Muestras | 2.000 | 2.000 | 6.000 |
| Épocas / steps | 1 / 125 | 1 / 250 | 2 / 1.500 |
| Loss final | 1.6895 | 1.7091 | **1.6021** |
| Tiempo | 28.62 min | **11.20 min** | 62.12 min |
| Memoria pico (MB) | **2.364** | 3.512 | 3.416 |

---

## Incidencias técnicas

| Incidencia | Causa | Fix |
|-----------|-------|-----|
| NaN en TF v2 (step 40) | `float16` sin AMP → overflow logits con secuencias largas | `dtype=float32` + `max_grad_norm=0.3` |
| 0 steps en Unsloth v2 | `max_steps=-1` con `epochs=2` → schedule vacío en MLXTrainer | `max_steps=1500` explícito |
| SIGABRT en Unsloth v2 (step ~510) | Ejemplo de 6.887 tokens activa assert Metal con `max_seq_len=512` | `max_seq_len=384` |
| OOM / SIGKILL en Unsloth v2 | Ejecución simultánea con Transformers agota 16 GB compartidos | Ejecución secuencial |

Ver `analysis2.md` para diagnóstico completo y curvas de loss.  
Ver `promps.md` para respuestas completas por modelo y por framework.
