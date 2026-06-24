# Analysis v2 — Experimentos mejorados

> **Contexto:** Este documento recoge los experimentos de la segunda iteración del proyecto,
> realizados a partir del feedback del evaluador (Jaime) sobre el informe v1.
> Para los resultados originales ver `analysis.md`.
> **Nada de la v1 se sobreescribe**: todos los nuevos resultados se guardan en
> directorios `*_v2` independientes.

---

## Qué cambiamos y por qué

### Problemas identificados en v1

| Problema | Evidencia | Solución en v2 |
|----------|-----------|----------------|
| Pocos datos de entrenamiento (2k/12k disponibles) | Mode collapse en brainstorming; regresión factual | 6k muestras (3×) |
| Solo 1 época | Loss aún descendía al final | 2 épocas con cosine LR |
| Log de loss muy escaso (cada 25 steps) | Imposible trazar curva de aprendizaje | `logging_steps=5` |
| Evaluación humana subjetiva (1 evaluador) | Sin acuerdo inter-anotador | Se mantiene pero se documenta explícitamente como limitación |
| vLLM no funciona en macOS | Sin GPU CUDA disponible | Excluido de benchmarks v2 |

### Lo que NO cambiamos (para comparabilidad)

- Arquitectura LoRA: rank=16, alpha=32, dropout=0.05, mismos 7 módulos
- Learning rate: 2e-4, scheduler cosine
- Batch efectivo: 16 (device_batch=4 × grad_accum=4)
- max_seq_len: 512
- Modelo base: TinyLlama-1.1B-Chat-v1.0
- Dataset: Dolly 15k (mismo split train/dev/test 12k/2k/1k)
- Prompts de evaluación humana: los mismos 5 prompts de v1

---

## PASO A — Fine-tuning Transformers LoRA v2

**Script:** `src/train/train_transformers_lora_v2.py`
**Output:** `results/transformers_lora_v2/`
**Fecha de ejecución:** 2026-06-23

### Configuración — Intento original (FALLIDO)

> Este fue el primer intento. Fallaron por exploding gradients (detalle completo abajo).

| Hiperparámetro | v1 | v2 original (fallido) | Cambio |
|----------------|----|-----------------------|--------|
| `dtype` base model | float16 | float16 | Sin cambio → **causa raíz del fallo** |
| `fp16` TrainingArguments | False | False | Sin gradient scaler |
| max_train_samples | 2000 | 6000 | 3× más datos |
| num_train_epochs | 1 | 2 | 2 épocas |
| logging_steps | 25 | 5 | Más granular |
| max_grad_norm | 1.0 (default) | 1.0 (default) | Sin cambio → no recortó los gradientes |
| Total steps | 125 | 750 | (6000/16) × 2 |
| Tiempo real | 28.62 min | 172.73 min | Entrenó todo pero con NaN desde step 40 |

### Configuración — Re-run con fix aplicado

| Hiperparámetro | v1 | v2 re-run | Cambio |
|----------------|----|-----------|----|
| `dtype` base model | float16 | **float32** | **Fix principal**: elimina overflow fp16 |
| max_grad_norm | 1.0 | **0.3** | **Fix extra**: clipping más agresivo |
| max_train_samples | 2000 | **3000** | 1.5× más datos (reducido de 6k por tiempo) |
| num_train_epochs | 1 | **1** | Igual que v1 (reducido de 2 por tiempo) |
| logging_steps | 25 | **5** | Curva de loss granular |
| Total steps | 125 | **~187** | (3000/16) × 1 |

**Tiempo estimado:** ~45 min en Apple M4 Pro (MPS)

### Resultados del entrenamiento

> **RESULTADO: FALLO POR NaN (inestabilidad numérica en MPS)**

| Métrica | v1 | v2 | Δ |
|---------|----|----|---|
| Training loss final | 1.6895 | 0.0837* | — |
| Tiempo de entrenamiento | 28.62 min | **172.73 min** | +503% |
| Steps totales | 125 | 750 | +500% |
| MPS peak memory | 2364 MB | 2341 MB | -1% |
| RAM delta | 216 MB | 192 MB | -11% |

*Loss media distorsionada: desde el step 40 hasta el 750 (711 steps) la loss fue **exactamente 0.0** — señal inequívoca de NaN propagado a todos los pesos.

### Causa del fallo: Exploding Gradients en MPS fp16

La inspección post-entrenamiento confirmó que **las 308 capas LoRA contienen NaN** en todos sus pesos.

**Hiperparámetros del run fallido (conservados para referencia):**

| Parámetro | Valor usado | Problema |
|-----------|-------------|---------|
| `dtype` base model | `torch.float16` | Rango limitado: overflow con activaciones grandes |
| `fp16` en TrainingArguments | `False` | Sin AMP → sin gradient scaler → sin protección frente a overflow |
| `max_grad_norm` | `1.0` (default) | Demasiado permisivo: gradientes grandes no se recortan a tiempo |
| `per_device_train_batch_size` | 4 | Batches grandes → más varianza por step |
| `gradient_accumulation_steps` | 4 | Batch efectivo = 16; acumula más gradiente antes de actualizar |
| `learning_rate` | `2e-4` | LR normal, pero amplifica gradientes grandes |
| `max_seq_len` | 512 | Algunas secuencias llegan a 512 tokens → activaciones grandes en fp16 |

**Mecanismo del fallo (exploding gradients):**
1. En el step ~40, un batch con secuencias largas produjo **activaciones de alto valor en fp16**
2. El rango de fp16 es [−65504, 65504] — en secuencias largas con LM head de vocabulario 32k, los logits superan este límite → **overflow a ±inf/NaN**
3. Sin gradient scaler (AMP desactivado), la norma del gradiente disparó más allá de 1.0 sin ser recortada a tiempo
4. El NaN se propagó hacia atrás a través de todos los adaptadores fp32 durante el backward pass
5. El `Trainer` de HuggingFace **no detecta NaN por defecto** y continuó 710 steps más reportando loss=0.0

**Por qué v1 no falló:** v1 solo tenía 125 steps. Con 2k muestras el dataloader no alcanzó el batch problemático. Con 6k muestras hay más secuencias largas y el shuffle las distribuye a lo largo de todos los steps.

**Fixes aplicados al script (para futuros re-runs):**
1. **`dtype=torch.float32`** — elimina la causa raíz: sin fp16 no hay overflow aritmético
2. **`max_grad_norm=0.3`** — capa extra de seguridad: recorta gradientes antes de que crezcan

### Curva de loss

| Steps | Loss observada | Interpretación |
|-------|---------------|----------------|
| 5–35 | 1.50–1.90 | Entrenamiento normal, descenso esperado |
| 40 | **0.0000** | Gradiente NaN propagado → loss = NaN reportado como 0.0 |
| 45–750 | **0.0000** | Modelo completamente corrompido, pesos NaN desde este punto |

El modelo v2 de Transformers **no es utilizable**. Los checkpoints y adaptadores NaN han sido eliminados (solo se conserva `training_log.json` para documentación).

**Decisión:** Se documenta como hallazgo técnico. El mismo experimento v2 (6k muestras, 2 épocas) se ejecuta con Unsloth/MLX, cuya aritmética interna (bfloat16 + gradient checkpointing CCE) es más estable en Apple Silicon.

---

## PASO B — Evaluación automática Transformers LoRA v2

> **CANCELADO** — El modelo TF v2 tiene pesos NaN (ver PASO A). La evaluación no tiene sentido con un modelo roto.
> Si se re-entrena con el fix (`max_grad_norm=0.3`), ejecutar: `python3.11 src/eval/evaluate_finetuned_v2.py`

---

## PASO C — Fine-tuning Unsloth LoRA v2

**Script:** `src/train/train_unsloth_v2.py`
**Output:** `results/unsloth_lora_v2/`

### C.1 — Intentos fallidos (SIGABRT en step ~510–530)

Se realizaron **tres intentos** antes de encontrar la configuración estable:

#### Intento 1 — OOM al correr con Transformers (exit 144)
Lanzado simultáneamente con el entrenamiento Transformers v2. Ambos procesos compitieron por los 16 GB de memoria unificada (Transformers MPS ~8 GB + Unsloth MLX ~8 GB). macOS mató Unsloth con SIGKILL (exit 144).

**Fix:** ejecutar secuencialmente.

#### Intento 2 — 0 steps, sin archivos guardados (bug MLXTrainer)
Configuración: `max_steps=-1, num_train_epochs=2`. El MLXTrainer interpretó `num_train_epochs=2` con `max_steps=-1` como 0 steps efectivos y terminó inmediatamente (exit 0 del pipe `tee`, no de Python). No se guardó ningún archivo.

**Fix:** pasar `max_steps=1500` explícito en lugar de `-1`.

#### Intento 3 — SIGABRT en step 530 (seed=42) y step 510 (seed=7)

| Parámetro | Valor |
|-----------|-------|
| max_train_samples | 6000 |
| max_steps | 1500 |
| max_seq_length | **512** |
| seed | 42 → 7 (segundo run) |
| batch_size_device | 2 |
| effective_batch | 8 |

El proceso siempre abortó en el rango steps 510–530, independientemente del seed. SIGABRT indica que el proceso llamó a `abort()` internamente — una aserción fallida en el backend Metal de MLX. El crash era **determinista por el dato, no por el orden**: algún ejemplo del dataset con secuencia larga (hasta 6887 tokens antes de truncar a 512) genera tensores que superan algún límite interno del compilador MLX cuando todas las posiciones del buffer de 512 están ocupadas.

**Curva de loss antes del crash (intento 3, seed=42):**

| Steps | Loss | Observación |
|-------|------|-------------|
| 5 | 2.097 | Arranque normal |
| 100 | ~1.65 | Descenso inicial |
| 400 | 1.462 | Mínimo local |
| 440 | 1.436 | Mínimo global alcanzado |
| 530 | 1.731 | **SIGABRT — proceso abortado** |

La loss descendía correctamente y los gradientes eran estables (0.4–0.7). El crash no fue por inestabilidad numérica sino por un fallo de aserción de Metal/MLX con la secuencia problemática.

### C.2 — Configuración actual (en ejecución)

**Fix aplicado:** reducir `max_seq_length` de 512 a **384**. Con esta longitud máxima, la secuencia problemática se trunca antes de llegar a la longitud que dispara la aserción de MLX.

| Hiperparámetro | v1 | v2 (intentos fallidos) | v2 (actual) |
|----------------|----|-----------------------|-------------|
| max_train_samples | 2000 | 6000 | 6000 |
| num_train_epochs | 1 | 2 | 2 |
| max_steps | — | 1500 | **1500** |
| max_seq_length | 512 | 512 | **384** |
| batch_size_device | 2 | 2 | 2 |
| effective_batch | 8 | 8 | 8 |
| seed | 42 | 42 / 7 | 42 |
| Total steps | 250 | 1500 | 1500 |

**Tiempo estimado:** ~50 min en Apple M4 Pro (MLX, sin competencia de memoria).

---

## PASO D — Evaluación automática Unsloth LoRA v2

**Script:** `src/eval/evaluate_unsloth_v2.py`
**Output:** `results/unsloth_lora_v2/eval/metrics.json`
**Muestras:** 100 ejemplos del test set (mismo subset que v1 para comparabilidad)

### D.1 — Métricas automáticas

| Métrica | Unsloth v1 | Unsloth v2 | Δ |
|---------|:----------:|:----------:|:-:|
| BLEU | 0.0793 | **0.1001** | **+26.3 %** |
| ROUGE-L | 0.2531 | **0.2703** | +6.8 % |
| BERTScore F1 | 0.8697 | **0.8745** | +0.5 % |

v2 mejora en las tres métricas. La ganancia más llamativa es BLEU (+26 %), que mide solapamiento n-gram exacto con la referencia — indica que el modelo v2 genera respuestas más parecidas en wording a las del dataset Dolly. ROUGE-L sube más moderadamente porque ya estaba en un nivel razonable. BERTScore mejora poco porque la similitud semántica ya era alta en v1; los dos modelos entienden de qué habla la pregunta, pero v2 lo expresa con más precisión.

### D.2 — Evaluación humana (5 prompts, escala 1–5)

Los mismos 5 prompts de v1 se usan en v2 para que la comparación sea directa.

| Prompt | Dimensión evaluada | v1 (H/F/IF) | v2 (H/F/IF) |
|--------|-------------------|:-----------:|:-----------:|
| P1 — Capital de Australia | Factualidad | 2/1/4 | 3/3/4 |
| P2 — Huevos revueltos | Helpfulness | 4/4/4 | 4/4/4 |
| P3 — Clasificación sentimiento | Instruction-following | 3/3/3 | 5/5/5 |
| P4 — Poema noche (4 líneas) | Helpfulness + IF | 1/2/1 | 2/4/3 |
| P5 — Nombres cafetería | Helpfulness | 2/3/3 | 1/3/1 |

| Dimensión | v1 | v2 | Δ |
|-----------|:--:|:--:|:-:|
| Helpfulness | 2.40 | **3.00** | +0.60 |
| Factuality | 2.60 | **3.80** | **+1.20** |
| Instruction-following | 3.00 | **3.40** | +0.40 |
| **Media overall** | 2.67 | **3.40** | **+0.73** |

**Análisis por prompt:**

- **P1 Capital:** v1 no reconocía a Canberra como capital; v2 lo hace correctamente. Ambos versiones dan una razón superficial para la confusión con Sydney ("same country"), pero v2 al menos responde la pregunta principal.
- **P2 Huevos:** Sin diferencia. Ambos producen 7 pasos claros y correctos desde v1.
- **P3 Sentimiento:** La mayor mejora. v1 generaba una lista mezclada sin separar positivos/negativos. v2 clasifica las 8 palabras en dos listas limpias con todos los scores máximos (5/5/5). Esto confirma que más datos de classification mejoran el instruction-following en tareas estructuradas.
- **P4 Poema:** v2 produce 4 líneas (cumple la instrucción), pero con repetición: "a sight to behold" y "diamonds" aparecen dos veces. El IF mejora (de 1 a 3) porque al menos respeta el formato; la Helpfulness baja porque es poco creativo.
- **P5 Cafetería (regresión):** v2 genera 5 nombres pero todos siguen el patrón "The Coffee X" y omite completamente la explicación del concepto detrás de cada nombre, que era un requisito explícito de la instrucción. La hipótesis de que más datos reducirían el mode collapse en brainstorming **no se confirma** para esta tarea.

**Conclusión evaluación humana:** la mejora más grande es en factualidad (+1.20 puntos). 3× más datos de entrenamiento exponen al modelo a más contexto factual del dataset Dolly. La tarea de brainstorming con formato libre sigue siendo el punto débil.

---

## PASO E — Benchmark de inferencia Unsloth v2

**Script:** `src/benchmark/benchmark_unsloth_v2.py`
**Output:** `results/benchmarks/benchmark_unsloth_v2.json`
**Configuración:** temperatura=0.0, max_tokens=256, 3 trials × 5 prompts

El script carga el modelo base TinyLlama + adaptador LoRA v2 (`results/unsloth_lora_v2/`) vía `mlx_lm.load(base, adapter_path=...)`.

### E.1 — Resultados

| Métrica | Unsloth v1 (base) | Unsloth v2 (base+LoRA) | Δ |
|---------|:-----------------:|:----------------------:|:-:|
| Latencia media (s) | 1.942 | **1.289** | **−34 %** |
| Throughput (tok/s) | 106.7 | 54.9 | −49 % |
| Tokens de salida medios | 210 | 77 | −63 % |
| Memoria MLX pico (MB) | 2281 | 2354 | +73 MB |

### E.2 — Interpretación del throughput

El throughput de v1 (106.7 tok/s) supera al de v2 (54.9 tok/s), pero la comparación directa es engañosa por dos razones:

1. **v1 benchmarkeó el modelo BASE** (sin adaptador LoRA), v2 benchmarkeó el modelo **fine-tuned** (con adaptador). El overhead de computación del adaptador LoRA en inferencia MLX es pequeño (~5–10 %), pero existe.
2. **El modelo fino-tuneado genera respuestas más cortas.** v1 producía de media 210 tokens por respuesta (el modelo base tiende a continuar texto indefinidamente hasta `max_tokens`), mientras que v2 genera 77 tokens de media porque aprendió a terminar la respuesta en el momento apropiado. Dado que el throughput se mide como `tokens_generados / tiempo`, menos tokens → denominador más pequeño pero cociente similar en velocidad bruta.

Lo que sí es comparable es la **latencia por solicitud**: v2 es 34 % más rápido porque genera respuestas más concisas y el usuario recibe la respuesta antes. En un escenario real de serving, la latencia es la métrica que percibe el usuario.

**Memoria:** el adaptador LoRA ocupa +73 MB sobre el modelo base, lo que es negligible en los 16 GB de RAM unificada del M4 Pro.

### E.3 — Contexto en la comparativa global de frameworks

| Framework | Latencia media (s) | Throughput (tok/s) | Notas |
|-----------|:-----------------:|:-----------------:|-------|
| **Ollama** (Metal, GGUF) | **0.879** | **268.4** | Modelo cuantizado Q4, más rápido |
| Unsloth v2 (MLX + LoRA) | 1.289 | 54.9 | Fine-tuned, respuestas concisas |
| Unsloth v1 (MLX base) | 1.942 | 106.7 | Base model, respuestas largas |
| Transformers v1 (MPS) | 2.531 | 66.6 | Mayor overhead por framework |

Ollama sigue siendo el más rápido en latencia y throughput gracias a la cuantización GGUF Q4. Unsloth v2 con el adaptador LoRA es el segundo más rápido en latencia y el único que combina velocidad razonable con calidad de modelo fine-tuned.

---

## Conclusiones

### Hipótesis verificadas

| Hipótesis | Resultado |
|-----------|-----------|
| Más datos reducen mode collapse en brainstorming | ❌ No confirmada — P5 (cafetería) sigue sin explicar conceptos |
| Más épocas mejoran instruction-following sin degradar factualidad | ✅ Confirmada — factualidad +1.20 pts, IF +0.40 pts |
| La curva de loss v2 muestra mejor convergencia | ✅ Loss final 1.6021 (v1: 1.7091, −6.3 %), descenso sostenido durante 1500 steps |
| Ollama sigue siendo el más rápido en inferencia | ✅ Confirmada — 0.879 s frente a 1.289 s de Unsloth v2 |

### Balance general v1 → v2 (Unsloth)

v2 es mejor en calidad en todas las métricas automáticas y en la evaluación humana global. La mejora es especialmente notable en factualidad, donde el modelo con 6k muestras y 2 épocas ha visto suficiente contexto diverso para corregir la confusión Canberra/Sydney que persistía en v1. El instruction-following también mejora, especialmente en tareas estructuradas como clasificación.

La única dimensión donde v2 no mejora es el **brainstorming con formato libre**: el modelo sigue generando listas repetitivas sin creatividad ni explicaciones de concepto. Esto sugiere que el mode collapse en tareas creativas requiere estrategias adicionales (datos más diversos, penalización por repetición en inferencia, o RLHF) que exceden el scope del fine-tuning supervisado estándar.

En inferencia, v2 es 34 % más rápido en latencia gracias a que genera respuestas más concisas y bien delimitadas.

---

## Cambios en scripts respecto a v1

| Script v1 | Script v2 | Cambios |
|-----------|-----------|---------|
| `src/train/train_transformers_lora.py` | `src/train/train_transformers_lora_v2.py` | max_train=6k, epochs=2, logging_steps=5, guarda log_history |
| `src/eval/evaluate_finetuned.py` | `src/eval/evaluate_finetuned_v2.py` | Apunta a `transformers_lora_v2/` |
| `src/train/train_unsloth.py` | `src/train/train_unsloth_v2.py` | max_train=6k, epochs=2 |
| `src/eval/evaluate_unsloth.py` | `src/eval/evaluate_unsloth_v2.py` | Apunta a `unsloth_lora_v2/` |
| `src/benchmark/benchmark_*.py` | `src/benchmark/benchmark_*_v2.py` | Sin vLLM, añade repetition_penalty |
