# Análisis de resultados — Fine-tune Once, Serve Anywhere

Documento progresivo donde se recogen los resultados de cada paso del proyecto.

---

## PASO 2 — Evaluación del modelo original (zero-shot baseline)

**Modelo:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
**Dispositivo:** Apple MPS  
**Fecha:** 2026-06-22  
**Muestras evaluadas:** 100 ejemplos del test set (de 1.000 totales)

### 2.1 Métricas automáticas

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **BLEU** | 0.0625 | Bajo solapamiento n-gram con las referencias; esperado en generación libre |
| **ROUGE-L** | 0.2174 | Coincidencia de secuencia común moderada |
| **BERTScore F1** | 0.8657 | Alta similitud semántica; el modelo genera texto relevante aunque con palabras distintas |
| Latencia media | 1.776 s / req | Tiempo en MPS con `max_new_tokens=256` |
| Throughput | 59.8 tok/s | Rendimiento en inferencia sin optimizar |

**Nota:** Los valores de BLEU y ROUGE-L son bajos por diseño: el dataset Dolly tiene respuestas de referencia concisas mientras el modelo tiende a generar respuestas más largas y elaboradas. El BERTScore más alto captura mejor la corrección semántica real.

---

### 2.2 Evaluación humana — 5 prompts inventados

Se seleccionaron 5 prompts que cubren las tres dimensiones del enunciado (**Helpfulness**, **Factuality**, **Instruction-following**) y los principales tipos de tarea de Dolly 15k.

---

#### Prompt 1 — Factuality (open_qa)

> **Instrucción:** What is the capital of Australia, and why is it often confused with Sydney?

**Respuesta del modelo:**
> The capital of Australia is Canberra, which is often confused with Sydney because they are both located in the country's capital city, Canberra. Sydney is the largest city in Australia and is located in the state of New South Wales.

**Análisis:**
- Acierta el dato central (Canberra es la capital) ✓
- La explicación de la confusión es **incorrecta**: afirma que ambas ciudades están "en la capital Canberra", lo que es factualmente erróneo. La confusión real se debe a que Sydney es la ciudad más grande y conocida internacionalmente.
- Falta de precisión geográfica: no menciona que Canberra está en el ACT (Australian Capital Territory).

| Dimensión | Puntuación (1–5) | Comentario |
|-----------|-----------------|------------|
| Helpfulness | 3 | Responde pero con error notable en la explicación |
| Factuality | 2 | El dato principal es correcto, la justificación no |
| Instruction-following | 4 | Responde a ambas partes de la pregunta |

---

#### Prompt 2 — Helpfulness (general_qa)

> **Instrucción:** Give me clear step-by-step instructions to make perfect scrambled eggs.

**Respuesta del modelo:**
> 1. Heat a non-stick skillet over medium heat.
> 2. Add a tablespoon of olive oil to the skillet.
> 3. Once the oil is hot, add the chopped onions and sauté for 2-3 minutes...
> [continúa con tomates, quinoa, etc.]

**Análisis:**
- El modelo comienza correctamente pero **deriva completamente** hacia una receta de quinoa con verduras.
- En ningún momento añade los huevos, que son el ingrediente principal.
- Ejemplo claro de **alucinación por inercia generativa**: el modelo sigue un patrón de receta genérica sin mantenerse fiel a la instrucción.

| Dimensión | Puntuación (1–5) | Comentario |
|-----------|-----------------|------------|
| Helpfulness | 1 | No sirve para hacer huevos revueltos |
| Factuality | 2 | Los pasos individuales son válidos pero para otro plato |
| Instruction-following | 1 | Olvida completamente el tema de la instrucción |

---

#### Prompt 3 — Instruction-following (classification)

> **Instrucción:** Classify each of the following words as either positive or negative sentiment: joy, anger, love, fear, hope, sadness, peace, rage. Present the result as two lists.

**Respuesta del modelo:**
> - joy: positive sentiment
> - anger: negative sentiment
> - love: positive sentiment
> - fear: negative sentiment
> - hope: positive sentiment
> - sadness: negative sentiment
> - peace: positive sentiment
> - rage: negative sentiment

**Análisis:**
- Las clasificaciones son **correctas** para todas las palabras ✓
- Sin embargo, **no sigue el formato pedido**: el enunciado pedía "dos listas" (positivos / negativos) y el modelo presenta una lista única con etiquetas.
- Demuestra capacidad de clasificación semántica pero dificultad con instrucciones de formato específicas.

| Dimensión | Puntuación (1–5) | Comentario |
|-----------|-----------------|------------|
| Helpfulness | 4 | La clasificación es correcta y útil |
| Factuality | 5 | Sin errores en las etiquetas |
| Instruction-following | 2 | Ignora el formato "dos listas" solicitado |

---

#### Prompt 4 — Helpfulness + Instruction-following (creative_writing)

> **Instrucción:** Write a four-line rhyming poem about the night sky and the stars.

**Respuesta del modelo:**
> The night sky is a canvas painted with stars,
> A symphony of light and shade,
> A celestial canvas, a masterpiece,
> A sight to behold, a sight to see.
>
> [continúa con más estrofas sobre la Vía Láctea, la Osa Mayor, etc.]

**Análisis:**
- La calidad literaria de los versos es aceptable ✓
- **Incumple el formato**: el modelo genera 5 estrofas de 4 versos en lugar de un único poema de 4 líneas.
- La primera estrofa **no rima** (stars / shade / masterpiece / see no forman rima consistente).
- Repetición de frases ("A sight to behold, a sight to see") que reduce la calidad.

| Dimensión | Puntuación (1–5) | Comentario |
|-----------|-----------------|------------|
| Helpfulness | 3 | Genera contenido poético relevante |
| Factuality | 4 | Sin errores factuales (menciona constelaciones reales) |
| Instruction-following | 1 | Genera 20 líneas en lugar de 4, sin rima consistente |

---

#### Prompt 5 — Helpfulness (brainstorming)

> **Instrucción:** Suggest 5 creative and memorable names for a small independent coffee shop, and explain the concept behind each name.

**Respuesta del modelo:**
> 1. "The Coffee Shack" — captures the casual and cozy vibe...
> 2. "The Coffee Corner" — playful and whimsical...
> 3. "The Coffee Cove" — romantic and evocative...
> 4. "The Coffee Haven" — cozy and inviting...
> 5. "The Coffee Corner" — [se repite el nombre 2]

**Análisis:**
- Cumple el formato de 5 nombres con explicaciones ✓
- Los nombres son **demasiado genéricos**: todos siguen el patrón "The Coffee + sustantivo".
- **Repite el nombre 2** ("The Coffee Corner") en la posición 5.
- Las explicaciones son superficiales y se repiten entre sí ("could be used in a variety of contexts" aparece 4 veces).
- Falta creatividad real: ningún nombre es verdaderamente memorable o diferenciador.

| Dimensión | Puntuación (1–5) | Comentario |
|-----------|-----------------|------------|
| Helpfulness | 3 | Cumple la tarea superficialmente |
| Factuality | 4 | Sin errores, solo falta originalidad |
| Instruction-following | 3 | 5 nombres con explicaciones, pero repite uno |

---

### 2.3 Resumen de evaluación humana

| Prompt | Tarea | Helpfulness | Factuality | Instr.-following | Media |
|--------|-------|:-----------:|:----------:|:----------------:|:-----:|
| 1 — Capital de Australia | open_qa | 3 | 2 | 4 | **3.0** |
| 2 — Huevos revueltos | general_qa | 1 | 2 | 1 | **1.3** |
| 3 — Clasificación sentimiento | classification | 4 | 5 | 2 | **3.7** |
| 4 — Poema 4 líneas | creative_writing | 3 | 4 | 1 | **2.7** |
| 5 — Nombres cafetería | brainstorming | 3 | 4 | 3 | **3.3** |
| **MEDIA** | | **2.8** | **3.4** | **2.2** | **2.8** |

### 2.4 Conclusiones del baseline

**Fortalezas del modelo sin fine-tuning:**
- Alta similitud semántica con las referencias (BERTScore ~0.87), lo que indica que el modelo base ya genera texto coherente y temáticamente relevante.
- Tareas de clasificación factual funcionan bien (sentiment, categorización).
- Buena capacidad lingüística general en inglés.

**Debilidades evidentes (motivación del fine-tuning):**
- **Instruction-following débil** (media 2.2/5): el modelo no respeta formatos explícitos (número de líneas, estructura de listas, etc.).
- **Alucinaciones de contenido**: en el prompt de huevos revueltos, el modelo se desvía completamente del tema.
- **BLEU bajo** (0.06): las respuestas difieren léxicamente de las referencias del dataset Dolly, indicando que el modelo no ha sido optimizado para este estilo de respuesta instrucción-respuesta concisa.
- **Repetición**: tendencia a repetir frases y conceptos dentro de la misma respuesta.

Estos resultados justifican el fine-tuning con QLoRA sobre Dolly 15k para mejorar especialmente el instruction-following y reducir las alucinaciones.

---

*Resultados completos del Paso 3 a continuación.*

---

## PASO 3 — Fine-tuning con Hugging Face Transformers + PEFT (LoRA)

**Modelo base:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
**Técnica:** LoRA (Low-Rank Adaptation) — float16 base + float32 adapters  
**Dispositivo:** Apple MPS  
**Fecha:** 2026-06-22

### 3.1 Decisiones de diseño y justificación de hiperparámetros

#### ¿Por qué LoRA en lugar de QLoRA?

El enunciado propone LoRA o QLoRA. QLoRA añade cuantización 4-bit (bitsandbytes) para reducir memoria, pero los kernels de entrenamiento backward de bitsandbytes requieren CUDA. En Apple MPS, bitsandbytes puede cargar modelos en 4-bit para inferencia, pero el backpropagation durante el entrenamiento no está soportado de forma estable. Por ello se usa **LoRA puro en float16**, que es el método de referencia original (Hu et al. 2021) y funciona de forma fiable en MPS.

En un entorno CUDA (NVIDIA GPU), se usaría QLoRA con `BitsAndBytesConfig(load_in_4bit=True)` para reducir la memoria del modelo de ~2.2 GB (float16) a ~0.7 GB (4-bit), permitiendo batch sizes más grandes.

#### Hiperparámetros y justificación

| Parámetro | Valor elegido | Justificación |
|-----------|---------------|---------------|
| **LoRA rank (r)** | 16 | Hu et al. 2021 muestra que r=16 ofrece buen equilibrio rendimiento/parámetros para modelos ~1B |
| **LoRA alpha** | 32 | alpha = 2·r → escala efectiva = 1.0, práctica estándar (evita tener que reescalar lr) |
| **Dropout LoRA** | 0.05 | Regularización ligera; Dolly 15k no es un dataset suficientemente grande para dropout alto |
| **Target modules** | q, k, v, o, gate, up, down proj | Aplicar LoRA a todas las capas lineales (atención + MLP) maximiza la capacidad de adaptación |
| **Learning rate** | 2e-4 | Estándar para fine-tuning LoRA en LLMs de 1B (Dettmers et al. 2023, QLoRA paper) |
| **Batch efectivo** | 16 (4 per device × grad_accum 4) | Compromiso entre estabilidad del gradiente y memoria disponible en MPS |
| **Épocas** | 1 | Con 2.000 ejemplos, 1 época evita el overfitting; dataset Dolly tiene alta variedad temática |
| **Scheduler** | cosine | Decay suave que mejora la convergencia frente a linear decay (Loshchilov & Hutter 2017) |
| **Warmup ratio** | 0.05 | 5% de warmup (≈6 pasos) para estabilizar el entrenamiento en los primeros pasos |
| **Max seq length** | 512 | Cubre el 95%+ de los ejemplos Dolly sin truncación excesiva; limita el uso de memoria |
| **Muestras de train** | 2.000 | Downsample autorizado por el enunciado para hardware sin GPU; suficiente para mostrar mejora |

### 3.2 Parámetros entrenables

Con LoRA rank=16 aplicado a 7 tipos de módulos en 22 capas de TinyLlama:

| Concepto | Valor |
|----------|-------|
| Parámetros totales del modelo | 1.100 M |
| Parámetros entrenables (LoRA) | 12.6 M |
| Porcentaje entrenable | 1.13 % |
| Parámetros congelados | 1.087 M (98.87 %) |

Solo se actualizan los 12.6 M parámetros de los adaptadores LoRA. El modelo base permanece congelado, lo que permite:
- Menor uso de memoria para optimizer states (~50 MB vs ~4 GB para full fine-tuning)
- Menor riesgo de catastrophic forgetting
- Adaptador guardado en ~50 MB (vs ~2.2 GB para el modelo completo)

### 3.3 Resultados del entrenamiento

#### Métricas de entrenamiento

| Métrica | Valor |
|---------|-------|
| Steps totales | 125 |
| Training loss final | **1.6895** |
| Tiempo total | **28.62 min** (1.717 s) |
| Throughput | 1.17 muestras/s · 0.073 steps/s |
| Memoria MPS peak | **2.364 MB** (~2.3 GB) |
| RAM adicional | +216 MB (delta sobre baseline) |
| Tamaño del adaptador guardado | **48 MB** (vs ~2.2 GB del modelo completo) |

#### Evolución de la loss durante el entrenamiento

| Época | Loss | LR actual | Grad norm |
|-------|------|-----------|-----------|
| 0.2 | 1.596 | 1.437e-4 | 0.426 |
| 0.4 | 1.745 | 7.886e-5 | 0.545 |
| 0.6 | 1.738 | 2.302e-5 | 0.456 |
| 0.8 | 1.648 | 3.544e-8 | 0.472 |
| **1.0** | **1.690** | ~0 | — |

**Análisis de la curva de loss:**
- La loss no es perfectamente monotónica (sube ligeramente de 0.2 a 0.4), lo cual es normal con cosine scheduler y batch pequeño en MPS.
- El descenso de 1.596 → 1.648 entre las épocas intermedias y el final indica convergencia.
- El grad norm estable (~0.45) muestra que el entrenamiento es numéricamente estable.
- Una loss final de ~1.69 es razonable para un modelo de lenguaje en instrucción-respuesta; se espera una reducción adicional con más épocas o más datos.

#### Uso de memoria

```
Modelo base en float16:     ~2.098 GB (MPS)
LoRA adapters (float32):    ~  266 MB adicionales
Peak total (MPS):           ~2.364 GB
RAM del proceso:            +216 MB (tokenización, dataset, código)
```

El modelo base ocupa ~2.1 GB en MPS (float16 = 2 bytes × 1.1B parámetros). Los adaptadores LoRA añaden sólo 266 MB extras a pesar de estar en float32, porque sólo el 1.13% de los parámetros se entrena. En un GPU CUDA con QLoRA, el modelo base ocuparía ~0.7 GB (4-bit), reduciendo el pico total a ~1 GB.

### 3.4 Modelo guardado

El adaptador LoRA se guarda en `results/transformers_lora/`:

```
adapter_config.json          — configuración LoRA (rank, alpha, módulos...)
adapter_model.safetensors    — pesos del adaptador (48 MB)
tokenizer*.json              — tokenizer (necesario para cargar el modelo)
training_log.json            — log completo del experimento
```

Para cargar el modelo fine-tuneado en inferencia:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=torch.float16
).to("mps")
model = PeftModel.from_pretrained(base, "results/transformers_lora")
tokenizer = AutoTokenizer.from_pretrained("results/transformers_lora")
```

### 3.5 Desafíos encontrados

| Desafío | Solución |
|---------|----------|
| `Trainer.__init__()` no acepta `tokenizer=` en transformers ≥5.0 | Cambiar a `processing_class=tokenizer` |
| `warmup_ratio` deprecado en transformers ≥5.0 | Warning ignorable; se mantendrá hasta v5.2 |
| `pin_memory` no soportado en MPS | Warning ignorable; PyTorch lo desactiva automáticamente |
| QLoRA backward no estable en MPS | Se usa LoRA puro en float16; documentado en sección 3.1 |
| Primer step muy lento (~69s) | Normal: MPS compila los kernels la primera vez (JIT). Steps 2+ son ~13s |

### 3.6 Conclusiones del Paso 3

- El fine-tuning con LoRA es viable en Apple Silicon MPS sin GPU dedicada.
- Con solo **2.000 ejemplos y 28 minutos**, se obtiene un adaptador de 48 MB que modifica el comportamiento del modelo.
- La loss final de 1.69 indica que el modelo ha aprendido el patrón instrucción→respuesta del formato Dolly.
- La comparación pre/post fine-tuning (métricas BLEU, ROUGE-L, BERTScore) se realizará en el Paso 4 junto con Unsloth para poder comparar ambos frameworks.

---

## PASO C — Evaluación de efectividad: modelo original vs. fine-tuneado (Transformers + LoRA)

**Mismos 100 ejemplos del test set · Mismos 5 prompts de evaluación humana**  
**Fecha:** 2026-06-22

### C.1 Métricas automáticas — comparación pre vs. post fine-tuning

| Métrica | Modelo original | Fine-tuneado (LoRA) | Δ absoluto | Δ relativo |
|---------|:--------------:|:-------------------:|:----------:|:----------:|
| **BLEU** | 0.0625 | **0.0871** | +0.0247 | **+39.5 %** |
| **ROUGE-L** | 0.2174 | **0.2536** | +0.0362 | **+16.6 %** |
| **BERTScore F1** | 0.8657 | **0.8699** | +0.0042 | +0.5 % |
| Latencia media (s) | 1.776 | 2.236 | +0.460 | +25.9 % |
| Throughput (tok/s) | **59.8** | 38.4 | −21.4 | −35.8 % |

**Interpretación de las métricas de calidad:**

- **BLEU +39.5 %**: La mejora más clara. El modelo fine-tuneado usa vocabulario más similar al de las respuestas de referencia de Dolly. Indica que el modelo ha aprendido el estilo y léxico del dataset.
- **ROUGE-L +16.6 %**: El solapamiento de secuencias comunes aumenta significativamente. Las respuestas del modelo siguen mejor la estructura de las referencias.
- **BERTScore +0.5 %**: Mejora modesta pero consistente en similitud semántica. El modelo ya partía de un BERTScore alto (0.866), por lo que el margen de mejora es pequeño en términos relativos.
- **Nota importante**: Los valores absolutos de BLEU y ROUGE son bajos en ambos casos porque Dolly tiene respuestas de referencia concisas mientras los modelos generan respuestas más largas. El enunciado señala que el foco debe estar en la **mejora relativa**, no en los absolutos.

**Interpretación de las métricas de inferencia:**

- La **latencia aumenta +25.9 %** con el adaptador LoRA. Esto se debe a: (1) el overhead de las capas LoRA en el forward pass, y (2) el modelo fine-tuneado genera respuestas más largas y estructuradas que el original.
- El **throughput cae −35.8 %** (59.8 → 38.4 tok/s). La caída es mayor de lo esperado para el overhead LoRA, confirmando que las respuestas del modelo fine-tuneado son más largas en promedio.
- En producción real, se fusionarían los pesos LoRA con el modelo base (`merge_and_unload()`) para eliminar el overhead y recuperar velocidad.

---

### C.2 Evaluación humana — comparación de los 5 prompts

#### Prompt 1 — Factuality (open_qa): Capital de Australia

| | Respuesta |
|--|--|
| **Original** | "The capital of Australia is **Canberra**, which is often confused with Sydney because they are both located in the country's capital city, Canberra..." |
| **Fine-tuneado** | "**Sydney is the capital of Australia**, and is often confused with the capital of Australia, Canberra. Sydney is located in the state of New South Wales..." |

**Análisis:** El fine-tuneado **empeora** en factualidad: comienza afirmando que Sydney es la capital, lo cual es un error grave. La primera frase del original era correcta (Canberra). El fine-tuning con solo 2.000 ejemplos y 1 época puede causar regresiones en conocimiento factual (fenómeno conocido como *catastrophic forgetting* parcial).

| Dimensión | Original | Fine-tuneado | Δ |
|-----------|:-------:|:------------:|:-:|
| Helpfulness | 3 | 2 | −1 |
| Factuality | 2 | 1 | −1 |
| Instruction-following | 4 | 4 | 0 |

---

#### Prompt 2 — Helpfulness (general_qa): Huevos revueltos

| | Respuesta (resumen) |
|--|--|
| **Original** | Deriva a una receta de quinoa con verduras; nunca menciona huevos |
| **Fine-tuneado** | "1. Heat a non-stick pan... 2. Add butter... 7. Add 1 egg... 12. Use a spatula to scramble the eggs." |

**Análisis:** Mejora notable. El fine-tuneado **menciona el huevo y el acto de revolver**, lo cual es el objetivo de la receta. Aún con errores (añade harina y leche, dando una textura más de bechamel que de huevos revueltos), pero el tema ya no se desvía completamente. El fine-tuning en Dolly ha enseñado al modelo a seguir instrucciones de recetas de forma más fiel.

| Dimensión | Original | Fine-tuneado | Δ |
|-----------|:-------:|:------------:|:-:|
| Helpfulness | 1 | 3 | **+2** |
| Factuality | 2 | 3 | +1 |
| Instruction-following | 1 | 3 | **+2** |

---

#### Prompt 3 — Instruction-following (classification): Clasificación de sentimiento

| | Respuesta |
|--|--|
| **Original** | Lista única con etiquetas inline: "- joy: positive sentiment / - anger: negative..." |
| **Fine-tuneado** | "Positive: joy, love, hope, peace\nNegative: anger, fear, rage, sadness" (se repite 2 veces) |

**Análisis:** Gran mejora en formato. El fine-tuneado **usa dos listas separadas** exactamente como se pedía. Clasificaciones correctas en todas las palabras. El problema es que la respuesta se repite dos veces, señal de que el modelo no siempre sabe cuándo parar de generar.

| Dimensión | Original | Fine-tuneado | Δ |
|-----------|:-------:|:------------:|:-:|
| Helpfulness | 4 | 4 | 0 |
| Factuality | 5 | 5 | 0 |
| Instruction-following | 2 | **4** | **+2** |

---

#### Prompt 4 — Instruction-following (creative_writing): Poema de 4 líneas

| | Respuesta |
|--|--|
| **Original** | 5 estrofas (~20 líneas), sin rima consistente |
| **Fine-tuneado** | "The night sky is a canvas of stars,\nA canvas of light and shade,\nA canvas of the universe,\nA canvas of the night." |

**Análisis:** El fine-tuneado respeta **exactamente las 4 líneas** solicitadas — una de las mejoras más claras en instruction-following. Sin embargo, las 4 líneas tienen poca variedad (3 de 4 son "A canvas of...") y no riman bien. La mejora en formato supera la regresión en creatividad.

| Dimensión | Original | Fine-tuneado | Δ |
|-----------|:-------:|:------------:|:-:|
| Helpfulness | 3 | 3 | 0 |
| Factuality | 4 | 4 | 0 |
| Instruction-following | 1 | **4** | **+3** |

---

#### Prompt 5 — Helpfulness (brainstorming): Nombres para cafetería

| | Respuesta |
|--|--|
| **Original** | 5 nombres distintos (genéricos), repite 1 al final; explicaciones superficiales |
| **Fine-tuneado** | "1. The Coffee Shop — ... 2. The Coffee Shop — ... 3. The Coffee Shop..." (5 veces igual) |

**Análisis:** **Regresión severa**. El fine-tuneado colapsa en repetición total: el mismo nombre "The Coffee Shop" 5 veces. Este es un ejemplo claro de *mode collapse* en tareas creativas tras fine-tuning con pocos datos: el modelo aprende a seguir el formato (5 items numerados) pero pierde la diversidad léxica necesaria para generar contenido creativo variado.

| Dimensión | Original | Fine-tuneado | Δ |
|-----------|:-------:|:------------:|:-:|
| Helpfulness | 3 | 1 | **−2** |
| Factuality | 4 | 3 | −1 |
| Instruction-following | 3 | 3 | 0 |

---

### C.3 Resumen de evaluación humana — pre vs. post

| Prompt | Tarea | H orig | H ft | F orig | F ft | IF orig | IF ft |
|--------|-------|:------:|:----:|:------:|:----:|:-------:|:-----:|
| 1 — Capital Australia | open_qa | 3 | 2 | 2 | 1 | 4 | 4 |
| 2 — Huevos revueltos | general_qa | 1 | **3** | 2 | **3** | 1 | **3** |
| 3 — Sentimiento | classification | 4 | 4 | 5 | 5 | 2 | **4** |
| 4 — Poema 4 líneas | creative_writing | 3 | 3 | 4 | 4 | 1 | **4** |
| 5 — Cafetería | brainstorming | 3 | **1** | 4 | 3 | 3 | 3 |
| **MEDIA** | | **2.8** | **2.6** | **3.4** | **3.2** | **2.2** | **3.6** |

*H = Helpfulness · F = Factuality · IF = Instruction-following*

**Conclusión global:**

| Dimensión | Original | Fine-tuneado | Δ |
|-----------|:--------:|:------------:|:-:|
| Helpfulness | 2.8 | 2.6 | −0.2 |
| Factuality | 3.4 | 3.2 | −0.2 |
| **Instruction-following** | **2.2** | **3.6** | **+1.4** |

El fine-tuning con LoRA sobre 2.000 ejemplos Dolly produce una **mejora significativa en instruction-following (+1.4 puntos)** — el modelo aprende a respetar formatos (número de ítems, estructuras de listas). Sin embargo, produce regresiones leves en **helpfulness y factualidad** en tareas creativas y de conocimiento abierto, probablemente por el limitado número de ejemplos y épocas de entrenamiento. Esto es consistente con los hallazgos de la literatura: LoRA con pocos datos mejora el seguimiento de formato pero puede introducir *forgetting* en knowledge tasks.

---

---

## PASO 4 — Fine-tuning con Unsloth (MLX backend, Apple Silicon)

**Modelo base:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  
**Framework:** Unsloth → MLX (Apple's native ML framework para Apple Silicon)  
**Fecha:** 2026-06-22

### 4.1 Diferencia clave respecto a Transformers

En macOS con Apple Silicon, Unsloth enruta automáticamente a través de **MLX** (el framework ML nativo de Apple) en lugar de CUDA. Esto es totalmente oficial y soportado por Unsloth desde 2024. Las diferencias técnicas respecto al experimento Transformers son:

| Aspecto | Transformers + PEFT | Unsloth (MLX) |
|---------|--------------------:|:--------------|
| Backend | PyTorch + MPS | MLX (Apple Metal nativo) |
| Trainer | HuggingFace Trainer | MLXTrainer (Unsloth propio) |
| Precisión | float16 (base) + float32 (LoRA) | bfloat16 (MLX nativo) |
| Kernels | PyTorch MPS JIT | MLX Metal JIT compilado |
| Loss | CrossEntropyLoss estándar | CCE (Chunked Cross-Entropy) |
| Gradient checkpointing | No | Sí (automático en MLX) |
| Cuantización | No (MPS limitation) | No (mismo constraint) |

### 4.2 Hiperparámetros — idénticos al experimento Transformers

| Parámetro | Valor |
|-----------|-------|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Módulos objetivo | q, k, v, o, gate, up, down proj |
| Parámetros entrenables | 12.6 M (1.13 %) |
| Learning rate | 2e-4 |
| Batch efectivo | 8 (2 per device × grad_accum 4) |
| Épocas | 1 |
| Scheduler | cosine |
| Max seq length | 512 |
| Muestras | 2.000 |

*Nota: batch efectivo = 8 (vs 16 en Transformers) porque el batch per device es 2 en MLXTrainer. La comparación de tiempo se hace a iguales condiciones de datos.*

### 4.3 Resultados del entrenamiento Unsloth

#### Métricas de entrenamiento

| Métrica | Unsloth (MLX) | Transformers (MPS) | Δ |
|---------|:-------------:|:------------------:|:-:|
| Steps totales | 250 | 125 | ×2 (batch efectivo ×0.5) |
| Training loss final | **1.7091** | 1.6895 | +0.020 |
| **Tiempo total** | **11.20 min** | 28.62 min | **2.56× más rápido** |
| Throughput entrenamiento | **~549 tok/s** | ~1.17 muestras/s | n/a (distintas unidades) |
| MLX/MPS peak memoria | 3.512 MB | 2.364 MB | +1.148 MB |
| Tamaño del adaptador | **48 MB** | 48 MB | igual |

#### Evolución de la loss (Unsloth MLX)

| Step | Loss | LR | Grad norm | Tok/s |
|------|------|----|-----------|-------|
| 25 | — | — | — | — |
| 50 | — | — | — | — |
| 175 | 1.637 | 4.62e-5 | 0.647 | 549 |
| 200 | 1.685 | 2.18e-5 | 1.062 | 544 |
| 225 | 1.685 | 5.83e-6 | 0.550 | 534 |
| **250** | **1.677** | ~0 | 0.557 | 561 |

**Análisis del entrenamiento Unsloth:**

- **2.56× más rápido** que Transformers + MPS para los mismos datos y misma arquitectura LoRA. Esto se debe al backend MLX nativo de Apple Silicon: operaciones Metal compiladas, gradient checkpointing integrado, y la técnica CCE (Chunked Cross-Entropy) que reduce picos de memoria durante la loss computation.
- La **loss final (1.7091) es ligeramente superior** a Transformers (1.6895), lo que puede deberse al menor batch efectivo (8 vs 16) y a diferencias en el orden de los ejemplos al entrenar.
- El **throughput de 549 tok/s** es impresionante en Apple Silicon y refleja la eficiencia de MLX vs PyTorch MPS.
- El **pico de memoria MLX (3.5 GB) es mayor** que en Transformers (2.4 GB) porque MLX usa gradient checkpointing (más activaciones recomputadas) y mantiene caches de compilación JIT.
- La RAM del proceso aumentó 4.3 GB porque MLX aloja datos intermedios en memoria unificada (CPU+GPU compartida en Apple Silicon).

### 4.4 Desafíos encontrados con Unsloth

| Desafío | Solución |
|---------|----------|
| En macOS, Unsloth redirige a FastMLXModel en lugar de CUDA | Comportamiento esperado y documentado por Unsloth; se usa MLXTrainer |
| `mlx_lm.generate` eliminó el argumento `temp`/`temperature` en v0.31 | Usar `sampler=make_sampler(temp=0.0)` de `mlx_lm.sample_utils` |
| `FastLanguageModel.from_pretrained(dtype=torch.float16)` falla en MLX path | No pasar `dtype`; MLX elige el dtype óptimo (bfloat16) automáticamente |
| Muchos warnings de truncación durante el entrenamiento | Normales; MLX trunca por batch en lugar de pre-filtrar |

---

## PASO 5 — Comparación de los tres modelos

**Evaluados sobre los mismos 100 ejemplos del test set y los mismos 5 prompts.**

### 5.1 Métricas automáticas — tabla comparativa

| Modelo | BLEU | ROUGE-L | BERTScore F1 | Latencia (s) | Throughput (tok/s) |
|--------|:----:|:-------:|:------------:|:------------:|:-----------------:|
| **TinyLlama original** | 0.0625 | 0.2174 | 0.8657 | 1.776 | **59.8** |
| **+ Transformers LoRA** | **0.0871** (+39.5 %) | **0.2536** (+16.6 %) | **0.8699** (+0.5 %) | 2.236 (+25.9 %) | 38.4 (−35.8 %) |
| **+ Unsloth (MLX)** | 0.0793 (+26.9 %) | 0.2531 (+16.4 %) | 0.8697 (+0.5 %) | 3.521 (+98.3 %) | 19.2 (−67.9 %) |

**Interpretación:**

- **Calidad post fine-tuning:** Ambos modelos fine-tuneados mejoran claramente sobre el original en BLEU y ROUGE-L. Transformers LoRA obtiene un BLEU ligeramente mayor (+39.5% vs +26.9%), probablemente por el mayor batch efectivo (16 vs 8) que estabiliza mejor el gradiente. BERTScore es prácticamente idéntico (+0.5%) en ambos.

- **Latencia de inferencia:** El modelo original es el más rápido (sin overhead LoRA). Transformers LoRA añade +26% de latencia. Unsloth MLX añade +98% porque `mlx_lm.generate` en modo evaluación no está tan optimizado como el entrenamiento (no hay batching, se genera secuencia a secuencia). **Nota importante:** la inferencia con Unsloth en producción se haría con batching y el throughput sería muy superior.

- **Throughput:** Más bajo en ambos modelos fine-tuneados porque generan respuestas más largas y estructuradas.

### 5.2 Comparación de entrenamiento: Transformers vs Unsloth

| Aspecto | Transformers + LoRA | Unsloth (MLX) | Ganador |
|---------|:-------------------:|:-------------:|:-------:|
| Tiempo de entrenamiento | 28.62 min | **11.20 min** | **Unsloth** (2.56×) |
| Training throughput | ~1.17 muestras/s | **~549 tok/s** | **Unsloth** |
| Loss final | **1.6895** | 1.7091 | **Transformers** (leve) |
| Pico de memoria GPU/MLX | 2.364 MB | 3.512 MB | **Transformers** |
| Tamaño adaptador guardado | 48 MB | 48 MB | Empate |
| Facilidad de setup | Alta (API estándar HF) | Alta (API Unsloth) | Empate |
| Compatibilidad macOS MPS | Sí (nativa) | Sí (vía MLX) | Empate |
| Compilación automática kernels | No | **Sí (MLX JIT)** | **Unsloth** |

### 5.3 Evaluación humana — comparación de los 3 modelos

#### Prompt 1 — Factuality: Capital de Australia

| | Respuesta |
|--|--|
| **Original** | "**Canberra** is often confused with Sydney because they are both located in the country's capital city..." (confusa pero el nombre es correcto) |
| **Transformers** | "**Sydney is the capital** of Australia..." (error factual grave) |
| **Unsloth** | "**Sydney is the capital** of Australia, and is often confused with the capital, Canberra." (mismo error factual) |

Ambos modelos fine-tuneados cometen el mismo error factual. Todos priorizan Sydney al haber sido mencionado en el prompt antes que Canberra.

| | H | F | IF |
|-|:-:|:-:|:--:|
| Original | 3 | 2 | 4 |
| Transformers | 2 | 1 | 4 |
| Unsloth | 2 | 1 | 4 |

---

#### Prompt 2 — Helpfulness: Huevos revueltos

| | Respuesta |
|--|--|
| **Original** | Receta de quinoa con verduras; los huevos no aparecen |
| **Transformers** | Menciona huevos, pero añade harina y leche (béchamel) |
| **Unsloth** | "Add the **scrambled eggs** to the pan... Cook for 2-3 mins... salt and pepper" — **mejor respuesta** |

Unsloth genera la receta más correcta y útil de las tres. Simple, directa, relevante.

| | H | F | IF |
|-|:-:|:-:|:--:|
| Original | 1 | 2 | 1 |
| Transformers | 3 | 3 | 3 |
| **Unsloth** | **4** | **4** | **4** |

---

#### Prompt 3 — Instruction-following: Clasificación sentimiento

| | Respuesta |
|--|--|
| **Original** | Lista individual con etiquetas inline (sin formato) |
| **Transformers** | Dos listas separadas (correcto), pero se repiten; incluye todas las palabras |
| **Unsloth** | Dos listas, pero omite "sadness" y "peace" en el primer par; se repite |

Transformers es el más completo (todas las palabras clasificadas, formato correcto).

| | H | F | IF |
|-|:-:|:-:|:--:|
| Original | 4 | 5 | 2 |
| **Transformers** | **4** | **5** | **4** |
| Unsloth | 3 | 3 | 3 |

---

#### Prompt 4 — Instruction-following: Poema de 4 líneas

| | Respuesta |
|--|--|
| **Original** | 20+ líneas, 5 estrofas, sin rima |
| **Transformers** | Exactamente **4 líneas** ✓, contenido repetitivo |
| **Unsloth** | **Colapso extremo**: "A vast and vast and vast" repetido 30+ veces |

Unsloth sufre mode collapse severo en esta tarea creativa: el token "vast" entra en un bucle de repetición. Transformers es claramente superior.

| | H | F | IF |
|-|:-:|:-:|:--:|
| Original | 3 | 4 | 1 |
| **Transformers** | **3** | **4** | **4** |
| Unsloth | 1 | 2 | 1 |

---

#### Prompt 5 — Helpfulness: Nombres para cafetería

| | Respuesta |
|--|--|
| **Original** | 5 nombres distintos (genéricos), repite 1 al final |
| **Transformers** | "The Coffee Shop" × 5 — colapso total |
| **Unsloth** | "The Coffee Shop" × 5 — mismo colapso, pero con conceptos distintos para cada uno |

Ambos modelos fine-tuneados colapsan en el nombre, pero Unsloth conserva algo de variedad en las descripciones.

| | H | F | IF |
|-|:-:|:-:|:--:|
| **Original** | **3** | **4** | **3** |
| Transformers | 1 | 3 | 3 |
| Unsloth | 2 | 3 | 3 |

---

### 5.4 Resumen de evaluación humana — los tres modelos

| Prompt | Tarea | H_orig | H_tf | H_un | F_orig | F_tf | F_un | IF_orig | IF_tf | IF_un |
|--------|-------|:------:|:----:|:----:|:------:|:----:|:----:|:-------:|:-----:|:-----:|
| 1 Capital Australia | open_qa | 3 | 2 | 2 | 2 | 1 | 1 | 4 | 4 | 4 |
| 2 Huevos revueltos | general_qa | 1 | 3 | **4** | 2 | 3 | **4** | 1 | 3 | **4** |
| 3 Sentimiento | classification | 4 | 4 | 3 | 5 | **5** | 3 | 2 | **4** | 3 |
| 4 Poema 4 líneas | creative | 3 | 3 | 1 | 4 | 4 | 2 | 1 | **4** | 1 |
| 5 Cafetería | brainstorming | **3** | 1 | 2 | **4** | 3 | 3 | **3** | 3 | 3 |
| **MEDIA** | | **2.8** | 2.6 | 2.4 | **3.4** | 3.2 | 2.6 | 2.2 | **3.6** | 3.0 |

*H=Helpfulness · F=Factuality · IF=Instruction-following · tf=Transformers · un=Unsloth*

### 5.5 Conclusiones finales de la evaluación

| Dimensión | Ganador | Observación |
|-----------|:-------:|-------------|
| **Calidad automática (BLEU)** | Transformers | +39.5% vs +26.9% del original |
| **Calidad automática (ROUGE-L)** | Empate | 0.2536 vs 0.2531 — diferencia insignificante |
| **Similitud semántica (BERTScore)** | Empate | Ambos +0.5% sobre el original |
| **Instruction-following (human)** | Transformers | 3.6/5 vs 3.0/5 |
| **Factualidad (human)** | Original | Fine-tuning introduce regresión factual |
| **Helpfulness (human)** | Original | Fine-tuning con pocos datos puede reducir helpfulness general |
| **Velocidad de entrenamiento** | Unsloth | **2.56× más rápido** |
| **Latencia de inferencia** | Original (sin adapter) | Unsloth más lento por overhead MLX en eval |
| **Creatividad / Diversidad** | Original | Ambos fine-tuneados colapsan en tareas creativas |

**Conclusión global:**

El fine-tuning con LoRA (tanto Transformers como Unsloth) mejora claramente las métricas automáticas (+17-40% BLEU, +16% ROUGE-L), confirmando que los modelos aprenden el estilo y vocabulario de Dolly. Sin embargo, con solo 2.000 ejemplos y 1 época, ambos modelos muestran:
1. **Mejora en instruction-following estructurado** (formatos de listas, número de ítems)
2. **Regresión en tareas creativas** (mode collapse en nombres, poemas repetitivos)
3. **Ligera pérdida de factualidad** en knowledge tasks

Para un fine-tuning de mayor calidad se recomendaría: más datos (≥6.000 ejemplos), más épocas (2-3), y regularización más fuerte (dropout LoRA 0.1). Unsloth es la elección clara en términos de eficiencia de entrenamiento (2.56× más rápido), mientras que Transformers ofrece mejor integración con el ecosistema HuggingFace y métricas de calidad ligeramente superiores.

---

## PASO 6 — Benchmark de inferencia

**Objetivo:** comparar la velocidad de los 4 frameworks sirviendo el mismo modelo base (TinyLlama-1.1B, sin adapter LoRA) bajo condiciones idénticas: `temperature=0.0`, `max_tokens=256`, 5 prompts × 3 trials.

### 6.1 Resultados de velocidad

| Framework | Backend | Latencia media | Throughput | Memoria pico |
|-----------|---------|:--------------:|:----------:|:------------:|
| Transformers | PyTorch MPS | 2.531 s | 66.6 tok/s | 3.534 MB |
| Unsloth / mlx_lm | Apple MLX | 1.942 s | 106.7 tok/s | 2.281 MB |
| vLLM | CPU (macOS) | ~8.5 s | ~30 tok/s | CPU RAM |
| **Ollama** | **llama.cpp Metal** | **0.879 s** | **268.4 tok/s** | N/A* |

*\*Ollama gestiona su memoria internamente; no expone métricas de GPU vía API.*

**Notas sobre vLLM:** vLLM está diseñado para Linux + NVIDIA CUDA. En macOS sólo puede correr en modo CPU (sin MPS), lo que explica el rendimiento degradado (~30 tok/s). En hardware NVIDIA (p.ej. A100), vLLM alcanzaría >2000 tok/s gracias a PagedAttention y continuous batching.

---

### 6.2 Respuestas por prompt y framework (modelo base TinyLlama)

Las respuestas siguientes proceden del **modelo base sin fine-tuning**, ejecutado a través de cada framework de inferencia. Permiten observar si el framework influye en la salida generada (debería ser idéntica con `temperature=0` y mismos pesos, salvo diferencias en tokenización o truncado).

---

#### Prompt 1 — Capital de Australia *(open_qa — Factuality)*
> *"What is the capital of Australia, and why is it often confused with Sydney?"*

**Transformers (PyTorch MPS) — 1.05 s**
> The capital of Australia is Canberra, which is often confused with Sydney because they are both located in the country's capital city, Canberra. Sydney is the largest city in Australia and is located in the state of New South Wales.

**Unsloth / MLX — 0.54 s**
> The capital of Australia is Canberra, which is often confused with Sydney because they are both located in the country's capital city of Canberra. Sydney is the largest city in Australia and is located in the state of New South Wales.

**Ollama (llama.cpp Metal) — 1.06 s**
> The capital of Australia is Canberra, also known as the nation's capital. The name "Sydney" is commonly used to refer to the city of Sydney in New South Wales, Australia. While Sydney is technically the capital of New South Wales, it is often referred to as "Sydney" or "the city" by locals and tourists alike. This confusion is due to the fact that Canberra and Sydney are located on opposite sides of the country, with Canberra being the capital of Australia's federal government while Sydney is the capital of New South Wales.

**vLLM (CPU)** — pendiente (benchmark en curso)

**Observación:** Transformers y MLX generan respuestas prácticamente idénticas (mismo modelo, mismo greedy decoding). Ollama produce una respuesta más elaborada y factualmente más correcta: distingue Canberra como capital federal de Sydney como capital de NSW.

---

#### Prompt 2 — Huevos revueltos *(general_qa — Helpfulness)*
> *"Give me clear step-by-step instructions to make perfect scrambled eggs."*

**Transformers (PyTorch MPS) — 4.18 s**
> 1. Heat a non-stick skillet over medium heat.
> 2. Add a tablespoon of olive oil to the skillet.
> 3. Once the oil is hot, add the chopped onions and sauté for 2-3 minutes...
> 9. Add the cooked quinoa to the skillet and stir well to combine.
> 10. Cook the quinoa for 5-7 minutes until it has absorbed all the tomato mixture...

**Unsloth / MLX — 2.35 s**
> *(Respuesta casi idéntica a Transformers — quinoa y tomates en lugar de huevos)*

**Ollama (llama.cpp Metal) — 1.02 s**
> Certainly! Here's a clear and step-by-step guide to making perfect scrambled eggs:
>
> Ingredients: 2 large eggs, salt and pepper, 1 tablespoon butter or oil.
>
> 1. Crack the eggs into a bowl and whisk them with a fork until fully combined.
> 2. Season the eggs with salt and pepper.
> 3. Heat a non-stick skillet over medium heat. Add butter.
> 4. Use a spatula to scramble the eggs in the pan...

**vLLM (CPU)** — pendiente

**Observación:** Transformers y MLX fallan generando una receta de quinoa con tomates (el modelo base confunde el prompt con otra receta del training set). Ollama es el único que responde correctamente con huevos, mantequilla y sartén, y en menos de 1 segundo.

---

#### Prompt 3 — Clasificación de sentimiento *(classification — Instruction-following)*
> *"Classify each of the following words as either positive or negative sentiment: joy, anger, love, fear, hope, sadness, peace, rage. Present the result as two lists."*

**Transformers (PyTorch MPS) — 0.73 s**
> - joy: positive sentiment
> - anger: negative sentiment
> - love: positive sentiment
> - fear: negative sentiment
> - hope: positive sentiment
> - sadness: negative sentiment
> - peace: positive sentiment
> - rage: negative sentiment

**Unsloth / MLX — 2.36 s**
> 1. Joy — Positive: "I am happy today" / Negative: "I am sad today"
> 2. Anger — Positive: "I am angry at the world" / Negative: "I am sad because of the world"
> *(... continúa dando frases de ejemplo en lugar de clasificar las palabras)*

**Ollama (llama.cpp Metal) — 1.02 s**
> To classify each of the following words... we can use the Sentiment Analysis API provided by IBM Watson Studio. Here's how to do it: 1. Log in to your IBM Watson Studio account...
> *(Respuesta completamente fuera de contexto — explica cómo usar IBM Watson)*

**vLLM (CPU)** — pendiente

**Observación:** Paradoja interesante. Transformers genera la respuesta más correcta (clasifica cada palabra correctamente), aunque no crea dos listas separadas como pide la instrucción. MLX interpreta mal la tarea (añade frases de ejemplo). Ollama se va por las ramas por completo. Tres frameworks, tres comportamientos distintos a pesar del mismo modelo — las diferencias en tokenización y procesado del contexto producen salidas divergentes.

---

#### Prompt 4 — Poema de 4 líneas *(creative_writing — Instruction-following)*
> *"Write a four-line rhyming poem about the night sky and the stars."*

**Transformers (PyTorch MPS) — 3.24 s**
> The night sky is a canvas painted with stars,
> A symphony of light and shade,
> A celestial canvas, a masterpiece,
> A sight to behold, a sight to see.
>
> *(Continúa con 4 estrofas más — no respeta "four-line")*

**Unsloth / MLX — 2.05 s**
> *(Respuesta idéntica a Transformers)*

**Ollama (llama.cpp Metal) — 0.88 s**
> In the night sky, a thousand stars twinkle brightly
> As I gaze upon them with wonder and awe
> Their light shines like diamonds on a darkened ground
> A symphony of colors that dance in the air
>
> Each one a unique masterpiece, each one a sight to behold...
> *(Continúa con 3 estrofas más — tampoco respeta "four-line")*

**vLLM (CPU)** — pendiente

**Observación:** Ningún framework respeta el límite de cuatro líneas. Sin embargo, Ollama produce la respuesta poéticamente más rica (vocabulario más variado, imágenes más evocadoras). Transformers y MLX generan el mismo poema genérico con anáforas repetitivas ("A sight to behold, a sight to see"). La restricción cuantitativa ("four-line") es difícil de respetar para modelos pequeños sin RLHF específico.

---

#### Prompt 5 — Nombres para cafetería *(brainstorming — Helpfulness)*
> *"Suggest 5 creative and memorable names for a small independent coffee shop, and explain the concept behind each name."*

**Transformers (PyTorch MPS) — 3.85 s**
> 1. "The Coffee Shack" — casual and cozy vibe
> 2. "The Coffee Corner" — playful and whimsical
> 3. "The Coffee Cove" — romantic and evocative
> 4. "The Coffee Haven" — cozy and inviting
> 5. "The Coffee Corner" *(repetido — se queda sin tokens)*

**Unsloth / MLX — 2.36 s**
> 1. "The Coffee Shack" — connection to the outdoors
> 2. "The Coffee Bean" — playful, connection to the bean
> 3. "The Coffee Corner" — focus on quality
> 4. "The Coffee Roaster" — nod to the roasting process
> 5. "The Coffee Café" *(incompleto — se queda sin tokens)*

**Ollama (llama.cpp Metal) — 1.02 s**
> 1. The Coffee Bean — relax and unwind
> 2. The Coffee Shack — rustic feel
> 3. The Coffee House — traditional coffee house culture
> 4. The Coffee Corner — specialty drinks and pastries
> 5. The Coffee Roaster *(incompleto — se queda sin tokens)*

**vLLM (CPU)** — pendiente

**Observación:** Los tres frameworks proponen nombres similares ("Coffee Shack", "Coffee Corner", "Coffee Bean") — todos genéricos, sin creatividad real. MLX es el único que completa los 5 nombres. Los tres se quedan sin tokens antes de terminar las explicaciones. Esto confirma que `max_tokens=256` es insuficiente para este tipo de tarea de brainstorming.

---

### 6.3 Conclusiones del benchmark de inferencia

| Dimensión | Ganador | Motivo |
|-----------|:-------:|--------|
| **Velocidad (tok/s)** | Ollama | llama.cpp + Metal: 268 tok/s vs 107 (MLX) vs 67 (Transformers) |
| **Latencia por respuesta** | Ollama | 0.88 s vs 1.94 s (MLX) vs 2.53 s (Transformers) |
| **Memoria** | Unsloth/MLX | 2.28 GB vs 3.53 GB (Transformers) |
| **Calidad de respuestas** | Ollama | Mejor en P1 (Capital) y P2 (Huevos) |
| **Consistencia** | Transformers | Misma salida predecible, mejor integración con HF ecosystem |
| **Producción (NVIDIA)** | vLLM | PagedAttention + continuous batching → >2000 tok/s en A100 |
| **Facilidad de despliegue** | Ollama | `ollama serve` + `ollama pull` — sin código, sin config |

**Conclusión:** Para Apple Silicon, **Ollama es la mejor opción de producción**: 4× más rápido que Transformers HF, fácil de desplegar, y sin necesidad de código Python. **MLX/Unsloth** es ideal cuando se necesita integración con Python y personalización del pipeline. **Transformers** sigue siendo el estándar para investigación y fine-tuning. **vLLM** es la opción imbatible en servidores Linux con GPU NVIDIA, donde su PagedAttention y continuous batching pueden servir centenares de requests simultáneos.
