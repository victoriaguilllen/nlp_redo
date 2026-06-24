# Prompts, Respuestas y Scores — Evaluación e Inferencia

---

## 1. Evaluación Humana (5 prompts × 4 modelos)

Escala 1–5 por dimensión: **H** = Helpfulness, **F** = Factuality, **IF** = Instruction-following.

---

### P1 — Capital de Australia *(open_qa — Factuality)*

> **Prompt:** *"What is the capital of Australia, and why is it often confused with Sydney?"*

**TinyLlama original** | H=3 F=2 IF=4 → media 3.00
> The capital of Australia is Canberra, which is often confused with Sydney because they are both located in the country's capital city, Canberra. Sydney is the largest city in Australia and is located in the state of New South Wales.

*Nombra Canberra pero acto seguido dice que Canberra es "la capital city" donde se ubica Sydney — contradicción factual leve.*

---

**Transformers LoRA v1** | H=2 F=1 IF=4 → media 2.33
> Sydney is the capital of Australia, and is often confused with the capital of Australia, Canberra. Sydney is located in the state of New South Wales, while Canberra is located in the state of Australian Capital Territory.

*Error factual directo: dice que Sydney es la capital. Regresión respecto al baseline.*

---

**Unsloth v1** | H=2 F=1 IF=4 → media 2.33
> Sydney is the capital of Australia, and is often confused with the capital of Australia, Canberra.

*Mismo error que TF v1 — confusión Sydney/Canberra — pero en una sola frase.*

---

**Unsloth v2** | H=3 F=3 IF=4 → media 3.33
> Canberra is the capital of Australia, and is often confused with Sydney because they are both located in the same country.

*Canberra correcta, pero la razón de la confusión ("same country") es demasiado genérica.*

---

### P2 — Huevos revueltos *(general_qa — Helpfulness)*

> **Prompt:** *"Give me clear step-by-step instructions to make perfect scrambled eggs."*

**TinyLlama original** | H=1 F=2 IF=1 → media 1.33
> 1. Heat a non-stick skillet over medium heat.
> 2. Add a tablespoon of olive oil to the skillet.
> 3. Once the oil is hot, add the chopped onions and sauté for 2-3 minutes...
> 6. Add the diced tomatoes and their juices to the skillet. Stir well to combine.
> 9. Add the cooked quinoa to the skillet and stir well to combine.
> 10. Cook the quinoa for 5-7 minutes until it has absorbed all the tomato mixture...

*Genera una receta de quinoa con tomates. No menciona huevos en ningún momento.*

---

**Transformers LoRA v1** | H=3 F=3 IF=3 → media 3.00
> 1. Heat a non-stick pan over medium heat.
> 2. Add 1 tablespoon of butter to the pan.
> 3. Once the butter is melted, add 1 tablespoon of flour to the pan.
> 4. Cook the flour for 1 minute.
> 5. Add 1 cup of milk to the pan.
> 6. Cook the milk for 2 minutes.
> 7. Add 1 egg to the pan.
> 8. Cook the egg for 2 minutes.
> ...
> 12. Use a spatula to scramble the eggs.
> 13. Serve the scrambled eggs.

*Menciona huevos, pero el proceso (flour + milk antes de añadir el huevo) describe una bechamel, no huevos revueltos.*

---

**Unsloth v1** | H=4 F=4 IF=4 → media 4.00
> 1. Heat a non-stick pan over medium heat.
> 2. Add a tablespoon of butter to the pan.
> 3. Once the butter is melted, add the scrambled eggs to the pan.
> 4. Cook the eggs for 2-3 minutes.
> 5. Once the eggs are cooked, remove the pan from the heat.
> 6. Add salt and pepper to taste.
> 7. Serve the scrambled eggs hot.

*Mejor respuesta del grupo para este prompt: pasos claros, ingredientes correctos, concisa.*

---

**Unsloth v2** | H=4 F=4 IF=4 → media 4.00
> 1. Heat a non-stick pan over medium heat.
> 2. Add a tablespoon of butter to the pan.
> 3. Once the butter has melted, add your scrambled eggs to the pan.
> 4. Use a spatula to gently stir the eggs until they are fully cooked.
> 5. Once the eggs are fully cooked, remove the pan from the heat and let it sit for a few minutes.
> 6. Use a spatula to scoop the scrambled eggs into a bowl.
> 7. Sprinkle some salt and pepper on top of the scrambled eggs.

*Muy similar a v1, añade el detalle del reposo y usa la espátula dos veces. Igualmente correcto.*

---

### P3 — Clasificación de sentimiento *(classification — Instruction-following)*

> **Prompt:** *"Classify each of the following words as either positive or negative sentiment: joy, anger, love, fear, hope, sadness, peace, rage. Present the result as two lists."*

**TinyLlama original** | H=4 F=5 IF=2 → media 3.67
> - joy: positive sentiment
> - anger: negative sentiment
> - love: positive sentiment
> - fear: negative sentiment
> - hope: positive sentiment
> - sadness: negative sentiment
> - peace: positive sentiment
> - rage: negative sentiment

*Clasificación correcta pero en formato lista-por-ítem, no dos listas separadas como pide la instrucción.*

---

**Transformers LoRA v1** | H=4 F=5 IF=4 → media 4.33
> Positive: joy, love, hope, peace
> Negative: anger, fear, rage, sadness
> Positive: joy, love, hope, peace
> Negative: anger, fear, rage, sadness

*Formato correcto (dos listas) y clasificación perfecta. Pero repite el output completo dos veces — overfitting estructural.*

---

**Unsloth v1** | H=3 F=3 IF=3 → media 3.00
> Positive: joy, love, hope
> Negative: anger, fear, rage
> Positive: joy, love, peace
> Negative: anger, fear, rage

*Dos pares de listas inconsistentes: "sadness" y "peace" aparecen en posiciones distintas; "sadness" desaparece en la segunda iteración.*

---

**Unsloth v2** | H=5 F=5 IF=5 → media 5.00
> Positive: joy, love, hope, peace
> Negative: anger, fear, rage, sadness

*Respuesta perfecta: dos listas, todas las palabras clasificadas correctamente, sin repetición.*

---

### P4 — Poema de 4 líneas *(creative_writing — Helpfulness + Instruction-following)*

> **Prompt:** *"Write a four-line rhyming poem about the night sky and the stars."*

**TinyLlama original** | H=3 F=4 IF=1 → media 2.67
> The night sky is a canvas painted with stars,
> A symphony of light and shade,
> A celestial canvas, a masterpiece,
> A sight to behold, a sight to see.
>
> The Milky Way, a river of light,
> A pathway to the infinite,
> *(continúa con 5 estrofas más...)*

*No respeta "four-line". Genera 5 estrofas con anáforas repetitivas ("A sight to behold, a sight to see").*

---

**Transformers LoRA v1** | H=3 F=4 IF=4 → media 3.67
> The night sky is a canvas of stars,
> A canvas of light and shade,
> A canvas of the universe,
> A canvas of the night.

*Exactamente 4 líneas — respeta la instrucción. Pero "A canvas" se repite en cada línea; poco creativo.*

---

**Unsloth v1** | H=1 F=2 IF=1 → media 1.33
> The night sky is a vast and vast
> A vast and vast and vast
> A vast and vast and vast
> A vast and vast and vast
>
> The stars are shining bright
> A bright and bright and bright
> *(bucle de repetición durante ~200 tokens)*

*Mode collapse severo: entra en un bucle de repetición que ocupa todo el límite de tokens.*

---

**Unsloth v2** | H=2 F=4 IF=3 → media 3.00
> The night sky is a sight to behold
> The stars twinkle like diamonds in the night
> The Milky Way is a sight to behold
> The stars are like diamonds in the night sky

*4 líneas correctas. Factualmente bien. Pero "a sight to behold" se repite (L1 y L3), y "diamonds in the night" casi se repite (L2 y L4). Creatividad limitada.*

---

### P5 — Nombres para cafetería *(brainstorming — Helpfulness)*

> **Prompt:** *"Suggest 5 creative and memorable names for a small independent coffee shop, and explain the concept behind each name."*

**TinyLlama original** | H=3 F=4 IF=3 → media 3.33
> 1. "The Coffee Shack" — captures the casual and cozy vibe
> 2. "The Coffee Corner" — playful and whimsical, cozy atmosphere
> 3. "The Coffee Cove" — romantic and evocative, secluded spot
> 4. "The Coffee Haven" — cozy and inviting atmosphere
> 5. "The Coffee Corner" *(repetido — se agota el límite de tokens)*

*5 nombres con explicación. Pero el nombre 5 repite al 2 por límite de tokens. Todos siguen el patrón "The Coffee X".*

---

**Transformers LoRA v1** | H=1 F=3 IF=3 → media 2.33
> 1. The Coffee Shop — nod to the coffee shop's location in a small town.
> 2. The Coffee Shop — nod to the coffee shop's location in a small town.
> 3. The Coffee Shop — nod to the coffee shop's location in a small town.
> 4. The Coffee Shop — nod to the coffee shop's location in a small town.
> 5. The Coffee Shop — nod to the coffee shop's location in a small town.

*Mode collapse completo: 5 veces el mismo nombre con la misma frase.*

---

**Unsloth v1** | H=2 F=3 IF=3 → media 2.67
> 1. The Coffee Shop — cozy and intimate, playful and whimsical
> 2. The Coffee Shop — more upscale, sophisticated and elegant
> 3. The Coffee Shop — more casual, fun and playful
> 4. The Coffee Shop — more laid back, relaxed and comfortable
> 5. The Coffee Shop — more modern and trendy, modern and stylish

*5 conceptos distintos pero todos con el mismo nombre. La variación está en el concepto, no en el nombre.*

---

**Unsloth v2** | H=1 F=3 IF=1 → media 1.67
> 1. The Coffee Shop
> 2. The Coffee Shack
> 3. The Coffee House
> 4. The Coffee Corner
> 5. The Coffee Cove

*5 nombres distintos, pero sin ninguna explicación del concepto — omite la mitad de la instrucción.*

---

## 2. Inferencia — Modelo Base (sin adaptador LoRA)

Configuración: temperature=0.0, max_tokens=256, 5 prompts × 3 trials, 1 warm-up descartado.
Frameworks comparados sobre el mismo modelo base TinyLlama sin fine-tuning.

---

### P1 — Capital de Australia

> *"What is the capital of Australia, and why is it often confused with Sydney?"*

**Transformers (PyTorch MPS) — 1.05 s**
> The capital of Australia is Canberra, which is often confused with Sydney because they are both located in the country's capital city, Canberra. Sydney is the largest city in Australia and is located in the state of New South Wales.

**Unsloth / MLX (base) — 0.54 s**
> The capital of Australia is Canberra, which is often confused with Sydney because they are both located in the country's capital city of Canberra. Sydney is the largest city in Australia and is located in the state of New South Wales.

**Ollama (llama.cpp Metal Q4) — 1.06 s**
> The capital of Australia is Canberra, also known as the nation's capital. The name "Sydney" is commonly used to refer to the city of Sydney in New South Wales, Australia. While Sydney is technically the capital of New South Wales, it is often referred to as "Sydney" or "the city" by locals and tourists alike. This confusion is due to the fact that Canberra and Sydney are located on opposite sides of the country, with Canberra being the capital of Australia's federal government while Sydney is the capital of New South Wales.

**Observación:** Transformers y MLX generan respuestas casi idénticas (mismo modelo, mismo greedy decoding). Ollama produce una respuesta más elaborada y factualmente más correcta: distingue Canberra como capital federal de Sydney como capital de NSW, y explica la confusión geográfica.

---

### P2 — Huevos revueltos

> *"Give me clear step-by-step instructions to make perfect scrambled eggs."*

**Transformers (PyTorch MPS) — 4.18 s**
> 1. Heat a non-stick skillet over medium heat.
> 2. Add a tablespoon of olive oil to the skillet.
> 3. Once the oil is hot, add the chopped onions and sauté for 2-3 minutes...
> 9. Add the cooked quinoa to the skillet and stir well to combine.
> *(Genera receta de quinoa con tomates — no menciona huevos)*

**Unsloth / MLX (base) — 2.35 s**
> *(Respuesta casi idéntica a Transformers — quinoa y tomates en lugar de huevos)*

**Ollama (llama.cpp Metal Q4) — 1.02 s**
> Certainly! Here's a clear and step-by-step guide to making perfect scrambled eggs:
>
> Ingredients: 2 large eggs, salt and pepper, 1 tablespoon butter or oil.
>
> 1. Crack the eggs into a bowl and whisk them with a fork until fully combined.
> 2. Season the eggs with salt and pepper, to taste.
> 3. Heat a non-stick skillet over medium heat. Once it's hot, add butter.
> 4. Use a spatula to scramble the eggs in the pan...

**Observación:** Transformers y MLX fallan — generan una receta de quinoa con tomates, probable interferencia del training set del modelo base. Ollama es el único que responde correctamente, con ingredientes reales y en menos de 1 segundo.

---

### P3 — Clasificación de sentimiento

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

**Unsloth / MLX (base) — 2.36 s**
> 1. Joy — Positive: "I am happy today" / Negative: "I am sad today"
> 2. Anger — Positive: "I am angry at the world" / Negative: "I am sad because of the world"
> *(Genera frases de ejemplo para cada palabra en lugar de clasificarlas)*

**Ollama (llama.cpp Metal Q4) — 1.02 s**
> To classify each of the following words... we can use the Sentiment Analysis API provided by IBM Watson Studio. Here's how to do it: 1. Log in to your IBM Watson Studio account and navigate to the "Sentiment Analysis" service...
> *(Respuesta completamente fuera de contexto — explica cómo usar IBM Watson)*

**Observación:** Paradoja notable — los tres frameworks interpretan la tarea de forma distinta a pesar del mismo modelo y temperatura 0. Transformers clasifica correctamente pero en formato incorrecto (no dos listas). MLX interpreta como tarea de generación de frases. Ollama responde como si fuera una tarea de API programming. Ilustra cómo las diferencias en tokenización y formato de contexto producen salidas divergentes incluso en greedy decoding.

---

### P4 — Poema de 4 líneas

> *"Write a four-line rhyming poem about the night sky and the stars."*

**Transformers (PyTorch MPS) — 3.24 s**
> The night sky is a canvas painted with stars,
> A symphony of light and shade,
> A celestial canvas, a masterpiece,
> A sight to behold, a sight to see.
>
> *(Continúa con 4 estrofas más — no respeta "four-line")*

**Unsloth / MLX (base) — 2.05 s**
> *(Respuesta idéntica a Transformers)*

**Ollama (llama.cpp Metal Q4) — 0.88 s**
> In the night sky, a thousand stars twinkle brightly
> As I gaze upon them with wonder and awe
> Their light shines like diamonds on a darkened ground
> A symphony of colors that dance in the air
>
> *(Continúa con 3 estrofas más — tampoco respeta "four-line")*

**Observación:** Ningún framework respeta el límite de cuatro líneas. Ollama produce la respuesta poéticamente más rica (vocabulario más variado, imágenes más evocadoras). Transformers y MLX generan el mismo poema genérico. La restricción cuantitativa es difícil de respetar para modelos pequeños sin instrucción de parada explícita.

---

### P5 — Nombres para cafetería

> *"Suggest 5 creative and memorable names for a small independent coffee shop, and explain the concept behind each name."*

**Transformers (PyTorch MPS) — 3.85 s**
> 1. "The Coffee Shack" — casual and cozy vibe, referencing a "shack" that serves coffee
> 2. "The Coffee Corner" — playful and whimsical, cozy intimate atmosphere
> 3. "The Coffee Cove" — romantic and evocative, secluded peaceful spot
> 4. "The Coffee Haven" — cozy and inviting atmosphere, haven or sanctuary
> 5. "The Coffee Corner" *(repetido — se agota el límite de 256 tokens)*

**Unsloth / MLX (base) — 2.36 s**
> 1. "The Coffee Shack" — connection to the outdoors
> 2. "The Coffee Bean" — playful, connection to the bean itself
> 3. "The Coffee Corner" — focus on quality coffee and customer service
> 4. "The Coffee Roaster" — nod to the coffee roasting process
> 5. "The Coffee Café" *(incompleto — se agota el límite de tokens)*

**Ollama (llama.cpp Metal Q4) — 1.02 s**
> 1. The Coffee Bean — relax and unwind, friendly and welcoming vibe
> 2. The Coffee Shack — cozy and intimate with rustic feel
> 3. The Coffee House — nod to traditional coffee house culture
> 4. The Coffee Corner — variety of specialty drinks and pastries
> 5. The Coffee Roaster *(incompleto — se agota el límite de tokens)*

**Observación:** Los tres frameworks siguen el patrón "The Coffee X" sin excepción — probable sesgo del pretraining. Todos se quedan sin tokens antes de completar el nombre 5. Ollama produce las descripciones más detalladas en el menor tiempo.
