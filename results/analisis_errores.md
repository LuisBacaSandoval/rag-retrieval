# Análisis de errores por tipo de consulta (Hito 6)

Datos: `results/comparacion.csv` y `results/comparacion_por_consulta.csv`, generados
por `scripts/compare_methods.py` a partir de `results/{bm25,dense,hybrid}.json`.
Las métricas por consulta se miden a nivel de fragmento (relevancia heredada del
artículo); el recall, a nivel de artículo (ver sección 2.2 del cuaderno).

## Panorama agregado

| Método | MRR | P@5 | Recall art.@5 | nDCG@10 | Léxica (MRR) | Semántica (MRR) | Difícil (MRR) |
|---|---|---|---|---|---|---|---|
| BM25 | 0.794 | 0.578 | 0.824 | 0.590 | **0.926** | 0.667 | 0.789 |
| Denso | 0.781 | 0.585 | **0.855** | 0.555 | 0.873 | 0.668 | 0.803 |
| **Híbrido** | **0.849** | **0.659** | 0.833 | **0.635** | 0.889 | **0.787** | **0.870** |

**Aciertos y fallos totales** (de 54 consultas):

| Método | RR = 1 (relevante en #1) | RR = 0 (fallo total) |
|---|---|---|
| BM25 | 38 | 3 |
| Denso | 37 | 2 |
| **Híbrido** | **41** | **2** |

El híbrido tiene a la vez **el mayor número de aciertos en primera posición** y
**el menor número de fallos totales** (empatado con el denso). No domina consulta
a consulta —de hecho empeora en 11 casos frente al mejor método individual y solo
mejora en 3—, pero su comportamiento agregado es el más robusto porque sube el
piso (menos ceros) y el techo (más unos) al mismo tiempo.

## Casos concretos

### 1. El denso cierra una brecha semántica de BM25 — q20 (semántica)

> *"hacer que una máquina aprenda a partir de ejemplos en lugar de reglas fijas"* → relevante: `aprendizaje_automatico`

BM25 falla del todo (RR = 0; su #1 es `traduccion_automatica`) porque la consulta
no comparte términos con el artículo. El denso lo coloca en #1 (RR = 1): los
embeddings acercan "aprender a partir de ejemplos" a "aprendizaje automático" sin
coincidencia léxica. **Es la ganancia semántica que motiva la variante densa.**

### 2. La fusión rescata lo que ninguno tenía arriba — q13, q47 (semánticas)

> q13: *"representar palabras como vectores numéricos que capturan su significado"* → `word_embedding`, `word2vec`

BM25 lo tiene en RR = 0.25 y el denso en 0.50; el híbrido lo sube a **1.0**. Ningún
método individual lo situaba en #1, pero ambos lo tenían a media altura y el
**consenso de RRF lo promueve**. Es el valor propio de la fusión: acierta donde
los dos coinciden aunque ninguno destaque.

### 3. La fusión degrada un acierto que ambos ya tenían — q08, q11 (léxicas)

> q08: *"traducción automática estadística y neuronal"* → `traduccion_automatica`

BM25 y el denso lo aciertan en #1 (RR = 1.0 cada uno), pero el híbrido cae a
**0.5**: su #1 es un fragmento de `pln` (no relevante) y el de `traduccion_automatica`
queda en #2. Mecanismo: **relevancia por artículo + fusión por fragmento**. Un
fragmento de `pln` queda 2.º–3.º en *ambas* listas y su RRF acumulado supera al
fragmento relevante que era 1.º en una lista pero bajo en la otra. **RRF premia el
acuerdo intermedio por encima del acierto especializado.**

### 4. La fusión pierde un hallazgo profundo del denso — q42 (semántica)

> *"palabras muy frecuentes que se descartan al construir el índice"* → `palabra_vacia`

BM25 no recupera `palabra_vacia` en absoluto; el denso sí, pero en #5 (RR = 0.20).
Al fusionar, el cero de BM25 **diluye** el hallazgo y el artículo queda fuera del
top-10 (RR = 0). La fusión no ayuda cuando solo un método encuentra el documento y
lo hace en una posición profunda.

### 5. El denso estorba en consultas léxicas exactas — q03, q37 (léxicas)

> q03: *"índice invertido en recuperación de información"* → `indice_invertido`

BM25 acierta en #1 (RR = 1.0); el denso cae a 0.125 porque **difumina la
coincidencia exacta** de términos técnicos, y arrastra al híbrido a 0.5. En
consultas que ya usan las palabras del documento, el aporte semántico es ruido.

### 6. Ninguno lo resuelve — q12 (semántica)

> *"cómo puntuar la pertinencia de un texto frente a las palabras buscadas"* → `bm25`

Los tres métodos fallan (RR = 0). La consulta **describe la función de BM25 sin
usar su vocabulario**: ni la coincidencia léxica ni la similitud semántica anclan
el artículo correcto (el ranking se va a `motor_busqueda`, `entropia`…). Es un
límite real —corpus pequeño y una sola formulación de la relevancia— más que un
defecto de un método concreto.

## Síntesis y recomendación

- **BM25** gana en léxicas y es imbatible en costo; falla en paráfrasis sin solape.
- **Denso** mejora el recall y rescata semánticas (caso 1), pero estorba en léxicas
  exactas (caso 5) y cuesta ~20× más por consulta.
- **Híbrido** logra el mejor MRR/nDCG global y más aciertos en #1, a cambio de dos
  modos de fallo identificados: **dilución** (caso 4) y **consenso sobre
  especialización** (caso 3), además del mayor costo.

**Recomendación:** usar el híbrido por robustez ante tráfico mixto; quedarse con
BM25 solo si el tráfico es puramente léxico o el costo es crítico. **Mejora futura:**
fusión ponderada (dar más peso a BM25 en consultas con alto solape léxico) o
agregar los fragmentos por artículo antes de fusionar, para mitigar el caso 3.
