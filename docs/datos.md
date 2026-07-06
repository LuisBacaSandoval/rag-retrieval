# Datos: corpus y consultas

## Origen y licencia

El corpus se compone de 40 artículos de **Wikipedia en español** sobre
procesamiento de lenguaje natural, recuperación de información y aprendizaje
automático. El texto es contenido de Wikipedia, disponible bajo licencia
**Creative Commons Atribución-CompartirIgual 4.0 (CC BY-SA 4.0)**. Cada archivo
en `data/corpus/raw/` conserva en su cabecera el título, la URL de origen y la
fecha de descarga.

La descarga es reproducible con `scripts/fetch_sources.py` (usa la API de
Wikipedia y solo la biblioteca estándar). Los archivos `raw/` se versionan en el
repositorio, de modo que el resto del pipeline funciona sin conexión.

### Artículos del corpus

| source_id | Artículo | Fragmentos | URL |
|---|---|---:|---|
| red_neuronal | Red neuronal artificial | 82 | <https://es.wikipedia.org/wiki/Red_neuronal_artificial> |
| analisis_grupos | Análisis de grupos | 52 | <https://es.wikipedia.org/wiki/An%C3%A1lisis_de_grupos> |
| traduccion_automatica | Traducción automática | 37 | <https://es.wikipedia.org/wiki/Traducci%C3%B3n_autom%C3%A1tica> |
| entropia | Entropía (información) | 28 | <https://es.wikipedia.org/wiki/Entrop%C3%ADa_(informaci%C3%B3n)> |
| aprendizaje_automatico | Aprendizaje automático | 25 | <https://es.wikipedia.org/wiki/Aprendizaje_autom%C3%A1tico> |
| motor_busqueda | Motor de búsqueda | 24 | <https://es.wikipedia.org/wiki/Motor_de_b%C3%BAsqueda> |
| retropropagacion | Propagación hacia atrás | 24 | <https://es.wikipedia.org/wiki/Retropropagaci%C3%B3n> |
| svm | Máquina de vectores de soporte | 22 | <https://es.wikipedia.org/wiki/M%C3%A1quina_de_vectores_de_soporte> |
| analisis_sentimiento | Análisis de sentimiento | 20 | <https://es.wikipedia.org/wiki/An%C3%A1lisis_de_sentimiento> |
| naive_bayes | Clasificador bayesiano ingenuo | 19 | <https://es.wikipedia.org/wiki/Clasificador_bayesiano_ingenuo> |
| recuperacion_informacion | Búsqueda y recuperación de información | 19 | <https://es.wikipedia.org/wiki/B%C3%BAsqueda_y_recuperaci%C3%B3n_de_informaci%C3%B3n> |
| word2vec | Word2vec | 18 | <https://es.wikipedia.org/wiki/Word2vec> |
| mineria_textos | Minería de textos | 15 | <https://es.wikipedia.org/wiki/Miner%C3%ADa_de_textos> |
| linguistica_computacional | Lingüística computacional | 14 | <https://es.wikipedia.org/wiki/Ling%C3%BC%C3%ADstica_computacional> |
| ngrama | N-grama | 14 | <https://es.wikipedia.org/wiki/N-grama> |
| pln | Procesamiento de lenguajes naturales | 14 | <https://es.wikipedia.org/wiki/Procesamiento_de_lenguajes_naturales> |
| hmm | Modelo oculto de Márkov | 13 | <https://es.wikipedia.org/wiki/Modelo_oculto_de_M%C3%A1rkov> |
| ner | Reconocimiento de entidades nombradas | 13 | <https://es.wikipedia.org/wiki/Reconocimiento_de_entidades_nombradas> |
| busqueda_semantica | Búsqueda semántica | 11 | <https://es.wikipedia.org/wiki/B%C3%BAsqueda_sem%C3%A1ntica> |
| perceptron_multicapa | Perceptrón multicapa | 11 | <https://es.wikipedia.org/wiki/Perceptr%C3%B3n_multicapa> |
| reconocimiento_habla | Reconocimiento del habla | 11 | <https://es.wikipedia.org/wiki/Reconocimiento_del_habla> |
| modelo_lenguaje | Modelo de lenguaje | 10 | <https://es.wikipedia.org/wiki/Modelaci%C3%B3n_del_lenguaje> |
| aprendizaje_profundo | Aprendizaje profundo | 9 | <https://es.wikipedia.org/wiki/Aprendizaje_profundo> |
| atencion | Atención (aprendizaje automático) | 9 | <https://es.wikipedia.org/wiki/Atenci%C3%B3n_(aprendizaje_autom%C3%A1tico)> |
| etiquetado_gramatical | Etiquetado gramatical | 8 | <https://es.wikipedia.org/wiki/Etiquetado_gramatical> |
| modelo_espacio_vectorial | Modelo de espacio vectorial | 8 | <https://es.wikipedia.org/wiki/Modelo_de_espacio_vectorial> |
| precision_exhaustividad | Precisión y exhaustividad | 8 | <https://es.wikipedia.org/wiki/Precisi%C3%B3n_y_exhaustividad> |
| transformador | Transformador (modelo de aprendizaje automático) | 8 | <https://es.wikipedia.org/wiki/Transformador_(modelo_de_aprendizaje_autom%C3%A1tico)> |
| tfidf | Tf-idf | 7 | <https://es.wikipedia.org/wiki/Tf-idf> |
| bert | BERT (modelo de lenguaje) | 6 | <https://es.wikipedia.org/wiki/BERT_(modelo_de_lenguaje)> |
| red_neuronal_recurrente | Red neuronal recurrente | 6 | <https://es.wikipedia.org/wiki/Red_neuronal_recurrente> |
| similitud_coseno | Similitud coseno | 5 | <https://es.wikipedia.org/wiki/Similitud_coseno> |
| sistema_recomendacion | Sistema de recomendación | 5 | <https://es.wikipedia.org/wiki/Sistema_de_recomendaci%C3%B3n> |
| bm25 | Okapi BM25 | 4 | <https://es.wikipedia.org/wiki/Okapi_BM25> |
| word_embedding | Word embedding | 4 | <https://es.wikipedia.org/wiki/Word_embedding> |
| indice_invertido | Índice invertido | 3 | <https://es.wikipedia.org/wiki/%C3%8Dndice_invertido> |
| lematizacion | Lematización | 3 | <https://es.wikipedia.org/wiki/Lematizaci%C3%B3n> |
| ley_zipf | Ley de Zipf | 2 | <https://es.wikipedia.org/wiki/Ley_de_Zipf> |
| palabra_vacia | Palabra vacía | 2 | <https://es.wikipedia.org/wiki/Palabra_vac%C3%ADa> |
| stemming | Stemming | 2 | <https://es.wikipedia.org/wiki/Stemming> |

## Del texto al corpus (chunking)

`scripts/prepare_corpus.py` transforma los artículos en fragmentos:

1. Quita la cabecera de procedencia.
2. Corta el cuerpo en la primera sección final no-prosa (*Véase también*,
   *Referencias*, *Bibliografía*, *Enlaces externos*), que no aporta contenido
   recuperable.
3. Normaliza espacios y descarta líneas triviales.
4. Trocea por ventanas de caracteres con solape.

Parámetros de chunking:

| Parámetro | Valor |
|---|---|
| Tamaño de fragmento | 800 caracteres |
| Solape | 120 caracteres |
| Mínimo para no fusionar la cola | 200 caracteres |

Resultado: **595 fragmentos** a partir de 40 artículos, con una longitud media
de ~784 caracteres por fragmento.

### Formato

`data/corpus/corpus.jsonl`, una línea JSON por fragmento:

```json
{"doc_id": "bm25__c03", "source_id": "bm25", "title": "Okapi BM25", "text": "..."}
```

- `doc_id`: identificador único del fragmento (unidad que indexan y devuelven
  los recuperadores).
- `source_id`: artículo de origen; se usa para mapear la relevancia.
- `title`, `text`.

## Consultas de evaluación

`data/queries/queries.jsonl`, una línea JSON por consulta:

```json
{"query_id": "q01", "query": "...", "type": "lexica", "relevant_source_ids": ["bm25"]}
```

Son **54 consultas**, repartidas en tres tipos (18 de cada uno) para poder
analizar en qué situaciones gana cada método de recuperación:

| Tipo | Descripción | Nº |
|---|---|---:|
| `lexica` | Usan las mismas palabras que el artículo relevante (favorecen a BM25). | 18 |
| `semantica` | Parafraseadas, con sinónimos y sin coincidencia literal (favorecen al recuperador denso). | 18 |
| `dificil` | Ambiguas, multi-artículo o con términos engañosos. | 18 |

### Criterio de relevancia

La relevancia se juzga **a nivel de artículo** (`relevant_source_ids`), no de
fragmento. Una consulta se considera respondida por un artículo si ese artículo
trata el tema preguntado. En la evaluación (Hito 3), cada fragmento recuperado
se mapea a su `source_id` y se compara contra los artículos relevantes.

Motivo del nivel de artículo: etiquetar 595 fragmentos × 54 consultas a mano es
inviable y, además, se rompería al cambiar los parámetros de chunking. Juzgar
qué artículo responde es manejable y robusto frente al re-troceado.

## Limitaciones

- **Anotador único:** los juicios de relevancia los hizo una sola persona; no hay
  acuerdo entre anotadores.
- **Tamaño moderado:** 40 artículos y 595 fragmentos; suficiente para comparar
  métodos, pero no un corpus a gran escala.
- **Desbalance de fuentes:** algunos artículos son mucho más largos (p. ej.
  *Red neuronal artificial*, 82 fragmentos) que otros (*Stemming*, 2).
- **Relevancia binaria y a nivel de artículo:** no distingue fragmentos más o
  menos útiles dentro de un mismo artículo relevante.
- **Dominio único (PLN/IR/ML):** los resultados no se extrapolan directamente a
  otros dominios.
