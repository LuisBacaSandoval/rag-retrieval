# Limitaciones

Qué no demuestra el proyecto, organizado por origen: datos, métodos,
evaluación, alcance y despliegue.

## De los datos

- **Tamaño moderado.** 40 artículos y 595 fragmentos; suficiente para comparar
  métodos, pero no un corpus a gran escala.
- **Desbalance de fuentes.** Algunos artículos son mucho más largos (p. ej.
  *Red neuronal artificial*, 82 fragmentos) que otros (*Stemming*, 2), lo que da
  a los grandes más oportunidades de aparecer en el top-k.
- **Relevancia binaria y a nivel de artículo.** No distingue fragmentos más o
  menos útiles dentro de un mismo artículo relevante (su efecto sobre las
  métricas se detalla en "De la evaluación").

## De los métodos

- **Un solo modelo de embeddings.** Toda la conclusión sobre "recuperación
  densa" se apoya en `paraphrase-multilingual-MiniLM-L12-v2`. Un modelo mayor o
  especializado en español podría mover la comparación (sobre todo las consultas
  semánticas, donde el denso apenas empató con BM25: MRR 0.668 vs 0.667). No se
  hizo ablación de modelos.
- **Sin reranking neural.** La variante híbrida es *fusión de rankings* (RRF),
  no reranking aprendido. Un cross-encoder sobre el top-20 quedó fuera por costo
  y tiempo; es la extensión natural y probablemente atacaría los dos modos de
  fallo de la fusión (dilución y consenso sobre especialización).
- **Hiperparámetros sin ajustar.** BM25 usa los valores por defecto
  ($k_1 = 1.5$, $b = 0.75$) y RRF los del paper original ($c = 60$,
  pool = 50). No hubo búsqueda de hiperparámetros; los resultados comparan
  configuraciones "de fábrica", no configuraciones óptimas por método.
- **Escala no probada.** FAISS `IndexFlatIP` es búsqueda exacta por fuerza
  bruta: correcto y suficiente para 595 vectores, pero las latencias medidas no
  se extrapolan a corpus de millones de fragmentos, donde haría falta un índice
  aproximado (IVF/HNSW) con su propio compromiso recall/velocidad.

## De la evaluación

- **54 consultas, sin significancia estadística.** Las diferencias entre métodos
  (p. ej. MRR 0.849 vs 0.794) se reportan sin intervalo de confianza ni test
  pareado; con 18 consultas por tipo, una consulta mueve un MRR por tipo en
  ~0.05. Las conclusiones son direccionales, no inferencia estadística.
- **Círculo autor–anotador.** Las consultas las escribió la misma persona que
  etiquetó la relevancia y conoce el corpus. El etiquetado por *tipo* de
  consulta (léxica/semántica/difícil) también es de autor: una "semántica" se
  redactó deliberadamente para no compartir vocabulario, lo que puede exagerar
  el contraste entre métodos frente a consultas de usuarios reales.
- **Latencia y memoria dependen del entorno.** Se miden en CPU dentro del
  contenedor, con calentamiento y mediana (p50), pero siguen siendo de una
  máquina concreta. La relación entre métodos (~20× BM25 vs denso) es más fiable
  que los valores absolutos.
- **Relevancia heredada del artículo.** P@k y nDCG a nivel de fragmento cuentan
  como acierto *cualquier* fragmento de un artículo relevante, aunque ese
  fragmento concreto no responda la consulta. Este mismo mapeo explica parte del
  modo de fallo "consenso sobre especialización" del híbrido (q08/q11).

## Del alcance

- **Solo la etapa de recuperación.** El proyecto responde qué método recupera
  mejor, no si un RAG completo respondería mejor: no hay generación, ni medición
  de fidelidad al contexto, ni de citación (eso sería el Proyecto 6). La
  hipótesis implícita —mejor recuperación ⇒ mejor respuesta— queda sin verificar
  aquí.
- **Dominio y lengua únicos.** Corpus de PLN/IR/ML en español de Wikipedia; la
  ventaja del híbrido podría cambiar en dominios con vocabulario cerrado (legal,
  médico) donde lo léxico pesa más, o en corpus multilingües.
- **Caso sin resolver.** q12 (*"cómo puntuar la pertinencia de un texto frente a
  las palabras buscadas"* → `bm25`) falla en los tres métodos: con 4 fragmentos
  el artículo objetivo apenas ofrece anclas, y ninguna representación conecta la
  descripción funcional con el nombre propio del algoritmo. Marca el límite de
  lo que la recuperación sola puede hacer sobre un corpus pequeño.

## Del despliegue

- **Servicio de demostración, no de producción.** Un solo proceso `uvicorn`, sin
  autenticación, sin límites de tasa y con el índice completo en memoria. El
  endpoint `/metrics` expone metadatos del índice, no telemetría de operación
  (throughput, errores, drift), que es lo que exigiría un despliegue real.
- **Sin reindexación en caliente.** Si el corpus cambia hay que reconstruir el
  índice y la imagen; el manifiesto con hash detecta la obsolescencia pero no la
  corrige automáticamente.
