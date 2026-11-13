# 🧠 PLAN MAESTRO DE INGENIERÍA — PROYECTO `INTI-GRAPH`
**Track:** QualyData — Big Data (Detección de narrativas y campañas coordinadas a escala)  
**Evento:** Hackathon Argentina Space 2026 (11, 12 y 13 de Noviembre · Salta)  
**Premio del Track:** USD 2.000 (1.º USD 1.000 | 2.º USD 600 | 3.º USD 400)  
**Repositorio Oficial del Desafío:** `https://github.com/qualydata-org/desafio-qualydata`

---

## 1. Identidad del Proyecto (Para Hacki y el Pitch)

* **Nombre del Proyecto:** **INTI-GRAPH**
* **Tagline (Subtítulo corto):** *Motor Híbrido DuckDB + Grafos Temporales para Descubrimiento de Narrativas Emergentes y Caza de Operaciones Coordinadas a Escala con Costo Cero.*
* **Pitch en 30 segundos:**
  > *"Analizar 200.000 comentarios con un LLM cuesta caro y tarda horas; hacerlo con palabras clave falla porque la gente cambia las palabras todos los días y las granjas de trolls camuflan su redacción. **INTI-GRAPH** resuelve ambos problemas en segundos en una notebook común sin GPU y con costo $0: ingesta todo en **DuckDB sobre Parquet**, reconstruye los árboles de respuestas para heredar posturas en comentarios cortos, descubre narrativas sin K fijo usando embeddings densos y detecta campañas coordinadas invisibles mediante **grafos de sincronización temporal y co-ocurrencia anormal**, separando momentos virales genuinos de operaciones armadas."*

---

## 2. Ingeniería Inversa del Evaluador (`evaluacion/puntuar.py`) y de la Verdad (`datos/practica/verdad/`)

El jurado evalúa 55 puntos de forma 100% automática corriendo `python evaluacion/puntuar.py --verdad <verdad> --entrega entrega/` y 45 puntos por arquitectura de datos (DuckDB/Parquet, costo en `costo.json`, reproducibilidad en 1 comando y claridad visual).

Al inspeccionar `evaluacion/puntuar.py` y los archivos de `datos/practica/verdad/` (`resumen.json`, `narrativas.csv`, `campanas.csv`), descubrimos **exactamente cómo están construidos los datos y cómo maximizar cada métrica**:

### 2.1. Radiografía Exacta del Dataset (`resumen.json` y `narrativas.csv`)
* **Volumen:** 200.000 comentarios, 15.991 cuentas, 506 publicaciones a lo largo de 14 días, 8 fuentes (`fuentes.csv`).
* **Ruido (`narrativa == "ninguna"`):** Exactamente **37.710 comentarios (18.85%)** son ruido (reacciones sueltas, menciones, saludos, fuera de tema).
* **Estructura de las Narrativas (en práctica son 13 narrativas divididas en 3 grandes temas):**
  * Tema `litio`: `L1`, `L2`, `L3`, `L4` (presentes desde el día 1) + `L5` (**emergente**, nace el día 7 en `p00212`: *"Hubo un derrame en el salar y lo están ocultando"*). Además, `L1` **cambia de redacción desde el día 8**.
  * Tema `colectivo`: `B1`, `B2`, `B3` (desde el día 1) + `B4` (**emergente**, nace el día 10 en `p00318`: *"Van a eliminar líneas de colectivo de los barrios"*).
  * Tema `hospital`: `H1`, `H2`, `H4` (desde el día 1) + `H3` (**emergente**, nace el día 6 a la madrugada `01:12:04-03:00` en `p00177`: *"El hospital se inaugura en campaña, solo para la foto"*).
* **Fórmula del Puntaje de Narrativas (30 puntos):**
  $$\text{Puntos Narrativas} = 30 \times \left(0.50 \cdot \text{BCubed\_F1} + 0.15 \cdot \text{Ruido\_F1} + 0.20 \cdot \text{Postura\_MacroF1} + 0.15 \cdot \text{Origen\_Score}\right)$$
  * **Detalle clave de `Origen_Score` en `puntuar.py`:** Solo evalúa las narrativas con `emergente == "si"`. Empareja cada narrativa emergente real con nuestro cluster predicho que tenga mayor cantidad de comentarios en común (`overlap`), y mide la diferencia en horas entre nuestra `fecha_origen` en `narrativas.csv` y la real:
    * Si el error es $\le 3\text{ horas}$ → **Puntaje = 1.0 (100%)**.
    * Si el error está entre 3 y 48 horas → cae linealmente hasta 0.
    * **Truco ganador para `fecha_origen`:** En `linea_base.py` usaban `quantile(0.01)` de las fechas del cluster, lo cual falla si el cluster tiene algún falso positivo de días anteriores. En cambio, una narrativa emergente nace de golpe en una publicación origen (`publicacion_origen`). Si detectamos la **primera hora/publicación donde el cluster tiene su primer salto brusco de actividad (changepoint: ej. $\ge 5$ comentarios en una misma publicación o ventana de 2 horas)** y tomamos el **timestamp del primer comentario de esa ráfaga inicial**, ¡clavamos la `fecha_origen` con un error menor a 15 minutos (**Score = 1.000**)!

### 2.2. Radiografía Exacta de las Campañas Coordinadas (`campanas.csv`)
En `datos/practica/verdad/campanas.csv` hay 3 operaciones coordinadas (115 cuentas en total sobre 15.991, con 645 comentarios de campaña):
1. **Campaña `C1` (60 cuentas, impulsa `L2` con postura `apoya`):**
   * *"Cuentas creadas pocos días antes del período, textos casi idénticos y hashtag propio, ráfagas diurnas de 12 minutos."*
2. **Campaña `C2` (30 cuentas, impulsa `L4` con postura `apoya`):**
   * *"Cuentas antiguas con actividad de cobertura, textos variados (sin plantilla propia), sincronizadas en las mismas publicaciones y respondiéndose entre sí."*
   * **¡OJO ACÁ!** `linea_base.py` falla completamente en `C2` porque `C2` **no repite el mismo texto** ni usa cuentas nuevas. Lo que delata a `C2` es:
     1. Aparecen juntas en el mismo subconjunto de `publicacion_id` dentro de ventanas cortas de tiempo.
     2. **Se responden entre sí (`respuesta_a`) con postura de apoyo mutuo** mucho más seguido que el azar.
     3. A pesar de tener "actividad de cobertura" en otros temas, convergen sincronizadamente sobre la misma narrativa (`L4`).
3. **Campaña `C3` (25 cuentas, impulsa `H3` con postura `apoya`):**
   * *"Cuentas nuevas que originan la narrativa en ráfagas nocturnas (01 a 05 h) con textos casi idénticos."*
* **Fórmula del Puntaje de Coordinación (25 puntos):**
  $$\text{Puntos Coordinación} = 25 \times \left(0.70 \cdot \text{F1\_Binario\_Cuentas} + 0.30 \cdot \text{BCubed\_F1\_Grupos}\right)$$
  * Como son ~115 cuentas coordinadas entre ~16.000, **la precisión es vital**: un exceso de falsos positivos destruye el `F1_Binario_Cuentas`.

---

## 3. Arquitectura del Pipeline Ganador `INTI-GRAPH`

### Estructura del Repositorio a Construir

```text
inti-graph/
├── datos/
│   └── practica/                   # (Copiado del repo oficial desafio-qualydata)
├── evaluacion/
│   └── puntuar.py                  # (Copiado intacto del repo oficial desafio-qualydata)
├── ejemplos/
│   └── linea_base.py               # (Referencia original)
├── src/
│   ├── __init__.py
│   ├── storage_duckdb.py           # Ingesta CSV.gz -> DuckDB + Parquet + Feature Engineering SQL ultra-rápido
│   ├── text_features.py            # Limpieza, extracción de hashtags, marcadores de postura, TF-IDF + SVD / Embeddings locales
│   ├── narrative_engine.py         # Clustering jerárquico/densidad + Clasificador de Ruido + Propagación en árbol `respuesta_a` + Detector exacto de `fecha_origen`
│   ├── coordination_hunter.py      # Detector Multi-Señal de Grafos (C1: ráfagas/plantillas/cuentas nuevas, C2: co-ocurrencia sincronizada + grafo `respuesta_a`, C3: ráfagas nocturnas) + Louvain/Leiden
│   └── pipeline.py                 # Ejecución end-to-end con un solo comando que genera `comentarios.csv`, `narrativas.csv`, `cuentas.csv` y `costo.json`
├── dashboard/
│   └── app.py                      # Dashboard interactivo (Streamlit o FastAPI+HTML) para el video de 3 minutos y el pitch
├── salida/
│   └── practica/                   # Salida de pruebas sobre el set de práctica
├── entrega/                        # Carpeta lista con los 4 archivos generados (`comentarios.csv`, `narrativas.csv`, `cuentas.csv`, `costo.json`)
├── requirements.txt
└── README.md
```

---

## 4. Diseño Detallado de los Módulos para Superar 50/55 Puntos Automáticos

### Módulo 1: `storage_duckdb.py` (20 puntos de Costo, Escala y Almacenamiento)
1. Conectar a **DuckDB** y convertir `fuentes.csv`, `publicaciones.csv.gz`, `comentarios.csv.gz` y `cuentas.csv.gz` a tablas columnares y archivos **Parquet** en `salida/warehouse/`.
2. Ejecutar joins vectorizados en SQL dentro de DuckDB para calcular en < 1 segundo:
   * Antigüedad de cada cuenta al inicio del dataset (`dias_antiguedad_cuenta`).
   * Hora del día en zona horaria argentina `-03:00` (`hora_ar`, detectando franja nocturna `01 <= hora_ar <= 05`).
   * Estadísticas por `publicacion_id`: cantidad total de comentarios, velocidad base de comentarios por minuto (para distinguir un post viral orgánico de una ráfaga coordinada en un post tranquilo).
   * Árbol de respuestas: join de cada comentario con su comentario padre (`respuesta_a -> comentario_id`) obteniendo `padre_cuenta_id`, `padre_texto`, `padre_ts`.

### Módulo 2: `narrative_engine.py` (30 puntos de Narrativas)
Para que el pipeline generalice el viernes 13 sobre el **set final** (donde cambiarán los temas y la cantidad de narrativas) sin sobreajustarse a mano al set de práctica:

1. **Separación deComentarios Raíz vs. Respuestas Cortas:**
   * Comentarios sin contenido semántico propio (ej. `< 4` tokens útiles tras quitar menciones/stopwords):
     * Si **NO tienen `respuesta_a`** (o son solo saludos/risas/etiquetas de amigos) → Son `ninguna` (Ruido).
     * Si **SÍ tienen `respuesta_a`** y usan expresiones de acuerdo/desacuerdo corto (*"tal cual"*, *"exacto"*, *"totalmente"*, *"así es"*, *"mentira"*, *"cualquiera"*, *"falso"*, *"nada que ver"*) → **Heredan la narrativa del comentario padre** una vez que el padre es clasificado.
2. **Detección de Ruido de Alta Precisión (`ruido_f1`):**
   * Un comentario raíz es `ninguna` si:
     * Tiene baja similitud semántica/léxica tanto con el `titular` de su `publicacion_id` como con los centroides de discusión.
     * O consiste en reacciones genéricas, insultos vacíos sin tema, spam comercial o frases fuera de tópico.
   * Entrenar un clasificador liviano generalizable (ej. `LogisticRegression` / `LightGBM` sobre features **agnósticas del tema**: longitud de tokens, similitud coseno comentario–titular de la publicación, distancia al centroide de la publicación, densidad de términos temáticos vs stopwords, si es respuesta o raíz) validado por validación cruzada por tema (Leave-One-Theme-Out: entrenar en Litio+Colectivo y validar en Hospital) para asegurar que el viernes funcione igual de bien en temas nuevos.
3. **Descubrimiento Automático del Número de Narrativas y Clustering (`bcubed_f1`):**
   * **Paso A (Macro-Temas por Publicación):** Agrupar primero las 500 publicaciones por sus `titular` + vocabulario agregado de sus comentarios usando TF-IDF + SVD (o embeddings multilingües rápidos) + Clustering Aglomerativo con umbral de distancia coseno automático (Silhouette / eigengap). Esto separa limpiamente los grandes temas (ej. Litio, Colectivo, Hospital) sin mezclar comentarios de temas distintos.
   * **Paso B (Sub-Narrativas dentro de cada Tema):** Dentro de cada macro-tema, representar los comentarios con una combinación de:
     * N-gramas de palabras y caracteres (TF-IDF sublineal reducido con TruncatedSVD a 64 dimensiones y normalizado L2, o embeddings `sentence-transformers` si está disponible rápido en CPU).
     * Aplicar clustering automático (HDBSCAN o AgglomerativeClustering / K-Means con selección automática de $K$ por Silhouette + fusión de clusters con centroides muy cercanos por similitud coseno $> 0.75$, lo cual une automáticamente una narrativa que cambia de redacción a mitad de semana como `L1`!).
4. **Clasificación de Postura (`apoya` vs `rechaza` — `postura_macro_f1`):**
   * Regla 1 (Comentarios sustantivos): Detectar marcadores de refutación, negación e ironía/desmentida (*"mentira"*, *"verso"*, *"falso"*, *"fake"*, *"no es cierto"*, *"nada que ver"*, *"trucho"*, *"desmintió"*, *"inventan"*, *"puro humo"*, *"dejen de mentir"*, *"nadie va a"*, *"no hubo"*, *"es al revés"*, *"jamás"*, etc.) + modelo lineal sobre n-gramas de función/polaridad (agnóstico a sustantivos del tema).
   * Regla 2 (Propagación por `respuesta_a` en cascada por orden cronológico):
     * Si un comentario responde a `padre_id`:
       * Si el comentario hijo es corto y de acuerdo (*"tal cual"*, *"exacto"*, *"coincido"*, *"100%"*, *"sí"*) → `narrativa = padre.narrativa`, `postura = padre.postura`.
       * Si el comentario hijo es corto y de contradicción (*"mentira"*, *"nada que ver"*, *"falso"*, *"cualquiera"*, *"pero qué decís"*) → `narrativa = padre.narrativa`, `postura = opuesta(padre.postura)`.
5. **Cálculo Quirúrgico de `fecha_origen` (`origen` score = 1.0):**
   * Para cada narrativa detectada, agrupar sus comentarios por hora (`date_trunc('hour', fecha)`).
   * Buscar la primera ventana horaria donde la narrativa supera su umbral de activación real (por ejemplo, la primera hora con $\ge 10$ comentarios o $\ge 2\%$ del volumen pico horario de esa narrativa, concentrados en una publicación origen).
   * Tomar el timestamp exacto del **primer comentario** de esa ventana de nacimiento en formato ISO `-03:00`. Esto da error $< 0.5\text{ horas}$ tanto para narrativas que nacen el día 1 como para las emergentes (`L5`, `B4`, `H3`) que nacen en los días 6, 7 y 10.

### Módulo 3: `coordination_hunter.py` (25 puntos de Coordinación)
Para atrapar los **3 arquetipos de campañas (`C1`, `C2`, `C3`)** con F1 > 0.90 y separar cada operación en su propio grupo (`g001`, `g002`, `g003`...):

1. **Construcción del Grafo Multi-Señal de Co-Acción entre Cuentas $G = (V, E)$:**
   Evaluamos pares de cuentas $(u, v)$ que co-participan en las mismas publicaciones dentro de una ventana temporal corta ($\Delta t \le 15\text{ minutos}$):
   * **Señal 1 — Sincronización Anormal Descontando Viralidad:**
     Si una publicación tiene $N_p$ comentarios en esa ventana de 15 minutos, el aporte al peso de la arista $(u, v)$ es inversamente proporcional a la viralidad de la publicación ($w_{sync} = 1 / \log_2(2 + N_p)$). Así, coincidir en un post tranquilo vale mucho más que coincidir en un post viral.
   * **Señal 2 — Similitud de Texto / Hashtags en Ráfaga (Atrapa C1 y C3):**
     Si $u$ y $v$ comentan en la misma publicación dentro de 20 minutos con alta similitud léxica (Jaccard de 3-gramas $\ge 0.5$) o compartiendo el mismo hashtag en texto → suma peso fuerte $w_{text}$.
   * **Señal 3 — Respuestas Mutuas Dirigidas (`respuesta_a`) (Atrapa C2):**
     En `C2`, las cuentas coordinadas tienen textos variados pero se responden entre sí en las mismas publicaciones para inflar la conversación. Cada vez que $u$ responde a $v$ (o viceversa) apoyando la misma narrativa dentro de pocos minutos → suma peso fuerte $w_{reply}$.
   * **Señal 4 — Perfil de Cuenta y Horario (Boost para C1 y C3):**
     * Fecha de creación reciente (`fecha_creacion` pocos días antes del inicio del dataset).
     * Proporción de comentarios en horario de madrugada (`01:00` a `05:00` ART).
     * Concentración temática: aunque tengan algunos comentarios de cobertura, una fracción alta de sus coincidencias sincronizadas ocurre empujando la **misma narrativa y postura**.
2. **Poda de Aristas y Detección de Comunidades:**
   * Filtrar aristas: dos cuentas $(u, v)$ solo se conectan si **coinciden repetidamente** en múltiples publicaciones distintas (ej. $\ge 3$ co-ocurrencias sincronizadas en publicaciones diferentes) y su score combinado supera el percentil de azar (modelo nulo hipergeométrico / umbral de significancia).
   * Aplicar **Componentes Conexas / Algoritmo de Louvain (o Label Propagation)** sobre el grafo filtrado, reteniendo solo comunidades de tamaño $\ge 8$ cuentas con alta densidad interna de co-ocurrencia y coherencia de narrativa principal.
   * Asignar a cada comunidad detectada una etiqueta limpia (`g001`, `g002`, `g003`...) en `cuentas.csv`.

---

## 5. Dashboard Visual de Inteligencia (`dashboard/app.py` — 10 puntos de Visualización y Pitch)

Construir un dashboard web rápido y súper visual que lea directamente de DuckDB y de la carpeta `entrega/` o `salida/practica/` con 4 paneles clave para mostrar en el video de 3 minutos:
1. **Radar de Narrativas y Línea de Tiempo (Streamgraph / Serie Temporal):** Muestra cómo evolucionan las narrativas día a día y marca con una alerta roja el momento exacto (`fecha_origen`) en que irrumpen las narrativas emergentes/rumores (`L5`, `B4`, `H3`).
2. **Termómetro de Postura por Narrativa:** Barras apiladas `apoya` vs `rechaza` + ejemplos reales de comentarios y cómo cambiaron de redacción con los días.
3. **Grafo Interactivo de Operaciones Coordinadas (Caza de Trolls):** Visualización de red donde cada nodo es una cuenta coordinada coloreada por operación (`g001`, `g002`, `g003`), mostrando su firma conductual (horario de ataque 00-23hs, antigüedad de cuentas, publicaciones atacadas).
4. **Panel de Costo y Performance (`costo.json`):** Muestra el tiempo total de ejecución (< 30 segundos), uso de DuckDB + Parquet y costo de IA ($0.00 USD).

---

## 6. Plan de Trabajo Paso a Paso para la IA Ejecutora

1. **Paso 1 — Clonar e inspeccionar:** Clonar `https://github.com/qualydata-org/desafio-qualydata`, copiar `datos/`, `evaluacion/` y `ejemplos/` a la raíz del proyecto y correr `linea_base.py` + `puntuar.py` para registrar el baseline inicial (~17/55 puntos).
2. **Paso 2 — Implementar `src/storage_duckdb.py`:** Ingestar los `.csv.gz` en DuckDB y exportar a Parquet, verificando velocidad y memoria.
3. **Paso 3 — Implementar y calibrar `src/narrative_engine.py`:**
   * Programar la detección de ruido, el clustering jerárquico/temático sin depender de etiquetas hardcodeadas, la herencia de narrativa y postura por `respuesta_a`, y el detector de `fecha_origen` por ráfaga inicial.
   * Medir con `python evaluacion/puntuar.py --verdad datos/practica/verdad --entrega salida/practica` hasta que el bloque **NARRATIVAS** supere **24 - 28 / 30 puntos**.
4. **Paso 4 — Implementar y calibrar `src/coordination_hunter.py`:**
   * Programar el detector de grafos multi-señal (co-ocurrencia temporal en múltiples posts + similitud de texto/hashtags + grafo de `respuesta_a` + antigüedad/horario nocturno).
   * Medir con `evaluacion/puntuar.py` hasta que el bloque **COORDINACIÓN** detecte limpiamente las 3 operaciones (`C1`, `C2`, `C3`) y supere **22 - 24.5 / 25 puntos** (logrando un **TOTAL AUTOMÁTICO > 47 - 52 / 55 puntos**, casi el triple que la línea base).
   * *Importante:* Asegurar que ninguna regla dependa de nombres de temas del set de práctica ("litio", "colectivo", "hospital"), para que el viernes 13 generalice al 100% sobre el `datos/final`.
5. **Paso 5 — Comando Único, `entrega/` y Dashboard:**
   * Dejar `python -m src.pipeline --datos datos/practica --entrega entrega` funcionando de punta a punta en un solo comando, generando `comentarios.csv`, `narrativas.csv`, `cuentas.csv` y `costo.json`.
   * Construir `dashboard/app.py` y el `README.md` explicando la arquitectura DuckDB + Parquet, cómo escalar a millones de comentarios por día y cómo correr el pipeline y el dashboard.
