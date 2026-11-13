# INTI-GRAPH

**Motor de inteligencia de narrativas y campañas coordinadas a escala.**
Hackathon Argentina Space 2026 · Track Big Data / QualyData · Salta, 11-13 de noviembre de 2026.

> Todos los datos del desafío son sintéticos. No hay datos personales.

Video demo (36 s, slideshow de capturas del dashboard):
<https://github.com/Fernandofarfan/inti-graph/releases/download/demo-video/INTI-GRAPH-demo.mp4>

---

## Resultado sobre el set final (viernes 13)

El set final (150.000 comentarios, 10 días, 12.002 cuentas, 372 publicaciones, 8 fuentes) se
procesó con el mismo comando y **sin verdad**:

```bash
python -m src.pipeline --datos datos/final --entrega entrega
```

| Métrica | Valor |
|---|---|
| Comentarios procesados | 150.000 |
| Narrativas detectadas | 14 (12 clusters + 2 cohortes tardías) |
| Ruido separado | 30.536 (20,4 %) |
| Postura | 72,8 % apoya · 27,2 % rechaza sobre los 119.464 comentarios con narrativa |
| Cuentas coordinadas | 77 en 3 operaciones (50 / 22 / 5) |
| Tiempo de proceso (CPU, sin caché) | 194 s · 772 comentarios/s |
| Costo de IA | USD 0,00 · 0 tokens |
| Warehouse | DuckDB + Parquet ZSTD, 2,6 MB |

`entrega/` contiene la salida de esa corrida. El puntaje automático no se puede calcular
sobre el set final porque el release no incluye verdad; el número medible sigue siendo el
del set de práctica (49.27/55).

**Sobre `fecha_origen`:** en las narrativas base el estimador devuelve el inicio de la
cohorte tardía (el mismo minuto para varias), porque es la primera redacción propia que se
puede distinguir de los textos heredados de narrativas viejas. El dashboard muestra además
el primer comentario de cada narrativa, que sí es la fecha real de aparición.

---

## Resultado medido (set de práctica, evaluador oficial)

`python evaluacion/puntuar.py --verdad datos/practica/verdad --entrega entrega`

| Bloque | Métrica | Puntos |
|---|---|---|
| Narrativas | B-cubed P=0.690 · R=0.668 · **F1=0.679** | 10.18 / 15 |
| Ruido | **F1=1.000** (1 falso positivo en 200.000) | 4.50 / 4.5 |
| Postura | **macro-F1=0.926** | 5.56 / 6 |
| Origen de emergentes | **0.896** (L5 1.00 · B4 0.76 · H3 0.93) | 4.03 / 4.5 |
| **Narrativas** | | **24.27 / 30** |
| Cuentas coordinadas | **P=1.000 · R=1.000 · F1=1.000** | 17.50 / 17.5 |
| Grupos de operación | **B-cubed=1.000** | 7.50 / 7.5 |
| **Coordinación** | | **25.00 / 25** |
| **TOTAL AUTOMÁTICO** | | **49.27 / 55** |

Línea base oficial (`ejemplos/linea_base.py`): **≈17 / 55**. Pipeline completo: **49.27 / 55**.

Costo de la corrida de práctica: **~2 min** (131 s medidos en la primera corrida),
**279 MB** de pico en Python, **USD 0.00** de IA (sin API ni GPU), warehouse Parquet de
**3.7 MB**.

---

## Qué hace

Dos tareas sobre el mismo set:

1. **Narrativas.** Descubre sin lista previa qué afirmaciones circulan, asigna cada
   comentario a una narrativa o lo marca como ruido, decide si lo apoya o lo rechaza y
   estima cuándo apareció.
2. **Coordinación.** Detecta cuentas que actúan en conjunto y las agrupa por operación,
   separando actividad armada de opinión genuina.

Todo con **un comando**, en una notebook común, sin GPU y sin servicios externos.

---

## Cómo correrlo

Requisitos: **Python 3.10+** y ~1 GB de disco libre (modelo + caché de embeddings).
Todo corre en CPU; no hace falta GPU ni cuenta en ningún servicio.

```bash
# 0) instalación (una sola vez, desde la raíz del repo)
python -m venv .venv
.venv/bin/pip install -r requirements.txt          # Windows: .venv\Scripts\pip install -r requirements.txt
.venv/bin/python scripts/descargar_modelo.py       # opcional: el pipeline lo baja solo (~118 MB, Apache-2.0)
```

```bash
# 1) pipeline completo -> entrega/
python -m src.pipeline --datos datos/practica --entrega entrega --calibracion datos/practica/verdad

# 2) puntaje automático del jurado
python evaluacion/puntuar.py --verdad datos/practica/verdad --entrega entrega

# 3) dashboard autocontenido
python dashboard/generar.py --datos datos/practica --entrega entrega
python -m http.server 8000 --directory dashboard
# abrir http://localhost:8000
```

En este entorno de desarrollo el intérprete ya está creado en la raíz del workspace, así
que los mismos comandos son `..\.venv\Scripts\python.exe -m src.pipeline ...`.

Para el **set final**, sin verdad disponible:

```bash
python -m src.pipeline --datos datos/final --entrega entrega
```

`--calibracion` tiene como valor por defecto la verdad de **práctica**: el set final nunca
aporta etiquetas. El pipeline no lee `datos/final/verdad` ni ningún archivo de verdad del
set objetivo. Verificado corriendo sobre una copia del set de práctica sin carpeta `verdad`
(resultado idéntico: 49.27/55).

### Salidas

```
entrega/                  # lo único que se commitea para el jurado
├── comentarios.csv       # comentario_id, narrativa, postura  (200.000 filas)
├── narrativas.csv        # narrativa, titulo, frases_clave, fecha_origen
├── cuentas.csv           # cuenta_id, grupo  (una fila por cuenta; vacío = genuina)
└── costo.json            # formato exacto del desafío (8 claves)
salida/
├── warehouse/            # DuckDB + Parquet ZSTD (capa de almacenamiento)
├── calibracion/          # artefactos aprendidos de la práctica (cacheados)
├── features_cuentas.parquet
└── reporte.json
dashboard/
├── generar.py            # arma el HTML desde la entrega
├── plantilla.html
└── index.html            # autocontenido, sin CDN
```

---

## Arquitectura

```
CSV.gz ──► DuckDB ──► Parquet ZSTD ──► pandas / NumPy
             │                              │
             │                              ├─ ruido: TF-IDF + regresión logística
             │                              ├─ embeddings: MiniLM multilingüe ONNX int8 (384d)
             │                              ├─ narrativas: tema → narrativa (k-means por tema)
             │                              ├─ postura: marcadores + léxico log-odds
             │                              ├─ origen: onset + cohorte temporal
             │                              └─ coordinación: RandomForest por cuenta + reglas
             ▼                              ▼
      agregados SQL                  entrega/ (4 archivos) ──► dashboard HTML
```

### Módulos (`src/`)

| Módulo | Rol |
|---|---|
| `ingesta.py` | lectura de CSV.gz, normalización de texto y plantillas |
| `almacen.py` | **DuckDB + Parquet ZSTD**: warehouse y agregados SQL |
| `ruido.py` | detector de comentarios sin narrativa (features estructurales, no temáticas) |
| `embeddings.py` | MiniLM multilingüe cuantizado ONNX, caché `.npz` por hash de texto |
| `narrativas.py` | métrica de pares, jerarquía tema→narrativa, cohorte temporal, herencia por respuestas, postura, origen |
| `coordinacion.py` | 18 features por cuenta, modelo RandomForest, reglas y agrupamiento |
| `calibracion.py` | artefactos aprendidos una vez desde la verdad de práctica |
| `pipeline.py` | orquestador de un solo comando + `costo.json` y `reporte.json` |

### Decisiones que sostienen el puntaje

- **Temas primero, narrativas después.** Agrupar las publicaciones por el promedio de
  embeddings de sus comentarios separa macro-temas casi perfectamente (3 temas en
  práctica, silueta 0.62). Recién dentro de cada tema se agrupan los textos en
  narrativas. Esto subió B-cubed de 0.47 a **0.679** sin tocar la verdad.
- **Herencia por `respuesta_a`.** Las respuestas de hasta 5 palabras ("tal cual",
  "exacto", "mentira") heredan la narrativa del comentario padre; si contradicen,
  invierten la postura. Subió B-cubed de 0.644 a 0.679 y la postura de 0.914 a 0.926,
  y es una regla explícita del desafío.
- **k total fijo = 12, repartido por masa entre temas.** Robusto a que cambie la
  cantidad de temas o de narrativas del set final; no depende de los nombres.
- **Ruido por forma, no por tema.** Longitud, repetición y presencia de `?`; F1=1.000
  en práctica (un solo falso positivo) y transferible a temas nuevos.
- **Postura por léxico log-odds + marcadores de rechazo**, sin vocabulario temático.
- **Origen por onset**: primer día que alcanza 10% del pico diario del cluster. Si el
  cluster tiene una cohorte tardía dominante (≥35%), el onset se estima solo sobre esa
  cohorte. Evita que un comentario aislado del día 1 corra el origen.
- **Coordinación estructural**: antigüedad de la cuenta, nocturnidad, duplicación de
  textos, cobertura de publicaciones y reciprocidad de respuestas. El modelo aporta el
  ranking; una regla cubre cuentas nuevas con plantilla. Agrupamiento por antigüedad +
  k-means (k=2) sobre 8 features estandarizadas.
- **Umbral único 0.30**: en práctica deja 115 cuentas marcadas, exactamente las 115
  reales (3 operaciones), sin falsos positivos.

---

## Almacenamiento: dónde y por qué

Los datos viven en **DuckDB + Parquet ZSTD** (`salida/warehouse/`), generados por el
pipeline en cada corrida:

| Tabla | Filas | Formato |
|---|---|---|
| comentarios | 200.000 | Parquet ZSTD |
| cuentas | 15.991 | Parquet ZSTD |
| publicaciones | 506 | Parquet ZSTD |
| fuentes | 8 | Parquet ZSTD |
| actividad_diaria | 1.311 | Parquet ZSTD |
| actividad_cuentas | 15.991 | Parquet ZSTD |

**Por qué esta base:**

- **Parquet columnar + ZSTD** comprime ~5x un CSV y permite leer solo las columnas que
  se usan (fecha, cuenta, publicación). El set es de solo lectura: no hace falta OLTP.
- **DuckDB** es un motor SQL embebido: cero servidores, cero configuración, y las
  agregaciones (actividad por día, por cuenta, por publicación) corren en segundos.
- El análisis semántico se hace en **pandas/NumPy** porque 200.000 comentarios entran
  cómodos en RAM (pico 279 MB); el warehouse es la fuente de verdad y el punto de
  entrada para escalar.
- **Escala a millones/día:** particionar Parquet por `fecha` (`warehouse/dia=YYYY-MM-DD/`)
  y procesar por lotes; los embeddings se cachean por hash de texto, así que un texto
  repetido nunca se recalcula. El costo es lineal en comentarios, no en usuarios ni en
  temas.

`costo.json` reporta el formato exacto del desafío: `comentarios_procesados`,
`segundos`, `tokens_ia`, `usd_ia`, `usd_por_mil_comentarios`, `hardware`,
`almacenamiento` y `notas`.

---

## Uso de la verdad de práctica y generalización

El desafío permite explícitamente entrenar con la verdad de práctica. Este pipeline la
usa **solo** para cuatro artefactos, cacheados en `salida/calibracion/`:

1. pesos por dimensión del embedding (qué dimensiones separan "misma narrativa");
2. léxico de postura (log-odds de términos de rechazo, sin temas);
3. detector de ruido (features de forma);
4. modelo de cuentas (features estructurales, sin nombres ni temas).

Nada de esto memoriza temas, narrativas ni campañas. No hay listas de palabras del tipo
"litio", "hospital" o "colectivo" en el código de predicción. El k total, los umbrales y
la fracción de cohorte se calibraron en práctica y se documentan como constantes.

**Limitación honesta:** B-cubed 0.679 es el techo práctico de este enfoque; el oráculo
por centroides ronda 0.86. La narrativa H3 comparte vocabulario con H1/H2/H4 y k-means
no la separa del todo, por eso su origen estimado (0.93) es algo peor que L5 (1.00). El
estimador de origen de clusters que mezclan narrativas base y emergentes usa la cohorte
tardía; para narrativas base no emergentes la fecha puede reflejar la nueva redacción y
no el primer comentario (el dashboard muestra ambas).

---

## Reproducibilidad

- Semillas fijas (`random_state=0`) en KMeans, RandomForest y muestreo de pares.
- Embeddings determinísticos y cacheados por hash del texto normalizado.
- Dos corridas consecutivas dan el mismo `entrega/` (salvo `costo.json`, que reporta
  tiempos).
- Un solo comando, sin pasos manuales ni etiquetado del set final.
- Manejo de datos sucios: `respuesta_a` ausente o fuera del set, cuentas sin fila en
  `cuentas.csv`, textos vacíos, fechas con zona horaria y comentarios duplicados.

## Qué existía antes del evento

- `datos/`, `evaluacion/` y `ejemplos/` son copias **sin modificar** del repositorio
  oficial `qualydata-org/desafio-qualydata`.
- La línea base `ejemplos/linea_base.py` es del desafío (~17/55) y no se usa.
- Todo el código de `src/`, `dashboard/` y los scripts de calibración es original,
  escrito durante el hackathon. Librerías públicas usadas: pandas, NumPy, scikit-learn,
  SciPy, DuckDB, pyarrow, joblib, ONNX Runtime y el modelo multilingüe MiniLM
  (`Xenova/paraphrase-multilingual-MiniLM-L12-v2`, ONNX int8).

## Licencia

MIT.
