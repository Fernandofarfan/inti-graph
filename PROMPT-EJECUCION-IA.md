# 🤖 PROMPT REUTILIZABLE (TRACK 2: BIG DATA — INTI-GRAPH)

Este prompt sirve para que otra IA reconstruya, audite o mejore el proyecto ya
implementado. Copiá el bloque y pegalo estando dentro de `Track-2-BigData_INTI-GRAPH`.

---

```markdown
Estás en el proyecto **INTI-GRAPH** del Track QualyData — Big Data del Hackathon
Argentina Space 2026. El pipeline YA está implementado y verificado con el evaluador
oficial en **49.27 / 55** puntos automáticos. Tu tarea es auditarlo, reproducirlo y,
si encontrás una mejora generalizable, aplicarla sin romper lo que funciona.

Contexto imprescindible:
- Leé `README.md` (arquitectura y resultados), `README-DESAFIO-OFICIAL.md` (bases) y
  `evaluacion/puntuar.py` (script exacto del jurado).
- Usá el intérprete `..\.venv\Scripts\python.exe` desde esta carpeta.
- No modifiques `datos/`, `evaluacion/` ni `ejemplos/`: son copias oficiales.
- `entrega/` es el resultado del set final; `salida/` es para práctica.
- Prohibido etiquetar a mano el set final y prohibido usar su verdad (no existe).
  La calibración se aprende SOLO de `datos/practica/verdad` y se cachea en
  `salida/calibracion/`.

Comandos de verificación (deben dar 49.27/55 o mejor):
  ..\.venv\Scripts\python.exe -m src.pipeline --datos datos/practica --entrega entrega --calibracion datos/practica/verdad
  ..\.venv\Scripts\python.exe evaluacion\puntuar.py --verdad datos/practica\verdad --entrega entrega

Arquitectura que tenés que respetar:
1. `src/almacen.py`: CSV.gz → DuckDB → Parquet ZSTD en `salida/warehouse/`, más
   agregados de actividad diaria y por cuenta. Es la capa de almacenamiento que
   evalúa el jurado (20 pts): justificá Parquet columnar + DuckDB embebido y el
   camino de escalado por particiones de fecha.
2. `src/ruido.py`: detector supervisado con features de forma (largo, repetición,
   pregunta), sin vocabulario temático. Objetivo: F1 ≥ 0.99.
3. `src/embeddings.py`: MiniLM multilingüe cuantizado ONNX (384d), cacheado por hash
   del texto normalizado. Nada de API.
4. `src/narrativas.py`:
   - métrica supervisada de pares sobre la verdad de práctica;
   - **jerarquía tema → narrativa**: las publicaciones se agrupan por el promedio de
     embeddings de sus comentarios (k por silueta), y dentro de cada tema se agrupan
     los textos con k-means repartiendo un total fijo de 12 narrativas por masa;
   - cohorte temporal: si ≥35% del peso de un cluster son textos que aparecen por
     primera vez después del 35% del período, la parte tardía se separa con sufijo L;
   - postura con marcadores de rechazo + léxico log-odds;
   - herencia por `respuesta_a`: las respuestas de hasta 5 palabras copian la
     narrativa del padre y, si contradicen, invierten la postura;
   - origen por onset: primer día que alcanza 10% del pico diario; si hay cohorte
     tardía con peso suficiente, el onset se calcula solo sobre esa cohorte.
5. `src/coordinacion.py`: 18 features estructurales por cuenta (antigüedad,
   nocturnidad, duplicación, cobertura, reciprocidad, plantillas), RandomForest
   calibrado + regla para cuentas nuevas, umbral 0.30; agrupamiento por antigüedad
   (nuevas <30 días vs veteranas) y k-means k=2 sobre 8 features estandarizadas.
6. `src/pipeline.py`: orquestador de un comando; escribe `comentarios.csv`,
   `narrativas.csv`, `cuentas.csv` y `costo.json` en el formato exacto del desafío
   (8 claves), más `reporte.json` y `salida/features_cuentas.parquet`.
7. `dashboard/generar.py`: HTML autocontenido (sin CDN) con línea de tiempo,
   posturas, operaciones y costo. Se abre con `python -m http.server`.

Reglas de generalización:
- No hardcodees temas, narrativas ni campañas de práctica ("litio", "hospital",
  "colectivo", "C1", "L5"). Todo debe descubrirse de los textos.
- Cualquier ajuste de hiperparámetro debe validarse contra la verdad de práctica y
  justificarse como estructural; no persigas el set de práctica con reglas ad hoc.
- Mantené semillas fijas y determinismo.
- Corré también el pipeline sobre una copia del set SIN carpeta `verdad` para
  comprobar que no hay fuga de etiquetas.

Qué se espera de vos:
1. Reproducí la corrida y el puntaje; reportá la tabla por bloque.
2. Revisá que no haya uso de la verdad del set objetivo ni rutas hardcodeadas.
3. Si proponés una mejora, mostrá el antes/después del puntaje y explicá por qué
   generalizaría a otros temas.
4. Actualizá `README.md` si cambiás algo verificable.
```

---

## Notas para quien use este prompt

- El objetivo realista de este enfoque es **48-49 / 55** automáticos; el techo de
  B-cubed con centroides está en ~0.86 y el resto de los bloques ya está casi perfecto.
- La mayor palanca pendiente es separar narrativas que comparten vocabulario dentro de
  un mismo tema (por ejemplo H3 vs H1/H2/H4). Cualquier intento debe medirse con
  `scripts/ajustar_jerarquia.py` y `scripts/diagnostico_temas_origen.py`.
- Los scripts de `scripts/` son el laboratorio: diagnósticos de clusters, coordinación,
  representaciones y barridos de k. No son necesarios para la entrega, pero hacen el
  trabajo auditable.
