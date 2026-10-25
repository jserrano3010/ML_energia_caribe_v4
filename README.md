# Calor y energía en el Caribe: demanda horaria de Air-e y el modelo original ATER

Proyecto final de Machine Learning (Maestría en Ingeniería Industrial, Prof. Lihki Rubio). Autores: Edwin Yunis y Jairo Serrano.

Pronóstico, al cierre de cada día, de la demanda de las 24 horas del día siguiente del mercado de Air-e (CARIBE SOL, XM), donde está Sabanagrande, con el clima de Sabanagrande (NASA POWER). Modelo original **ATER**: agregación de expertos en línea con pesos por hora y calor y corrección de nivel (`src/original_model.py`).

**Resultado (prueba 2025-01 a 2026-10):** ATER MAPE 2,44 %; −65 % de MSE frente a la persistencia y −3,9 % frente al random forest relativo. Superioridad significativa con MCS, Giacomini-White y Diebold-Mariano (ajuste de Holm).

**Versión 4.0** (octubre de 2026): tablas rediseñadas en el dashboard y el libro, tabla de líderes con las 28 combinaciones del Entregable 2, 21 pruebas con ajuste de Holm y tiempos de cómputo de cada modelo frente a ATER.

- Dashboard v4: https://ml-energia-caribe-v4.onrender.com
- Jupyter Book v4: https://jserrano3010.github.io/ML_energia_caribe_v4/
- Versión 3.0 (se conserva para trazabilidad): https://ml-energia-caribe.onrender.com · https://jserrano3010.github.io/ML_energia_caribe/ · https://github.com/jserrano3010/ML_energia_caribe

## Estructura
```
app.py              # Dash multipágina (expone server para gunicorn)
pages/              # contexto.py, eda.py, modelos.py
src/                # original_model.py, data.py, plots.py, pipeline_energia.py, correr_*.py ...
artifacts/          # ater_dashboard.joblib + metadata.json / metadata_dashboard.json
data/               # datos livianos del dashboard
datos/              # datos crudos (XM, NASA POWER)
resultados/         # salidas de los experimentos (CSV/PKL), incl. tabla_maestra_experimentos.csv
tests/              # pytest
libro/              # Jupyter Book
```

## Instalación y uso
```bash
pip install -r requirements.txt          # dashboard
python app.py                            # http://127.0.0.1:8050
pip install -r requirements-dev.txt      # experimentos, pruebas y libro
pytest -q tests/
```
Orden completo de reproducción: `libro/reproducibilidad.md`. Semilla global: 2026.

## Despliegue
Render → New → Blueprint → este repositorio (`render.yaml`). Start: `gunicorn app:server`.
