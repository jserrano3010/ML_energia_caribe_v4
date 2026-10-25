"""Tiempos de cómputo de todos los modelos en la MISMA máquina y con los MISMOS datos.

Para cada modelo se mide:
* entrenamiento final: ajuste con todo el entrenamiento (oct-2020 a dic-2024, 36.552 h);
* inferencia en la prueba: pronóstico de las 15.672 h de 2025-2026;
* pronóstico de un día: las 24 horas de un día (mediana de 5 repeticiones), que es lo que se hace en operación;
* búsqueda de hiperparámetros: tiempo registrado en la validación anidada (Entregable 2) o medido aquí (Optuna de ATER).

Se ejecuta desde la raíz del proyecto:  python src/tiempos_computo.py
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np, pandas as pd, optuna
import pipeline_energia as P
from correr_e2 import construir_modelo
from original_model import AgregacionTermica, ModeloOriginal, _expertos

optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 2026; np.random.seed(SEED)
P.GRILLAS = P.GRILLAS_V2
CORTE = pd.Timestamp("2025-01-01"); GAP = pd.Timedelta(hours=48)
df = pd.read_pickle("resultados/datos_horarios.pkl")
X, y = P.construir(df); ok = X.notna().all(axis=1); X, y = X[ok], y[ok]
tr, te = X.index < CORTE - GAP, X.index >= CORTE
Xtr, ytr, Xte = X[tr], y[tr], X[te]
dia = Xte.iloc[-24:]


def cronometrar(fit, predict):
    t0 = time.perf_counter(); m = fit(); t_fit = time.perf_counter() - t0
    t0 = time.perf_counter(); predict(m, Xte); t_inf = time.perf_counter() - t0
    reps = []
    for _ in range(5):
        t0 = time.perf_counter(); predict(m, dia); reps.append(time.perf_counter() - t0)
    return t_fit, t_inf, float(np.median(reps))


filas = []
cfg = json.load(open("resultados/eleccion.json"))["configuraciones"]

# 1. tradicionales
for n in P.TRADICIONALES:
    f, i, d = cronometrar(lambda: None, lambda m, Z: P.pred_tradicional(n, Xtr, ytr, Z))
    filas.append(dict(modelo=n, grupo="Tradicional", optimizador="—", busqueda_s=0.0, entrenamiento_s=f, inferencia_prueba_s=i, pronostico_dia_s=d))
    print(n, flush=True)

# 2. benchmarks del proyecto (los expertos de ATER)
for n, g in cfg.items():
    f, i, d = cronometrar(lambda: P.GRILLAS[n][g]().fit(Xtr, ytr), lambda m, Z: m.predict(Z))
    filas.append(dict(modelo=n, grupo="Benchmark del proyecto", optimizador="Grid (validación)", busqueda_s=np.nan,
                      entrenamiento_s=f, inferencia_prueba_s=i, pronostico_dia_s=d))
    print(n, flush=True)

# 3. los 7 modelos vistos en clase (mejor combinación del Entregable 2)
E2M = pd.read_csv("resultados/e2_mejores.csv")
E2C = pd.read_csv("resultados/e2_combinaciones.csv")
for _, r in E2M.iterrows():
    h = json.loads(r.hiperparametros)
    f, i, d = cronometrar(lambda: construir_modelo(r.modelo, h).fit(Xtr, ytr), lambda m, Z: m.predict(Z))
    busq = float(E2C[(E2C.modelo == r.modelo) & (E2C.optimizador == r.optimizador)].tiempo_busqueda_s.iloc[0])
    filas.append(dict(modelo=f"{r.modelo} (E2)", grupo="Visto en clase (E2)", optimizador=r.optimizador, busqueda_s=busq,
                      entrenamiento_s=f, inferencia_prueba_s=i, pronostico_dia_s=d))
    print(r.modelo, flush=True)

# 4. ATER: búsqueda con Optuna (validación anidada 4 x 60 ensayos + final 80) y modelo completo
pv = pd.read_pickle("resultados/pred_validacion.pkl")
COL = {n: (n if n in P.TRADICIONALES else f"{n}|{cfg[n]}") for n in _expertos("completo")}
VAL = {}
for k in sorted(pv.fold.unique()):
    fk = pv[pv.fold == k]; Xi = X.loc[fk.index]
    M = np.column_stack([Xi.hora, Xi.calor, Xi.media_7d, pd.factorize(Xi.index.normalize())[0], fk[[COL[n] for n in COL]].values])
    VAL[k] = (M, fk.y.values)


def optimizar(folds_, n):
    def obj(t):
        par = dict(eta=t.suggest_float("eta", 1.0, 1000.0, log=True), lam=t.suggest_float("lam", 0.90, 1.0),
                   sigma=t.suggest_float("sigma", 0.5, 50.0, log=True), eta_b=t.suggest_float("eta_b", 0.0, 0.1))
        return np.mean([np.mean((AgregacionTermica(**par).iniciar_vacio(5).predict(*VAL[k]) - VAL[k][1]) ** 2) for k in folds_])
    optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED)).optimize(obj, n_trials=n)


t0 = time.perf_counter()
for k in range(1, 5):
    optimizar(list(range(k)), 60)
optimizar(list(range(5)), 80)
busq_ater = time.perf_counter() - t0
hp = json.load(open("resultados/orig_hiperparametros.json"))["hiperparametros"]
f, i, d = cronometrar(lambda: ModeloOriginal(**hp, random_state=SEED).fit(Xtr, ytr), lambda m, Z: m.predict(Z))
filas.append(dict(modelo="ATER (modelo original)", grupo="Original", optimizador="Bayesiano (Optuna TPE)", busqueda_s=busq_ater,
                  entrenamiento_s=f, inferencia_prueba_s=i, pronostico_dia_s=d))

t = pd.DataFrame(filas)
t["nucleos"] = os.cpu_count()
t.to_csv("resultados/tiempos_computo.csv", index=False)
print(t.round(3).to_string(index=False))
print("búsqueda ATER (320 ensayos):", round(busq_ater, 1), "s")
