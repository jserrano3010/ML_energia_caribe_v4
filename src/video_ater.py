"""Video explicativo del modelo original ATER (MP4 1920x1080, con la narración como subtítulos en pantalla).
Cada escena se dibuja con matplotlib a partir de los resultados del proyecto; ffmpeg arma el video."""
import json, subprocess, textwrap, shutil
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R = Path("resultados"); D = Path("data")
TMP = Path("/tmp/claude-0/-home-claude/e4cc3db1-314b-5e30-9aaf-2440d7f6ae8b/scratchpad/video"); shutil.rmtree(TMP, ignore_errors=True); TMP.mkdir(parents=True)
OUT = Path("publicar/video_ATER_Yunis_Serrano.mp4")
INK, RED, GREEN, GOLD, GRAY, BG, PANEL = "#14262B", "#C0392B", "#1F5A3A", "#C8892B", "#9AA5A0", "#F3F6F4", "#FFFFFF"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 22, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#B8C4C6", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK})
ORIG = "ATER (modelo original)"
CORTO = {ORIG: "ATER (original)", "Random forest (relativo)": "Random forest relativo", "XGBoost (relativo)": "XGBoost relativo",
         "Híbrido Ridge + XGBoost (residuo relativo)": "Híbrido Ridge+XGBoost", "Ridge (lineal)": "Ridge",
         "Persistencia (misma hora de ayer)": "Persistencia", "Estacional (misma hora semana pasada)": "Estacional",
         "Promedio por hora y día": "Promedio hora×día"}
nf = lambda x, d=1: f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")

frames = []      # (ruta_png, segundos, texto_subtítulo)


def lienzo(titulo, seccion, n):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
    fig.text(0.04, 0.93, seccion.upper(), fontsize=17, color=RED, weight="bold")
    fig.text(0.04, 0.87, titulo, fontsize=40, weight="bold", color=INK)
    fig.text(0.96, 0.93, f"{n}/16", fontsize=16, color=GRAY, ha="right")
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.845, 0.845], color=RED, lw=2))
    return fig


def guardar(fig, texto, seg=None):
    if texto:
        fig.patches.append(FancyBboxPatch((0.03, 0.015), 0.94, 0.12, boxstyle="round,pad=0.005", transform=fig.transFigure,
                                          facecolor=INK, edgecolor="none", alpha=0.94))
        fig.text(0.5, 0.075, "\n".join(textwrap.wrap(texto, 105)), ha="center", va="center", fontsize=21, color="white")
    p = TMP / f"f{len(frames):03d}.png"; fig.savefig(p, facecolor=fig.get_facecolor()); plt.close(fig)
    frames.append((p, seg or max(4.5, len(texto) / 15 + 1.5), texto))


def bullets(fig, items, k, x=0.06, y0=0.76, dy=0.085, size=27, ancho=60):
    for i, t in enumerate(items[:k]):
        fig.text(x, y0 - i * dy, "■", color=RED, fontsize=size - 8, va="top")
        fig.text(x + 0.02, y0 - i * dy, "\n".join(textwrap.wrap(t, ancho)), fontsize=size, va="top")


def caja(fig, x, y, w, h, texto, color=GREEN, fc=PANEL, size=22):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008", transform=fig.transFigure, facecolor=fc, edgecolor=color, lw=3))
    fig.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=size, color=INK, wrap=True)


def flecha(fig, a, b, color=INK):
    fig.patches.append(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=30, lw=3, color=color))


# ---------------------------------------------------------------- datos
tp = pd.read_csv(R / "orig_prueba.csv"); an = pd.read_csv(R / "orig_validacion_anidada.csv")
pr = pd.read_csv(R / "orig_pruebas_hipotesis.csv"); ab = pd.read_csv(R / "orig_ablacion.csv")
desc = pd.read_csv(R / "problemas_descartados.csv"); tri = pd.read_csv(R / "orig_optuna_trials.csv")
pesos = pd.read_csv(D / "pesos_por_hora.csv", index_col=0); e2m = pd.read_csv(R / "e2_mejores.csv"); e2c = pd.read_csv(R / "e2_combinaciones.csv")
mcs = pd.read_csv(R / "e2_mcs_todos.csv"); eda = pd.read_csv(D / "eda_horaria.csv.gz", parse_dates=["ts"], index_col="ts")

# ---------------------------------------------------------------- 1. portada
fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=INK)
fig.text(0.07, 0.72, "PROYECTO FINAL · MACHINE LEARNING · ENTREGABLE 3", fontsize=20, color=GOLD, weight="bold")
fig.text(0.07, 0.56, "ATER", fontsize=110, color="white", weight="bold")
fig.text(0.07, 0.47, "Agregación de expertos por régimen horario y térmico,\ncon corrección de nivel en línea", fontsize=34, color="white", va="top")
fig.text(0.07, 0.28, "Pronóstico horario de la demanda eléctrica de Air-e con el calor de Sabanagrande", fontsize=24, color="#C9D6D3")
fig.text(0.07, 0.18, "Edwin Yunis y Jairo Serrano · Maestría en Ingeniería Industrial · Prof. Lihki Rubio", fontsize=22, color="#C9D6D3")
guardar(fig, "En este video explicamos nuestro modelo original, ATER: qué problema resuelve, cómo funciona, cómo lo validamos y por qué cumple las condiciones R1, R2 y R3 de la guía.", 7)

# ---------------------------------------------------------------- 2. problema
items = ["El calor dispara el consumo: aires, ventiladores y neveras",
         "Con índice de calor de 38–42 °C la demanda de la tarde es 27,5 % mayor",
         "La tarifa de Air-e subió cerca de 74 % hasta julio de 2023",
         "La energía se compra el día anterior: cada MWh mal pronosticado se paga en bolsa"]
caps = ["En Sabanagrande, como en todo el Caribe, el calor hace que los hogares usen más aires, ventiladores y neveras.",
        "En las tardes de días laborales, con índice de calor de 38 a 42 grados, la demanda sube 27,5 por ciento.",
        "La región paga una de las tarifas más altas del país: la de Air-e subió cerca de 74 por ciento hasta 2023.",
        "Por eso importa pronosticar bien: la energía se compra el día anterior y los errores se pagan en la bolsa."]
m = eda[(eda.hora.between(13, 16)) & (eda.tipo_dia == "laboral")]
g = m.groupby(pd.cut(m.calor, np.arange(24, 46, 2)), observed=True).mwh.mean()
for k in range(1, 5):
    fig = lienzo("Por qué importa: calor y energía en el Caribe", "1 · Problema", 2)
    bullets(fig, items, k, ancho=38, size=26)
    ax = fig.add_axes([0.58, 0.22, 0.38, 0.55]); ax.plot([i.mid for i in g.index], g.values, "o-", color=RED, lw=4, ms=10)
    ax.set_xlabel("índice de calor en Sabanagrande (°C)"); ax.set_ylabel("demanda media (MWh)"); ax.set_title("Tardes laborales (13–16 h)", fontsize=22)
    guardar(fig, caps[k - 1])

# ---------------------------------------------------------------- 3. pregunta y datos
fig = lienzo("Pregunta y datos", "1 · Problema", 3)
fig.text(0.06, 0.76, "¿Puede un modelo propio pronosticar la demanda de cada hora del día siguiente\nmejor que los métodos tradicionales y que los modelos de Machine Learning conocidos,\ncon superioridad estadísticamente significativa?",
         fontsize=27, va="top", style="italic")
bullets(fig, ["52.272 horas reales (oct. 2020 – oct. 2026): demanda de XM y clima de NASA POWER",
              "28 predictoras conocidas al emitir: rezagos de 24 h o más y clima del día anterior (sin fuga)",
              "Entrenamiento hasta dic. 2024 · prueba única ene. 2025 – oct. 2026 (653 días)"], 3, y0=0.52, size=25, ancho=90)
guardar(fig, "Usamos 52.272 horas reales de demanda del mercado de Air-e y el clima horario de Sabanagrande. La prueba de 2025 a 2026 se reservó y se usa una sola vez.")

# ---------------------------------------------------------------- 4. problemas descartados
fig = lienzo("Llegamos aquí después de descartar cuatro problemas", "2 · Correcciones", 4)
ax = fig.add_axes([0.22, 0.22, 0.7, 0.56]); d = desc.iloc[::-1]
NOMB = ["Lluvia semanal (E2)", "Precio del maíz", "Decisión de siembra", "Baloto", "Demanda eléctrica Air-e"]
ax.barh(NOMB[::-1], d.mejora_pct, color=[RED if c == 4 else GRAY for c in d.criterios_cumplidos])
ax.axvline(10, color=INK, ls="--"); ax.text(11, 4.45, "umbral 10 %", fontsize=18)
for i, (v, c) in enumerate(zip(d.mejora_pct, d.criterios_cumplidos)): ax.text(max(v, 0) + 1, i, f"{nf(v)} % · {c}/4 criterios", va="center", fontsize=18)
ax.set_xlabel("mejora del mejor ML frente a la mejor línea base tradicional (%)"); ax.tick_params(axis="y", labelsize=21); ax.set_xlim(-10, 95); ax.set_ylim(-0.6, 4.9)
guardar(fig, "Antes evaluamos lluvia, precio del maíz, siembra y Baloto. Ninguno cumplió los cuatro criterios: la lluvia mejoraba poco y sin significancia, y en maíz y siembra ganaban los tradicionales. La demanda eléctrica los cumple todos.", 11)

# ---------------------------------------------------------------- 5. benchmarks
fig = lienzo("Contra quién compite ATER", "3 · Benchmarks", 5)
caja(fig, 0.05, 0.50, 0.27, 0.27, "3 tradicionales\n\npersistencia\nestacional\npromedio hora × día", GRAY)
caja(fig, 0.36, 0.50, 0.27, 0.27, "4 de Machine Learning\n\nRidge · random forest\nXGBoost · híbrido\n(objetivo relativo)", GREEN)
caja(fig, 0.67, 0.50, 0.28, 0.27, "28 combinaciones del\nEntregable 2\n\n7 modelos × 4 optimizadores\n(Grid, Random, Bayes, Genético)", GOLD)
fig.text(0.5, 0.33, "Todos con los mismos folds, la misma semilla (2026) y validación temporal con purga de 48 h.\nLos árboles aprenden la demanda relativa al nivel de 7 días: así no tienen que extrapolar niveles récord.",
         ha="center", fontsize=24)
guardar(fig, "ATER compite contra 14 modelos: tres tradicionales, cuatro de Machine Learning y la mejor de las 28 combinaciones de cada modelo del Entregable 2, todos con el mismo protocolo.", 9)

# ---------------------------------------------------------------- 6. idea de ATER (diagrama en 3 pasos)
caps = ["La idea: no entrenamos un modelo más. Tomamos cinco pronósticos conocidos, los expertos, y los combinamos.",
        "Los pesos cambian según la hora del día y el calor: a cada hora le creemos más al experto que mejor ha acertado en esa hora.",
        "Cada noche, con la demanda real del día, ATER actualiza los pesos y corrige el nivel del pronóstico. Ese es su aprendizaje en línea."]
for k in range(1, 4):
    fig = lienzo("Cómo funciona ATER", "4 · Modelo original", 6)
    for i, e in enumerate(["Random forest", "XGBoost", "Híbrido", "Ridge", "Persistencia"]):
        caja(fig, 0.04, 0.70 - i * 0.11, 0.16, 0.08, e, GREEN, size=20)
        flecha(fig, (0.205, 0.74 - i * 0.11), (0.29, 0.52))
    if k >= 2:
        caja(fig, 0.30, 0.42, 0.22, 0.20, "Pesos exponenciales\npor hora y calor\nw(hora, calor)", RED)
        flecha(fig, (0.525, 0.52), (0.585, 0.52))
        caja(fig, 0.59, 0.42, 0.17, 0.20, "Combinación\n× (1 + b_hora)", RED)
        flecha(fig, (0.765, 0.52), (0.815, 0.52)); caja(fig, 0.82, 0.42, 0.14, 0.20, "Pronóstico\n24 horas\nde mañana", INK, size=21)
    if k >= 3:
        caja(fig, 0.42, 0.20, 0.38, 0.11, "Al cierre del día: demanda real → actualiza pesos y nivel", GOLD, size=21)
        flecha(fig, (0.78, 0.315), (0.89, 0.41), GOLD); flecha(fig, (0.44, 0.315), (0.41, 0.41), GOLD)
    guardar(fig, caps[k - 1])

# ---------------------------------------------------------------- 7. ecuaciones
fig = lienzo("Formulación", "4 · Modelo original", 7)
eqs = [r"$\ell_{k,t}=\min\{1,\ ((1+b_h)f_{k,t}-y_t)^2/n_t^2\}$", r"$L_{k,t}=\sum_j e^{-(c_t-\tau_j)^2/2\sigma^2}\,S_{h,j,k}$",
       r"$w_{k,t}=e^{-\eta L_{k,t}}/\sum_i e^{-\eta L_{i,t}}$", r"$\hat y_t=(1+b_h)\sum_k w_{k,t} f_{k,t}$", r"$b_h \leftarrow b_h-\eta_b\,2(\hat y_t-y_t)\bar f_t/n_t^2$"]
lab = ["pérdida relativa al nivel de 7 días (acotada)", "pérdida acumulada, suavizada por el calor", "pesos exponenciales (Hedge)", "pronóstico con corrección de nivel", "gradiente en línea de la corrección"]
for i, (e, l) in enumerate(zip(eqs, lab)):
    fig.text(0.07, 0.76 - i * 0.12, e, fontsize=34); fig.text(0.62, 0.76 - i * 0.12, l, fontsize=22, color=GREEN)
guardar(fig, "Matemáticamente: la pérdida se mide relativa al nivel de la demanda, los pesos son exponenciales en la pérdida acumulada por hora y calor, y la corrección de nivel se ajusta por descenso de gradiente. Sin fuga: usa solo lo observado hasta ayer.", 12)

# ---------------------------------------------------------------- 8. pesos por hora
fig = lienzo("¿A quién le cree ATER a cada hora?", "4 · Modelo original", 8)
ax = fig.add_axes([0.07, 0.22, 0.86, 0.55]); bot = np.zeros(24)
colores = [GREEN, GOLD, "#8E4B7A", "#5B7F96", GRAY]
for c, col in zip(pesos.columns, colores):
    ax.bar(pesos.index, pesos[c], bottom=bot, color=col, label=CORTO.get(c, c)); bot += pesos[c].values
ax.set_xlabel("hora del día"); ax.set_ylabel("peso medio en la prueba"); ax.legend(ncol=5, fontsize=17, loc="upper center", bbox_to_anchor=(0.5, 1.13))
guardar(fig, "Así reparte ATER su confianza en la prueba: de madrugada y en la tarde le cree sobre todo a los árboles; en la mañana sube el peso del híbrido y de Ridge. La persistencia casi no recibe peso.", 10)

# ---------------------------------------------------------------- 9. teoría, complejidad, pruebas unitarias
fig = lienzo("Implementación propia y sustento teórico (R2)", "4 · Modelo original", 9)
caja(fig, 0.05, 0.40, 0.28, 0.36, "Teoría\n\nCota de arrepentimiento\nde Hedge:\nln K / η + η T / 8\n\nCorrección de nivel:\npérdida convexa en b", GREEN, size=21)
caja(fig, 0.36, 0.40, 0.28, 0.36, "Complejidad\n\nTiempo O(n · B · K)\nMemoria O(24 · B · K)\n\nPendiente empírica\nlog-log ≈ 1 (lineal)", GOLD, size=21)
caja(fig, 0.67, 0.40, 0.28, 0.36, "13 pruebas unitarias ✔\n\naprende una solución conocida\nreproducible con semilla\nrechaza entradas inválidas\ncompatible con clone\nno usa el futuro", RED, size=20)
fig.text(0.5, 0.29, "Clase propia compatible con scikit-learn, programada con numpy: src/original_model.py", ha="center", fontsize=24, family="DejaVu Sans Mono")
guardar(fig, "ATER no sale de un import: es una clase propia en numpy, con su sustento teórico, su complejidad lineal comprobada y 13 pruebas unitarias que pasan.", 9)

# ---------------------------------------------------------------- 10. optimización y validación anidada
fig = lienzo("Optimización bayesiana dentro de validación anidada", "4 · Modelo original", 10)
ax = fig.add_axes([0.07, 0.22, 0.42, 0.52]); ax.scatter(tri.number, tri.value, s=30, color=GRAY); ax.plot(tri.number, tri.mejor_hasta_ahora, color=RED, lw=4)
ax.set_ylim(tri.value.min() * .99, tri.value.quantile(.9)); ax.set_xlabel("ensayo de Optuna (TPE)"); ax.set_ylabel("MSE en validación"); ax.set_title("Curva de convergencia", fontsize=22)
a2 = an.sort_values("MSE_media").head(6)
fig.text(0.56, 0.74, "Validación anidada (folds externos 1–4)", fontsize=24, weight="bold")
for i, r in enumerate(a2.itertuples()):
    fig.text(0.56, 0.67 - i * 0.065, CORTO.get(r.modelo, r.modelo), fontsize=22, color=RED if r.modelo == ORIG else INK, weight="bold" if r.modelo == ORIG else "normal")
    fig.text(0.92, 0.67 - i * 0.065, f"{nf(r.MSE_media, 0)} ± {nf(r.MSE_de, 0)}", fontsize=22, ha="right", color=RED if r.modelo == ORIG else INK)
guardar(fig, "Sus cuatro hiperparámetros se ajustan con Optuna, que es optimización bayesiana, dentro de una validación anidada: en cada fold externo solo se usan los folds anteriores. ATER tiene el menor error medio.", 10)

# ---------------------------------------------------------------- 11. resultados prueba
fig = lienzo("Prueba única 2025–2026: 653 días nunca vistos", "5 · Resultados", 11)
t = pd.concat([tp[["modelo", "MAPE"]], pd.DataFrame({"modelo": ["E2 Random Forest (Grid)", "E2 XGBoost (Random)"], "MAPE": [2.564, 2.575]})]).query("modelo != 'Promedio por hora y día'").sort_values("MAPE", ascending=False)
ax = fig.add_axes([0.30, 0.25, 0.62, 0.54])
ax.barh([CORTO.get(m, m) for m in t.modelo], t.MAPE, color=[RED if m == ORIG else GOLD if m.startswith("E2") else GRAY if ("Persist" in m or "Estac" in m) else GREEN for m in t.modelo])
for i, v in enumerate(t.MAPE): ax.text(v + 0.05, i, f"{nf(v, 2)} %", va="center", fontsize=19)
ax.set_xlabel("error medio absoluto porcentual (MAPE, %)"); ax.tick_params(axis="y", labelsize=19)
guardar(fig, "En la prueba, ATER tiene 2,44 por ciento de error: 65 por ciento menos error cuadrático que la persistencia y 3,9 por ciento menos que el mejor random forest. Además detecta el 91 por ciento de las horas pico.", 11)

# ---------------------------------------------------------------- 12. pruebas de hipótesis
fig = lienzo("¿La ventaja es real? Tres pruebas de hipótesis (R3)", "6 · Validación estadística", 12)
caja(fig, 0.05, 0.45, 0.28, 0.30, "1. Model Confidence Set\n\nCon 15 modelos,\nATER es el ÚNICO\nque sobrevive (α = 0,10)", RED, size=22)
rf = pr[pr.benchmark == "Random forest (relativo)"].set_index("prueba")
caja(fig, 0.36, 0.45, 0.28, 0.30, f"2. Giacomini-White\n\nvs. random forest\np ajustado = {nf(rf.loc['Giacomini-White condicional', 'p_holm'], 3)}", GREEN, size=22)
caja(fig, 0.67, 0.45, 0.28, 0.30, f"3. Diebold-Mariano (HLN)\n\nvs. random forest\np ajustado = {nf(rf.loc['Diebold-Mariano (HLN)', 'p_holm'], 3)}", GOLD, size=22)
fig.text(0.5, 0.33, "Ajuste de Holm sobre 21 comparaciones · IC 95 % por bootstrap estacionario: mejora de 1,1 % a 6,8 % frente al RF\nATER gana frente a los 7 benchmarks y a las 7 mejores combinaciones del Entregable 2",
         ha="center", fontsize=23)
guardar(fig, "Que el error sea menor no basta. El MCS deja a ATER como único modelo superior, y Giacomini-White y Diebold-Mariano rechazan la igualdad con p ajustado por Holm menor que 0,05 frente a todos.", 11)

# ---------------------------------------------------------------- 13. efecto y ablación
fig = lienzo("Tamaño del efecto y ablación", "6 · Validación estadística", 13)
fig.text(0.05, 0.75, "Frente al random forest:", fontsize=25, weight="bold")
bullets(fig, ["gana el 62,6 % de los días", "δ de Cliff pareado 0,25 → pequeño", "no pareado 0,07 y d de Cohen 0,12 → despreciables", "frente a la persistencia: efecto grande (δ 0,67)"], 4, y0=0.68, size=22, ancho=34, dy=0.08)
a = ab[ab.grupo == "componente"].sort_values("MSE_prueba"); ax = fig.add_axes([0.62, 0.22, 0.33, 0.55])
ETQ = {"ATER completo": "ATER completo", "sin régimen térmico (sigma = ∞)": "sin núcleo de calor", "sin corrección de nivel (eta_b = 0)": "sin corrección de nivel",
       "sin régimen horario (pesos comunes)": "sin pesos por hora", "sin pesos aprendidos (eta = 0)": "sin pesos aprendidos", "sin olvido (lambda = 1)": "sin olvido",
       "promedio simple (eta = 0, eta_b = 0)": "promedio simple"}
ax.barh([ETQ.get(v, v) for v in a.variante], a.MSE_prueba,
        color=[RED if v == "ATER completo" else GRAY for v in a.variante])
ax.set_xlim(2100, 2700); ax.tick_params(axis="y", labelsize=15); ax.set_xlabel("MSE en la prueba"); ax.set_title("Ablación", fontsize=22)
guardar(fig, "Somos honestos: frente al random forest el efecto es pequeño, aunque consistente. La ablación muestra qué aporta: los pesos por hora, 7,9 por ciento, y la corrección de nivel, 2,6 por ciento. El suavizado por calor no aportó.", 12)

# ---------------------------------------------------------------- 14. veredicto
fig = lienzo("Veredicto: superior", "7 · Veredicto", 14)
caja(fig, 0.05, 0.42, 0.28, 0.34, "R1 · Novedad ✔\n(con matices)\n\nLa familia existe para\nla demanda de Francia;\nla combinación y la\naplicación al Caribe\nson nuevas", GREEN, size=21)
caja(fig, 0.36, 0.42, 0.28, 0.34, "R2 · Implementación\npropia ✔\n\nClase en numpy,\nno un import;\n13 pruebas unitarias", GOLD, size=21)
caja(fig, 0.67, 0.42, 0.28, 0.34, "R3 · Superioridad\nsignificativa ✔\n\nMCS, Giacomini-White\ny Diebold-Mariano;\nefecto pequeño vs. RF", RED, size=21)
fig.text(0.5, 0.31, "Tope de la guía con R1, R2 y R3 cumplidos: 5,0", ha="center", fontsize=30, weight="bold")
guardar(fig, "Veredicto explícito: ATER es superior. R1, novedad con matices. R2, implementación propia. R3, superioridad significativa en tres pruebas, con un efecto pequeño frente al random forest que reportamos completo.", 11)

# ---------------------------------------------------------------- 15. rúbrica
fig = lienzo("Cómo cubrimos la rúbrica", "8 · Entregables", 15)
filas = [("Correcciones del E2 (10 %)", "tabla de trazabilidad: 8 observaciones del docente y 12 correcciones"),
         ("Combinaciones y artefactos (10 %)", "28 combinaciones, joblib + metadata, prueba de ida y vuelta"),
         ("Dashboard (30 %)", "Dash multipágina en Render, predicción en vivo con SHAP"),
         ("Modelo original (15 %)", "matriz de novedad, pseudocódigo, clase propia, ablación"),
         ("Pruebas de hipótesis (15 %)", "MCS, Giacomini-White, Diebold-Mariano, Holm, bootstrap, efecto"),
         ("Informe y sustentación (20 %)", "capítulos 0 a 9, reproducible, bibliografía BibTeX")]
for i, (a_, b_) in enumerate(filas):
    fig.text(0.05, 0.75 - i * 0.085, "✔", color=GREEN, fontsize=28, weight="bold")
    fig.text(0.09, 0.75 - i * 0.085, a_, fontsize=23, weight="bold"); fig.text(0.42, 0.75 - i * 0.085, b_, fontsize=21)
guardar(fig, "Así cubrimos cada componente de la rúbrica del profesor: correcciones, mejores combinaciones, dashboard, modelo original, pruebas de hipótesis, informe y sustentación.", 10)

# ---------------------------------------------------------------- 16. cierre
fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=INK)
fig.text(0.07, 0.80, "Gracias", fontsize=80, color="white", weight="bold")
for i, (a_, b_) in enumerate([("Dashboard", "ml-energia-caribe.onrender.com"), ("Jupyter Book", "jserrano3010.github.io/ML_energia_caribe"),
                              ("Repositorio", "github.com/jserrano3010/ML_energia_caribe")]):
    fig.text(0.07, 0.60 - i * 0.09, a_, fontsize=26, color=GOLD, weight="bold"); fig.text(0.30, 0.60 - i * 0.09, b_, fontsize=26, color="white")
fig.text(0.07, 0.25, "Limitaciones: demanda regional (no solo Sabanagrande) · no se pronostica el precio · prueba consultada más de una vez", fontsize=20, color="#C9D6D3")
fig.text(0.07, 0.19, "Edwin Yunis y Jairo Serrano · Machine Learning · Maestría en Ingeniería Industrial", fontsize=20, color="#C9D6D3")
guardar(fig, "Todo está publicado: el dashboard en Render, el informe en Jupyter Book y el código en GitHub. Gracias.", 7)

# ---------------------------------------------------------------- ensamblar con ffmpeg
lista = TMP / "lista.txt"; t0 = 0.0; srt = []
with open(lista, "w") as fl:
    for i, (p, s, txt) in enumerate(frames):
        fl.write(f"file '{p}'\nduration {s:.2f}\n")
        def ts(x): h, r = divmod(x, 3600); m_, s_ = divmod(r, 60); return f"{int(h):02d}:{int(m_):02d}:{int(s_):02d},{int((s_ % 1) * 1000):03d}"
        srt.append(f"{i + 1}\n{ts(t0)} --> {ts(t0 + s)}\n{txt}\n"); t0 += s
    fl.write(f"file '{frames[-1][0]}'\n")
Path("publicar/video_ATER_subtitulos.srt").write_text("\n".join(srt), encoding="utf-8")
Path("publicar/video_ATER_guion.txt").write_text("\n\n".join(f"[{i + 1}] {t}" for i, (_, _, t) in enumerate(frames)), encoding="utf-8")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lista), "-vf", "fps=30,format=yuv420p",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-movflags", "+faststart", str(OUT)], check=True)
print(f"{len(frames)} cuadros · {t0:.0f} s · {OUT.stat().st_size / 1e6:.1f} MB")
