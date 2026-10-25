"""Sección 3: Modelos: tabla de líderes, predicción en vivo con SHAP local, diagnóstico, comparación estadística y modelo original."""
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html, no_update

import data as D
import plots as G

dash.register_page(__name__, path="/modelos", name="Modelos", order=2)

tp = D.csv("orig_prueba.csv")
an = D.csv("orig_validacion_anidada.csv").set_index("modelo")
cfg = D.js("eleccion.json")["configuraciones"]; hp = D.js("orig_hiperparametros.json")["hiperparametros"]
ver = D.js("orig_veredicto.json"); mcs = D.csv("orig_mcs.csv"); pr = D.csv("orig_pruebas_hipotesis.csv")
ab = D.csv("orig_ablacion.csv"); tri = D.csv("orig_optuna_trials.csv"); imp = D.csv("shap_importancia_global.csv", index_col=0)
q = D.csv("residuos_cuantiles.csv"); pesos = D.csv("pesos_por_hora.csv", index_col=0)
PT = D.pred_prueba(); XP = D.predictoras_prueba(); META = D.metadata()
RANGOS = META["columnas_entrada"]
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
D.modelo()          # carga el modelo una sola vez al iniciar

import json
import tablas as T

# ---------------- tabla de líderes: ATER, los 7 benchmarks del proyecto y los 7 modelos vistos en clase (E2)
E2C = D.csv("e2_combinaciones.csv"); E2M = D.csv("e2_mejores.csv"); E2A = D.csv("e2_anytime.csv", index_col=0)
mcs_todos = D.csv("e2_mcs_todos.csv").set_index("modelo")
grupo = lambda m: "Original" if m == D.ORIG else ("Tradicional" if m in D.TRAD else "Benchmark del proyecto")
fmt_cfg = lambda d: ", ".join(f"{k} = {v:.3g}" if isinstance(v, float) else f"{k} = {v}" for k, v in d.items())
mse_txt = lambda a, b: f"{a:,.0f} ± {b:,.0f}".replace(",", ".")
lider = pd.concat([
    pd.DataFrame({"modelo": [D.CORTO[m] for m in tp.modelo], "grupo": [grupo(m) for m in tp.modelo],
                  "optimizador": ["Bayesiano (Optuna TPE)" if m == D.ORIG else ("—" if m in D.TRAD else "Grid (validación)") for m in tp.modelo],
                  "cfg": [fmt_cfg(hp) if m == D.ORIG else cfg.get(m, "sin hiperparámetros") for m in tp.modelo],
                  "anidada": [mse_txt(an.loc[m, "MSE_media"], an.loc[m, "MSE_de"]) for m in tp.modelo],
                  "mse": tp.MSE.round(0), "mape": tp.MAPE.round(2), "auc": tp.AUC_pico.round(3), "inf": tp.inferencia_s.round(3),
                  "mcs": [T.si_no(mcs_todos.en_conjunto_10pct.get(m, False)) for m in tp.modelo]}),
    pd.DataFrame({"modelo": E2M.modelo + " (E2)", "grupo": "Visto en clase (E2)", "optimizador": E2M.optimizador,
                  "cfg": [fmt_cfg(json.loads(h)) for h in E2M.hiperparametros],
                  "anidada": [mse_txt(a, b) for a, b in zip(E2M.MSE_media, E2M.MSE_de)],
                  "mse": E2M.MSE_prueba.round(0), "mape": E2M.MAPE_prueba.round(2), "auc": E2M.AUC_pico.round(3),
                  "inf": E2M.inferencia_prueba_s.round(3), "mcs": [T.si_no(v) for v in E2M.en_conjunto_10pct]})],
    ignore_index=True).sort_values("mse").reset_index(drop=True)
lider.insert(0, "puesto", range(1, len(lider) + 1))
COLS_LIDER = [T.col("puesto", "#", 0), T.col("modelo"), T.col("grupo"), T.col("optimizador"),
              T.col("anidada", "MSE validación anidada (media ± de)"), T.col("mse", "MSE prueba (MWh²)", 0),
              T.col("mape", "MAPE prueba (%)", 2), T.col("auc", "AUC horas pico", 3), T.col("inf", "Inferencia 15.672 h (s)", 3),
              T.col("mcs", "En el MCS (10 %)")]
TT_LIDER = [{"modelo": {"value": f"**Configuración:** {c}", "type": "markdown"}} for c in lider.cfg]
EXTRA_LIDER = T.barras(lider, "mse", maximo=float(lider.mse.nlargest(2).iloc[-1]) * 1.1) + T.mejor(lider, "mape") + T.mejor(lider, "auc", minimo=False)

# ---------------- Entregable 2: matriz 7 modelos x 4 optimizadores (MSE externo medio) y detalle
OPTS = ["Grid Search", "Random Search", "Bayesiano (Optuna TPE)", "Genético"]
mat = E2C.pivot(index="modelo", columns="optimizador", values="MSE_media").round(0)[OPTS]
mat["mejor"] = mat[OPTS].idxmin(axis=1)
mat = mat.assign(_m=mat[OPTS].min(axis=1)).sort_values("_m").drop(columns="_m").reset_index()
mat.columns.name = None
COLS_MAT = [T.col("modelo", "Modelo")] + [T.col(o, o, 0) for o in OPTS] + [T.col("mejor", "Mejor optimizador")]
EXTRA_MAT = [{"if": {"filter_query": f'{{modelo}} = "{r.modelo}"', "column_id": r.mejor},
              "backgroundColor": T.VERDE_CLARO, "color": G.VERDE, "fontWeight": 700} for r in mat.itertuples()]
det = E2C.assign(anidada=[mse_txt(a, b) for a, b in zip(E2C.MSE_media, E2C.MSE_de)], mape=E2C.MAPE_media.round(2),
                 busq=(E2C.tiempo_busqueda_s).round(1), ent=E2C.tiempo_entrenamiento_s.round(2), inf=E2C.tiempo_inferencia_s.round(3))
COLS_DET = [T.col("modelo", "Modelo"), T.col("optimizador", "Optimizador"), T.col("anidada", "MSE externo (media ± de)"),
            T.col("mape", "MAPE (%)", 2), T.col("rango_en_modelo", "Puesto en su modelo", 0), T.col("evaluaciones", "Evaluaciones", 0),
            T.col("busq", "Búsqueda (s)", 1), T.col("ent", "Entrenamiento por fold (s)", 2), T.col("inf", "Inferencia por fold (s)", 3)]

# ---------------- tiempos de cómputo (misma máquina, mismos datos: src/tiempos_computo.py)
TC = D.csv("tiempos_computo.csv")
TC["nombre"] = [D.CORTO.get(m, m) for m in TC.modelo]
TC["dia_ms"] = (TC.pronostico_dia_s * 1000).round(1)
tc_tab = TC.assign(busq=TC.busqueda_s.round(1), ent=TC.entrenamiento_s.round(2), inf=TC.inferencia_prueba_s.round(3)).sort_values("ent", ascending=False)
tc_tab = tc_tab.assign(modelo=tc_tab.nombre)[["modelo", "grupo", "optimizador", "busq", "ent", "inf", "dia_ms"]]
COLS_TC = [T.col("modelo", "Modelo"), T.col("grupo", "Grupo"), T.col("optimizador", "Optimizador"),
           T.col("busq", "Búsqueda de hiperparámetros (s)", 1), T.col("ent", "Entrenamiento final (s)", 2),
           T.col("inf", "Inferencia 15.672 h (s)", 3), T.col("dia_ms", "Pronóstico de 1 día (ms)", 1)]
EXTRA_TC = T.barras(tc_tab, "ent", color="#F3D9D5")
ta = TC.set_index("modelo").loc[D.ORIG]
g_tc = TC[TC.grupo != "Tradicional"].sort_values("entrenamiento_s")
f_tc = go.Figure([go.Bar(y=g_tc.nombre, x=g_tc.entrenamiento_s, orientation="h", name="Entrenamiento final",
                         marker_color=[G.ROJO if m == D.ORIG else G.TINTA for m in g_tc.modelo], hovertemplate="%{y}: %{x:.2f} s<extra>entrenamiento</extra>"),
                  go.Bar(y=g_tc.nombre, x=g_tc.busqueda_s, orientation="h", name="Búsqueda de hiperparámetros", marker_color=G.ORO,
                         hovertemplate="%{y}: %{x:.1f} s<extra>búsqueda</extra>"),
                  go.Bar(y=g_tc.nombre, x=g_tc.inferencia_prueba_s, orientation="h", name="Inferencia 15.672 h", marker_color=G.GRIS,
                         hovertemplate="%{y}: %{x:.3f} s<extra>inferencia</extra>")])
f_tc.update_layout(barmode="group", title="Tiempos de cómputo por modelo (segundos, escala logarítmica)", xaxis_type="log",
                   xaxis_title="segundos", height=640, yaxis=dict(automargin=True), legend=dict(orientation="h", y=-0.12))

f_any = go.Figure([go.Scatter(x=E2A.index, y=E2A[c], mode="lines+markers", name=c) for c in E2A.columns])
f_any.update_layout(title="Desempeño en cualquier momento: mejor MSE interno hasta la evaluación t (relativo al mejor final)",
                    xaxis_title="evaluación", yaxis_title="MSE relativo (1 = mejor encontrado)", height=360)

EDIT = [("calor", "Índice de calor ayer, misma hora (°C)"), ("T2M", "Temperatura ayer (°C)"), ("RH2M", "Humedad ayer (%)"),
        ("lag24", "Demanda ayer, misma hora (MWh)"), ("media_7d", "Media de los últimos 7 días (MWh)"), ("max_dia_ant", "Máximo de ayer (MWh)")]


def campo(col, etiqueta):
    r = RANGOS[col]
    return dbc.Col([dbc.Label(etiqueta, html_for=f"in-{col}", className="small"),
                    dbc.Input(id=f"in-{col}", type="number", min=round(r["min"], 1), max=round(r["max"] * 1.15, 1), step=0.1)], md=4, className="mb-2")


# ---------------- figuras estáticas
f_conv = go.Figure([go.Scatter(x=tri.number, y=tri.value, mode="markers", marker=dict(color=G.GRIS, size=6), name="ensayo"),
                    go.Scatter(x=tri.number, y=tri.mejor_hasta_ahora, mode="lines", line=dict(color=G.ROJO, width=3), name="mejor hasta ahora")])
f_conv.update_layout(title="Curva de convergencia de Optuna (TPE)", xaxis_title="ensayo", yaxis_title="MSE medio en validación (MWh²)",
                     yaxis_range=[tri.value.min() * .99, tri.value.quantile(.9)], height=330)
a = ab.sort_values("MSE_prueba")
f_abl = go.Figure(go.Bar(x=a.MSE_prueba, y=a.variante, orientation="h", text=[f"{v:+.1f} %" for v in a["cambio_MSE_vs_completo_%"]], textposition="outside",
                         marker_color=[G.ROJO if v == "ATER completo" else {"componente": G.TINTA, "experto": G.ORO}.get(g, G.GRIS) for v, g in zip(a.variante, a.grupo)]))
f_abl.update_layout(title="Ablación: MSE en la prueba (texto = cambio frente a ATER completo)", xaxis_range=[a.MSE_prueba.min() * .93, a.MSE_prueba.max() * 1.04],
                    yaxis=dict(autorange="reversed", automargin=True), height=600, xaxis_title="MWh²", margin=dict(l=320))
ef = pr.drop_duplicates("benchmark").sort_values("mejora_MSE_pct", ascending=False)
f_ic = go.Figure(go.Scatter(x=ef.mejora_MSE_pct, y=[D.CORTO[b] for b in ef.benchmark], mode="markers", marker=dict(color=G.ROJO, size=10),
                            error_x=dict(type="data", symmetric=False, array=ef.IC95_hi - ef.mejora_MSE_pct, arrayminus=ef.mejora_MSE_pct - ef.IC95_lo, color=G.ROJO)))
f_ic.add_vline(x=0, line_color=G.TINTA); f_ic.update_layout(title="Reducción del MSE de ATER frente a cada benchmark (IC 95 %, bootstrap estacionario)",
                                                           xaxis_title="% de reducción del MSE", height=360, yaxis=dict(automargin=True))
f_pesos = go.Figure([go.Bar(x=pesos.index, y=pesos[c], name=D.CORTO[c], marker_color=G.color_modelo(c) if c != "Persistencia (misma hora de ayer)" else "#D5DBD8") for c in pesos.columns])
f_pesos.update_layout(barmode="stack", title="¿A quién le cree ATER a cada hora? (peso medio en la prueba)", xaxis_title="hora", yaxis_title="peso", height=360)

# ---------------- pruebas de hipótesis: resumen por benchmark (7 filas) y detalle (21 comparaciones)
PRU = {"Diebold-Mariano (HLN)": "dm", "Giacomini-White condicional": "gw", "Wilcoxon de rangos con signo": "wx"}
H0 = {"Diebold-Mariano (HLN)": "E[dₜ] = 0: igual error cuadrático esperado",
      "Giacomini-White condicional": "E[dₜ₊₁ | info. hasta t] = 0: igual capacidad predictiva",
      "Wilcoxon de rangos con signo": "mediana de dₜ = 0"}
res = pr.drop_duplicates("benchmark").set_index("benchmark")
piv = pr.pivot(index="benchmark", columns="prueba", values="p_holm")
resumen = pd.DataFrame({"benchmark": [D.CORTO[b] for b in res.index], "mejora": res.mejora_MSE_pct.round(1).values,
                        "ic": [f"{a:.1f} – {b:.1f}".replace(".", ",") for a, b in zip(res.IC95_lo, res.IC95_hi)],
                        **{v: [T.p_texto(piv.loc[b, k]) for b in res.index] for k, v in PRU.items()},
                        "cliff": res.delta_cliff_pareado.round(2).values, "gana": res.dias_gana_ATER_pct.round(1).values,
                        "veredicto": [T.si_no((piv.loc[b] < 0.05).all()) for b in res.index]}).sort_values("mejora").reset_index(drop=True)
resumen["veredicto"] = resumen.veredicto.str.replace("sí", "ATER mejor").str.replace("no", "no significativo")
COLS_RES = [T.col("benchmark", "ATER frente a…"), T.col("mejora", "Reducción del MSE (%)", 1), T.col("ic", "IC 95 % (bootstrap)"),
            T.col("dm", "Diebold-Mariano · p Holm"), T.col("gw", "Giacomini-White · p Holm"), T.col("wx", "Wilcoxon · p Holm"),
            T.col("cliff", "δ de Cliff pareado", 2), T.col("gana", "Días que gana ATER (%)", 1), T.col("veredicto", "Veredicto")]
EXTRA_RES = ([{"if": {"filter_query": f'{{benchmark}} = "{D.CORTO[b]}"', "column_id": v}, "color": G.VERDE, "fontWeight": 700}
              for b in res.index for k, v in PRU.items() if piv.loc[b, k] < 0.05]
             + T.barras(resumen, "mejora", color="#F3D9D5") + T.barras(resumen, "gana", color="#D9E6DE", maximo=100.0))
detalle = pr.assign(benchmark=pr.benchmark.map(D.CORTO), h0=pr.prueba.map(H0), est=pr.estadistico.round(2),
                    pc=pr.p_crudo.map(T.p_texto), ph=pr.p_holm.map(T.p_texto))
COLS_PDET = [T.col("prueba", "Prueba"), T.col("benchmark", "Benchmark"), T.col("h0", "Hipótesis nula H₀"), T.col("supuestos", "Supuestos verificados"),
             T.col("est", "Estadístico", 2), T.col("pc", "p crudo"), T.col("ph", "p ajustado (Holm)"), T.col("conclusion", "Conclusión")]

# ---------------- Model Confidence Set con los 15 modelos
mcs = D.csv("e2_mcs_todos.csv").sort_values("p_MCS", ascending=False)
nom_mcs = lambda m: (m[3:].replace(" (", " (E2, ", 1) if m.startswith("E2 ") else D.CORTO.get(m, m))
tab_mcs = pd.DataFrame({"modelo": mcs.modelo.map(nom_mcs),
                        "grupo": ["Visto en clase (E2)" if m.startswith("E2 ") else grupo(m) for m in mcs.modelo],
                        "p": mcs.p_MCS.map(T.p_texto), "dentro": mcs.en_conjunto_10pct.map(T.si_no)})
COLS_MCS = [T.col("modelo", "Modelo"), T.col("grupo", "Grupo"), T.col("p", "p-valor del MCS"), T.col("dentro", "¿En el conjunto de los mejores (α = 0,10)?")]

PSEUDO = """para cada día d (en orden):
    para cada hora t de d:                         # pronóstico emitido al cierre de d-1
        L[k] = Σ_j exp(-(c_t-τ_j)²/2σ²) · S[h_t, j, k]
        w[k] = exp(-η L[k]) / Σ_i exp(-η L[i])
        ŷ_t  = (1 + b[h_t]) · Σ_k w[k] f[t,k]
    al cierre de d (ya se conoce y):
        S ← λ S ;  S[h_t, banda(c_t), k] += min(1, (((1+b)f[t,k] - y_t)/n_t)²)
        b[h_t] ← b[h_t] - η_b · 2(ŷ_t - y_t)·f̄_t / n_t²"""

GANA = f"{pr.dias_gana_ATER_pct.iloc[12]:.1f}".replace(".", ",")

layout = html.Div([
    html.H2("Modelos"),
    dbc.Alert([html.B("Veredicto: "), f"ATER es estadísticamente superior a los 7 benchmarks (MCS, Giacomini-White y Diebold-Mariano con p ajustado por Holm < 0,05). "
               f"Frente al mejor benchmark (random forest relativo) reduce el MSE 3,9 % y gana el {GANA} % de los días: "
               "un efecto pequeño (δ de Cliff pareado 0,25) pero consistente."], color="success"),
    dbc.Card(dbc.CardBody([html.H4("Tabla de líderes"),
        html.P("ATER, los 7 benchmarks del proyecto y la mejor combinación (modelo + optimizador) de cada uno de los 7 modelos vistos en clase (Entregable 2), "
               "ordenados por el MSE de la prueba única 2025–2026. La barra muestra el MSE; en verde, el mejor MAPE y el mejor AUC. "
               "Pase el cursor por el nombre del modelo para ver su configuración. Se puede ordenar y filtrar.", className="small text-muted"),
        T.tabla(lider, COLS_LIDER, extra=EXTRA_LIDER, tooltip=TT_LIDER, ancho_texto={"modelo": 190, "anidada": 150})]), className="mb-3"),

    dbc.Card(dbc.CardBody([html.H4("Tiempos de cómputo"),
        html.P("Todos los modelos medidos en la misma máquina y con los mismos datos (src/tiempos_computo.py): entrenamiento final con las 36.552 horas de "
               "entrenamiento, inferencia sobre las 15.672 horas de la prueba y pronóstico de un día (24 horas, mediana de 5 repeticiones).", className="small text-muted"),
        dbc.Row([dbc.Col(dbc.Card(dbc.CardBody([html.Div(f"{ta.entrenamiento_s:.0f} s", className="fs-3 fw-bold text-danger"),
                                                html.Div("entrenar ATER completo (incluye sus 5 expertos)", className="small text-muted")])), md=3),
                 dbc.Col(dbc.Card(dbc.CardBody([html.Div(f"{ta.busqueda_s:.0f} s", className="fs-3 fw-bold text-danger"),
                                                html.Div("búsqueda de η, λ, σ y η_b con Optuna (4 × 60 + 80 ensayos)", className="small text-muted")])), md=3),
                 dbc.Col(dbc.Card(dbc.CardBody([html.Div(f"{ta.pronostico_dia_s * 1000:.0f} ms".replace(".", ","), className="fs-3 fw-bold text-danger"),
                                                html.Div("pronosticar las 24 horas del día siguiente", className="small text-muted")])), md=3),
                 dbc.Col(dbc.Card(dbc.CardBody([html.Div(f"{ta.inferencia_prueba_s:.2f} s".replace(".", ","), className="fs-3 fw-bold text-danger"),
                                                html.Div("pronosticar los 653 días de la prueba", className="small text-muted")])), md=3)], className="mb-3 g-2"),
        dcc.Graph(figure=f_tc, config={"displayModeBar": False}),
        html.P("Lectura: casi todo el costo de ATER es entrenar a sus 5 expertos (el random forest relativo solo toma 31 s); el núcleo de combinación es una "
               "sola pasada y tarda décimas de segundo. Su búsqueda de hiperparámetros (23 s) es más corta que la del random forest (147 s) o el XGBoost (89 s) "
               "del Entregable 2 porque cada ensayo solo recombina pronósticos ya calculados; no incluye la validación de los expertos, que se reutiliza de los "
               "benchmarks. Las búsquedas del Entregable 2 son las registradas en su validación anidada.", className="small"),
        T.tabla(tc_tab, COLS_TC, extra=EXTRA_TC, filtro=False, ancho_texto={"modelo": 190})]), className="mb-3"),

    dbc.Card(dbc.CardBody([html.H4("Entregable 2: las 28 combinaciones (7 modelos × 4 optimizadores)"),
        html.P("KNN, Ridge, Lasso, Decision Tree, Random Forest, XGBoost y SVR, cada uno optimizado con Grid Search, Random Search, Bayesiano (Optuna TPE) y genético, "
               "con el mismo presupuesto (12 evaluaciones por fold) y los mismos folds externos que ATER. Cada celda es el MSE medio del bucle externo (MWh²); "
               "en verde, el mejor optimizador de cada modelo.", className="small text-muted"),
        T.tabla(mat, COLS_MAT, extra=EXTRA_MAT, ater_col=None, filtro=False),
        dbc.Accordion([dbc.AccordionItem(T.tabla(det, COLS_DET, ater_col=None, alto="460px"),
                                         title="Ver el detalle de las 28 combinaciones (desviación, MAPE, evaluaciones y tiempos)")],
                      start_collapsed=True, className="mt-3"),
        dcc.Graph(figure=f_any, config={"displayModeBar": False})]), className="mb-3"),

    dbc.Card(dbc.CardBody([html.H4("Predicción en vivo"),
        html.P("Elija una hora del periodo de prueba: el formulario se llena con los datos reales que se conocían al emitir el pronóstico. "
               "Puede cambiar el calor o la demanda reciente para ver cómo responde el modelo.", className="small text-muted"),
        dbc.Row([dbc.Col([dbc.Label("Fecha", className="small"), dcc.DatePickerSingle(id="in-fecha", min_date_allowed=XP.index.min().date(), max_date_allowed=XP.index.max().date(),
                                                                                       date="2026-08-12", display_format="YYYY-MM-DD")], md=3),
                 dbc.Col([dbc.Label("Hora", className="small"), dcc.Dropdown(id="in-hora", options=list(range(24)), value=15, clearable=False)], md=2),
                 dbc.Col([dbc.Label("Modelo", className="small"), dcc.Dropdown(id="in-modelo", options=[{"label": D.CORTO[m], "value": m} for m in D.MODELOS_VIVO],
                                                                               value=D.ORIG, clearable=False)], md=4),
                 dbc.Col([dbc.Label("Festivo", className="small"), dbc.Switch(id="in-festivo", value=False)], md=3)], className="mb-2"),
        dbc.Row([campo(c, e) for c, e in EDIT]),
        dbc.Button("Pronosticar", id="btn-pred", color="danger", className="mt-1"),
        html.Div(id="out-alerta", className="mt-2"),
        dbc.Row([dbc.Col(html.Div(id="out-pred"), md=4), dbc.Col(dcc.Loading(dcc.Graph(id="out-shap", config={"displayModeBar": False})), md=8)], className="mt-2")]), className="mb-3"),

    dbc.Card(dbc.CardBody([html.H4("Diagnóstico del modelo seleccionado (prueba 2025–2026)"),
        dcc.Dropdown(id="diag-modelo", options=[{"label": D.CORTO[m], "value": m} for m in tp.modelo], value=D.ORIG, clearable=False, style={"maxWidth": 420}),
        dbc.Row([dbc.Col(dcc.Loading(dcc.Graph(id="d-real")), lg=6), dbc.Col(dcc.Graph(id="d-hist"), lg=6)]),
        dbc.Row([dbc.Col(dcc.Graph(id="d-tiempo"), lg=6), dbc.Col(dcc.Graph(id="d-acf"), lg=6)]),
        html.P(id="d-txt", className="small text-muted")]), className="mb-3"),

    dbc.Card(dbc.CardBody([html.H4("Interpretabilidad global (SHAP)"),
        dcc.Dropdown(id="shap-modelo", options=[{"label": "ATER (original)", "value": "ATER_media_abs_SHAP_MWh"}, {"label": "Random forest relativo", "value": "RF_media_abs_SHAP_MWh"}],
                     value="ATER_media_abs_SHAP_MWh", clearable=False, style={"maxWidth": 320}),
        dcc.Graph(id="shap-global", config={"displayModeBar": False}),
        html.P("Media del valor absoluto de SHAP (MWh) en 150 horas de la prueba. Los dos modelos dependen primero del nivel reciente; ATER pesa más la media de ayer "
               "y la interacción calor × hora (correlación de rangos 0,86).", className="small text-muted")]), className="mb-3"),

    dbc.Card(dbc.CardBody([html.H4("Comparación estadística"),
        html.H6("Model Confidence Set con los 15 modelos (α = 0,10, bootstrap estacionario, bloque medio de 7 días)"),
        dbc.Row([dbc.Col(T.tabla(tab_mcs, COLS_MCS, filtro=False), lg=7),
                 dbc.Col(dbc.Alert([html.B("Cómo se lee: "), "el MCS elimina uno a uno los modelos peores mientras rechaza que todos son iguales. "
                                    "Solo ATER queda en el conjunto de los mejores (p = 1). El random forest relativo y el XGBoost se eliminan con p = 0,005 "
                                    "y 0,001; todos los modelos del Entregable 2 quedan fuera."], color="light", className="small"), lg=5)], className="mb-3"),
        html.H6("Pruebas de hipótesis (21 comparaciones = 3 pruebas × 7 benchmarks, ajuste de Holm)"),
        html.P("H₀ en todas: ATER y el benchmark pronostican igual de bien (dₜ = pérdida del benchmark − pérdida de ATER). "
               "En verde, p ajustado < 0,05: se rechaza H₀. Las barras muestran la reducción del MSE y el porcentaje de días que gana ATER.", className="small text-muted"),
        T.tabla(resumen, COLS_RES, extra=EXTRA_RES, ater_col=None, filtro=False, ancho_texto={"benchmark": 170}),
        dcc.Graph(figure=f_ic, config={"displayModeBar": False}),
        dbc.Accordion([dbc.AccordionItem(T.tabla(detalle, COLS_PDET, ater_col=None, alto="480px", ancho_texto={"h0": 220, "supuestos": 260}),
                                         title="Ver las 21 comparaciones (hipótesis, supuestos, estadístico, p crudo y p ajustado)")],
                      start_collapsed=True)]), className="mb-3"),

    dbc.Card(dbc.CardBody([html.H4("Modelo original: ATER"),
        html.P(["ATER (agregación de expertos por régimen horario y térmico, con corrección de nivel en línea) combina cinco pronósticos conocidos con pesos exponenciales "
                "que dependen de la hora y del calor, y corrige cada día el nivel con la demanda que acaba de observar. Está programado con numpy en ",
                html.Code("src/original_model.py"), " y tiene 13 pruebas unitarias."]),
        dbc.Row([dbc.Col(html.Img(src=dash.get_asset_url("ater_algoritmo.png"),
                                  alt="Algoritmo ATER: pronóstico con pesos exponenciales por hora y calor, y actualización de pérdidas y corrección de nivel al cierre de cada día",
                                  style={"width": "100%", "maxWidth": "1000px", "height": "auto"}), lg=12, className="mb-3"),
                 dbc.Col(html.Ul([html.Li("Frente a Devaine et al. (2013) y Gaillard y Goude (2015): la especialización es continua (núcleo sobre el calor), no por reglas fijas, y se corrige el nivel."),
                                  html.Li("Frente al BOA suavizado (Ziel, 2022): suaviza en el calor, no entre horas."),
                                  html.Li("Frente al stacking: los pesos cambian cada día; en la ablación ATER le gana al stacking por MCO y al no negativo."),
                                  html.Li("Lo que aporta según la ablación: pesos por hora (7,9 % del MSE) y corrección de nivel (2,6 %). El núcleo de calor no aportó.")], className="small"), lg=12)]),
        dbc.Row([dbc.Col(dcc.Graph(figure=f_conv, config={"displayModeBar": False}), lg=6), dbc.Col(dcc.Graph(figure=f_pesos, config={"displayModeBar": False}), lg=6)]),
        dcc.Graph(figure=f_abl, config={"displayModeBar": False}),
        dbc.Alert([html.B("Veredicto explícito: superior. "), "R1 novedad con matices (la familia existe para la demanda de Francia; la combinación y la aplicación al Caribe son nuevas). "
                   "R2 implementación propia. R3 superioridad significativa con MCS, Giacomini-White y Diebold-Mariano; efecto frente al RF pequeño (δ de Cliff pareado 0,25; "
                   "no pareado 0,07 y d de Cohen 0,12, despreciables)."], color="danger", className="mt-2")]), className="mb-3"),
])


@callback([Output(f"in-{c}", "value") for c, _ in EDIT] + [Output("in-festivo", "value")], Input("in-fecha", "date"), Input("in-hora", "value"))
def precargar(fecha, hora):
    if not fecha or hora is None:
        return [no_update] * (len(EDIT) + 1)
    ts = pd.Timestamp(fecha) + pd.Timedelta(hours=int(hora))
    if ts not in XP.index:
        return [no_update] * (len(EDIT) + 1)
    r = XP.loc[ts]
    return [round(float(r[c]), 1) for c, _ in EDIT] + [bool(r.festivo)]


@callback(Output("out-alerta", "children"), Output("out-pred", "children"), Output("out-shap", "figure"),
          Input("btn-pred", "n_clicks"), State("in-fecha", "date"), State("in-hora", "value"), State("in-modelo", "value"), State("in-festivo", "value"),
          [State(f"in-{c}", "value") for c, _ in EDIT], prevent_initial_call=True)
def pronosticar(_, fecha, hora, modelo, festivo, *vals):
    try:
        ts = pd.Timestamp(fecha) + pd.Timedelta(hours=int(hora))
        if ts not in XP.index:
            return dbc.Alert("Elija una fecha del periodo de prueba (2025-01-01 a 2026-10-15).", color="warning"), no_update, no_update
        errores = []
        for (c, e), v in zip(EDIT, vals):
            lo, hi = RANGOS[c]["min"], RANGOS[c]["max"] * 1.15
            if v is None:
                errores.append(f"Falta: {e}.")
            elif not (lo <= float(v) <= hi):
                errores.append(f"{e}: {v} está fuera del rango válido ({lo:,.1f} a {hi:,.1f}).")
        if errores:
            return dbc.Alert(html.Ul([html.Li(x) for x in errores]), color="danger"), no_update, no_update
        x = XP.loc[[ts]].copy()
        for (c, _), v in zip(EDIT, vals):
            x[c] = float(v)
        x["festivo"] = int(festivo); x["calor_x_hora"] = x.calor * x.hora_sin
        if festivo: x["tipo_dia"] = 2
        p = float(D.predecir(x, modelo)[0])
        qq = q[(q.modelo == modelo) & (q.hora == int(hora))].iloc[0]
        real = float(PT.loc[ts, "y"]) if ts in PT.index else None
        tarjeta = dbc.Card(dbc.CardBody([html.Div(D.CORTO[modelo], className="small text-muted"),
                                         html.Div(f"{p:,.0f} MWh".replace(",", "."), className="fs-2 fw-bold text-danger"),
                                         html.Div(f"Intervalo de predicción del 90 %: {p + qq.q05:,.0f} a {p + qq.q95:,.0f} MWh".replace(",", "."), className="small"),
                                         html.Div(f"Demanda real de esa hora: {real:,.0f} MWh".replace(",", ".") if real else "", className="small mt-1"),
                                         html.Div(f"{DIAS[ts.dayofweek]} {ts:%d-%m-%Y}, {ts:%H}:00", className="small text-muted mt-1")]))
        phi = D.shap_local(x, modelo).sort_values(key=np.abs, ascending=False).head(10)
        f = go.Figure(go.Bar(x=phi.values, y=phi.index, orientation="h", marker_color=[G.ROJO if v > 0 else G.VERDE for v in phi.values],
                             hovertemplate="%{y}: %{x:+,.1f} MWh<extra></extra>"))
        f.update_layout(title="Explicación local (Shapley): cuánto sube (rojo) o baja (verde) el pronóstico cada variable", xaxis_title="MWh",
                        yaxis=dict(autorange="reversed"), height=380)
        return None, tarjeta, f
    except Exception as ex:   # nunca tumbar la app por una entrada
        return dbc.Alert(f"No se pudo calcular el pronóstico: {ex}", color="danger"), no_update, no_update


@callback(Output("d-real", "figure"), Output("d-hist", "figure"), Output("d-tiempo", "figure"), Output("d-acf", "figure"), Output("d-txt", "children"),
          Input("diag-modelo", "value"))
def diagnostico(m):
    e = PT.y - PT[m]; s = PT.sample(4000, random_state=1)
    a = go.Figure(go.Scattergl(x=s.y, y=s[m], mode="markers", marker=dict(size=4, opacity=.35, color=G.color_modelo(m))))
    lim = [PT.y.min(), PT.y.max()]; a.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=G.TINTA, dash="dash"), name="ideal"))
    a.update_layout(title="Real frente a pronosticado (4.000 horas)", xaxis_title="real (MWh)", yaxis_title="pronóstico (MWh)", showlegend=False, height=360)
    h = go.Figure(go.Histogram(x=e, nbinsx=80, marker_color=G.color_modelo(m))); h.update_layout(title="Histograma de residuos", xaxis_title="real − pronóstico (MWh)", height=360)
    ed = e.resample("D").mean()
    t = go.Figure(go.Scatter(x=ed.index, y=ed, mode="lines", line=dict(color=G.color_modelo(m), width=1)))
    t.add_hline(y=0, line_color=G.TINTA); t.update_layout(title="Residuo medio diario en el tiempo", yaxis_title="MWh", height=330)
    ac = G.fig_acf(G.acf(e.values, 72), "ACF de los residuos horarios (72 rezagos)", len(e))
    txt = (f"MAPE {100 * np.mean(np.abs(e) / PT.y):.2f} %, sesgo medio {e.mean():+.1f} MWh. Los residuos tienen memoria a 24 h: "
           "parte del error de hoy se repite mañana, que es justo lo que la corrección de nivel de ATER aprovecha.")
    return a, h, t, ac, txt


@callback(Output("shap-global", "figure"), Input("shap-modelo", "value"))
def shap_global(col):
    s = imp[col].sort_values(ascending=False).head(12)
    return G.barras_h(s, "Importancia global (media de |SHAP|, MWh)", "MWh", colores=G.ROJO if col.startswith("ATER") else G.VERDE)
