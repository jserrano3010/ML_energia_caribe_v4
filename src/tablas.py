"""Tablas con un mismo estilo para todo el dashboard (DataTable de Dash).

Formato colombiano de números (1.234,5), encabezado oscuro, filas alternas, números alineados a la derecha,
fila de ATER resaltada, mejor valor de cada columna en verde y barras de datos opcionales. Las columnas
numéricas siguen siendo numéricas, así que la tabla se puede ordenar y filtrar.
"""
from dash import dash_table
from dash.dash_table.Format import Format, Group, Scheme

from plots import GRIS, ROJO, TINTA, VERDE

FUENTE = "'Source Sans 3', 'Segoe UI', Arial, sans-serif"
CEBRA = "#F6F8F8"
ATER_FONDO = "#FBEAE8"
VERDE_CLARO = "#E3F0E8"


def num(decimales=0):
    """Formato numérico colombiano: punto de miles y coma decimal."""
    return Format(precision=decimales, scheme=Scheme.fixed, group=Group.yes, groups=3,
                  group_delimiter=".", decimal_delimiter=",")


def col(id_, nombre=None, dec=None):
    """Definición de columna. dec=None → texto; dec=n → número con n decimales."""
    c = {"id": id_, "name": nombre or id_}
    if dec is not None:
        c.update(type="numeric", format=num(dec))
    return c


def p_texto(p):
    """p-valor legible: '< 0,001' o con tres decimales y coma."""
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def si_no(v):
    return "✔ sí" if bool(v) else "✖ no"


def barras(df, columna, color="#D9E6DE", maximo=None):
    """Barras de datos dentro de la celda (proporcionales al valor)."""
    vmax = maximo or float(df[columna].max())
    reglas = []
    for v in sorted(df[columna].dropna().unique()):
        pct = min(100.0, 100 * float(v) / vmax) if vmax else 0      # valores atípicos: barra llena
        reglas.append({"if": {"filter_query": f"{{{columna}}} = {v}", "column_id": columna},
                       "background": f"linear-gradient(90deg, {color} 0%, {color} {pct:.1f}%, transparent {pct:.1f}%, transparent 100%)",
                       "paddingBottom": 2, "paddingTop": 2})
    return reglas


def mejor(df, columna, minimo=True):
    """Resalta en verde el mejor valor de una columna numérica."""
    v = df[columna].min() if minimo else df[columna].max()
    return [{"if": {"filter_query": f"{{{columna}}} = {v}", "column_id": columna},
             "color": VERDE, "fontWeight": 700, "backgroundColor": VERDE_CLARO}]


def tabla(df, columnas, *, id_=None, alto=None, ater_col="modelo", extra=None, tooltip=None,
          filtro=True, ancho_texto=None):
    """DataTable con el estilo del proyecto.

    columnas : lista de col(...); extra : reglas style_data_conditional adicionales;
    tooltip  : lista de dicts por fila (tooltip_data); ancho_texto : {col: px mínimo}.
    """
    numericas = [c["id"] for c in columnas if c.get("type") == "numeric"]
    cond = [{"if": {"row_index": "odd"}, "backgroundColor": CEBRA}]
    if ater_col and ater_col in df.columns:
        cond.append({"if": {"filter_query": f'{{{ater_col}}} contains "ATER"'},
                     "backgroundColor": ATER_FONDO, "fontWeight": 700, "borderLeft": f"4px solid {ROJO}"})
    for c in df.columns:
        if df[c].astype(str).str.startswith(("✔", "✖")).any():
            cond += [{"if": {"filter_query": f'{{{c}}} contains "✔"', "column_id": c}, "color": VERDE, "fontWeight": 600},
                     {"if": {"filter_query": f'{{{c}}} contains "✖"', "column_id": c}, "color": GRIS}]
    cond += extra or []
    kw = {"id": id_} if id_ else {}
    if tooltip:
        kw.update(tooltip_data=tooltip, tooltip_duration=None)
    if filtro:
        kw.update(filter_action="native", filter_options={"case": "insensitive", "placeholder_text": "filtrar…"})
    return dash_table.DataTable(
        data=df.to_dict("records"), columns=columnas, sort_action="native", style_as_list_view=True, **kw,
        style_table={"overflowX": "auto", "maxHeight": alto, "overflowY": "auto" if alto else "visible",
                     "borderRadius": "8px", "border": "1px solid #DDE4E4"},
        style_header={"backgroundColor": TINTA, "color": "white", "fontWeight": 600, "fontFamily": FUENTE,
                      "fontSize": 13, "whiteSpace": "normal", "height": "auto", "padding": "10px 10px",
                      "borderBottom": f"3px solid {ROJO}"},
        style_filter={"backgroundColor": "#FAFBFB", "fontSize": 12},
        style_cell={"fontFamily": FUENTE, "fontSize": 14, "padding": "8px 10px", "textAlign": "left",
                    "whiteSpace": "normal", "height": "auto", "color": TINTA, "borderBottom": "1px solid #E6ECEC",
                    "minWidth": 70},
        style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "right", "fontVariantNumeric": "tabular-nums"} for c in numericas]
        + [{"if": {"column_id": c}, "minWidth": w} for c, w in (ancho_texto or {}).items()],
        style_header_conditional=[{"if": {"column_id": c}, "textAlign": "right"} for c in numericas],
        style_data_conditional=cond,
        css=[{"selector": ".dash-table-tooltip", "rule": f"font-family: {FUENTE}; font-size: 13px; max-width: 420px;"}],
    )
