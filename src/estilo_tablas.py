"""Estilo único para las tablas del Jupyter Book.

`activar()` cambia la forma en que los notebooks muestran cualquier DataFrame: formato colombiano de números
(1.234,5), decimales según la magnitud, p-valores legibles, encabezado oscuro, filas alternas, números a la
derecha y la fila de ATER resaltada. No cambia ningún dato: solo su presentación en HTML.
"""
import numpy as np
import pandas as pd

TINTA, ROJO, CEBRA, ATER_FONDO = "#14262B", "#C0392B", "#F6F8F8", "#FBEAE8"

ESTILOS = [
    {"selector": "", "props": [("border-collapse", "collapse"), ("font-size", "0.86rem"), ("margin", "0.6rem 0 1.2rem"),
                               ("border", "1px solid #DDE4E4"), ("font-variant-numeric", "tabular-nums")]},
    {"selector": "thead th", "props": [("background", TINTA), ("color", "white"), ("font-weight", "600"), ("padding", "7px 10px"),
                                       ("border-bottom", f"3px solid {ROJO}"), ("text-align", "left"), ("vertical-align", "bottom")]},
    {"selector": "tbody td", "props": [("padding", "6px 10px"), ("border-bottom", "1px solid #E6ECEC")]},
    {"selector": "tbody th", "props": [("padding", "6px 10px"), ("border-bottom", "1px solid #E6ECEC"), ("text-align", "left"),
                                       ("font-weight", "600"), ("background", "#FAFBFB")]},
    {"selector": "tbody tr:nth-child(even) td", "props": [("background", CEBRA)]},
    {"selector": "tbody tr:hover td", "props": [("background", "#EEF3F3")]},
    {"selector": "caption", "props": [("caption-side", "top"), ("text-align", "left"), ("font-weight", "600"), ("padding", "4px 0")]},
]


def _es(x, dec):
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def formato_numero(serie):
    """Elige los decimales por magnitud de la columna."""
    nombre = str(serie.name).lower()
    if nombre.startswith(("p_", "p ", "pvalor", "p-valor")) or nombre in {"p", "p_mcs", "p_holm", "p_crudo"}:
        return lambda v: "" if pd.isna(v) else ("< 0,001" if v < 0.001 else _es(v, 3))
    m = np.nanmax(np.abs(serie.values.astype(float))) if serie.notna().any() else 0
    dec = 0 if m >= 1000 else (1 if m >= 100 else (2 if m >= 1 else 3))
    return lambda v: "" if pd.isna(v) else _es(v, dec)


def estilizar(df: pd.DataFrame):
    sty = df.style.set_table_styles(ESTILOS)
    num = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    ent = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    if num:
        sty = sty.format({c: formato_numero(df[c]) for c in num})
    if ent:
        sty = sty.format({c: (lambda v: _es(v, 0)) for c in ent})
    if num or ent:
        sty = sty.set_properties(subset=num + ent, **{"text-align": "right"})
    # fila de ATER resaltada (en cualquier columna de texto o en el índice)
    def _ater(fila):
        txt = " ".join(str(v) for v in [fila.name, *fila.values])
        return [f"background: {ATER_FONDO}; font-weight: 700" if "ATER" in txt else "" for _ in fila]
    sty = sty.apply(_ater, axis=1)
    if df.index.name is None and pd.api.types.is_integer_dtype(df.index) and df.index.nlevels == 1:
        sty = sty.hide(axis="index")          # índice 0, 1, 2… sin significado: no se muestra
    return sty


def activar():
    """Todas las tablas de pandas del notebook se muestran con el estilo del proyecto."""
    pd.DataFrame._repr_html_ = lambda self: estilizar(self.head(200)).to_html()
