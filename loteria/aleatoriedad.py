"""Pruebas estadísticas de aleatoriedad: ¿hay alguna señal explotable en el histórico?

Si ninguna prueba rechaza la hipótesis nula de "sorteo uniforme e independiente",
ningún modelo puede, de forma legítima, superar a la probabilidad 1/100.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .datos import N_NUMEROS


def _chi2_independencia(df: pd.DataFrame, col: str, objetivo: str) -> tuple[float, int, float]:
    tabla = pd.crosstab(df[col], df[objetivo])
    if tabla.shape[0] < 2 or tabla.shape[1] < 2:
        return np.nan, 0, np.nan
    chi2, p, gl, _ = stats.chi2_contingency(tabla)
    return chi2, gl, p


def _cramers_v(chi2: float, n: int, forma: tuple[int, int]) -> float:
    k = min(forma) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else np.nan


def analizar(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["decena"] = d["numero"] // 10
    d["unidad"] = d["numero"] % 10
    d["dia_semana"] = d["fecha"].dt.dayofweek
    d["mes"] = d["fecha"].dt.month
    d["dia_mes"] = d["fecha"].dt.day
    d["quincena"] = (d["dia_mes"] >= 15).astype(int)
    d["anio"] = d["fecha"].dt.year
    d["previo_decena"] = d["decena"].shift(1)
    d["previo_unidad"] = d["unidad"].shift(1)
    n = len(d)
    filas = []

    # 1. Uniformidad global del número, decena y unidad
    for col, k in (("numero", N_NUMEROS), ("decena", 10), ("unidad", 10)):
        obs = np.bincount(d[col], minlength=k)
        chi2, p = stats.chisquare(obs)
        filas.append(("Uniformidad", col, chi2, k - 1, p, np.nan))

    # 2. Independencia respecto al calendario / sorteo (decena y unidad: tablas manejables)
    contextos = ["dia_semana", "mes", "quincena", "anio", "sorteo"]
    for ctx in contextos:
        if d[ctx].nunique() < 2:
            continue
        for obj in ("decena", "unidad"):
            chi2, gl, p = _chi2_independencia(d, ctx, obj)
            v = _cramers_v(chi2, n, (d[ctx].nunique(), 10))
            filas.append((f"Independencia {ctx}", obj, chi2, gl, p, v))

    # 3. Dependencia serial (sorteo anterior -> actual)
    dd = d.dropna(subset=["previo_decena"])
    for prev, obj in (("previo_decena", "decena"), ("previo_unidad", "unidad")):
        chi2, gl, p = _chi2_independencia(dd, prev, obj)
        filas.append(("Serial (Markov)", f"{prev}->{obj}", chi2, gl, p,
                      _cramers_v(chi2, len(dd), (10, 10))))

    # 4. Prueba de rachas (Wald-Wolfowitz) sobre la mediana
    x = d["numero"].to_numpy()
    arriba = x > np.median(x)
    rachas = 1 + int(np.sum(arriba[1:] != arriba[:-1]))
    n1, n2 = int(arriba.sum()), int((~arriba).sum())
    if n1 and n2:
        mu = 2 * n1 * n2 / (n1 + n2) + 1
        var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
        z = (rachas - mu) / np.sqrt(var)
        filas.append(("Rachas", "numero>mediana", z, np.nan, 2 * stats.norm.sf(abs(z)), np.nan))

    # 5. Ljung-Box (autocorrelación hasta lag 10)
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        lb = acorr_ljungbox(x, lags=[10], return_df=True)
        filas.append(("Ljung-Box lag10", "numero", float(lb["lb_stat"].iloc[0]), 10,
                      float(lb["lb_pvalue"].iloc[0]), np.nan))
    except Exception:  # statsmodels no disponible
        pass

    res = pd.DataFrame(filas, columns=["prueba", "variable", "estadistico", "gl", "p_valor", "cramers_v"])
    # Corrección por comparaciones múltiples (Holm): con muchas pruebas alguna sale "significativa" por azar
    p = res["p_valor"].to_numpy()
    orden = np.argsort(p)
    m = len(p)
    ajust = np.empty(m)
    acum = 0.0
    for i, idx in enumerate(orden):
        acum = max(acum, min(1.0, (m - i) * p[idx]))
        ajust[idx] = acum
    res["p_holm"] = ajust
    res["senal"] = np.where(res["p_holm"] < 0.05, "SÍ", "no")
    return res
