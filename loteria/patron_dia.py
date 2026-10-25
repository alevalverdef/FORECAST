"""Método oficial (definido con el usuario): PATRÓN DEL DÍA primero, frecuencia general solo como ajuste.

1. Cluster de días similares al sorteo a pronosticar: mismo tipo de sorteo, misma posición
   (n.º de ese día de la semana dentro del mes), misma cantidad de ese día en el mes y si es el último.
2. Puntaje principal = veces que salió cada número en ese cluster.
3. Filtros que se sostienen dentro del cluster (en ambas mitades de la historia) se aplican.
4. Ajuste: la frecuencia en toda la historia solo desempata / suma un peso pequeño.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rasgos(fechas: pd.Series) -> pd.DataFrame:
    f = pd.to_datetime(fechas)
    n_dia = f.apply(lambda d: sum(1 for k in range(1, d.days_in_month + 1)
                                  if pd.Timestamp(d.year, d.month, k).dayofweek == d.dayofweek))
    return pd.DataFrame({"dow": f.dt.dayofweek.to_numpy(), "pos": ((f.dt.day - 1) // 7 + 1).to_numpy(),
                         "n_dia": n_dia.to_numpy(), "ultimo": (f.dt.days_in_month - f.dt.day < 7).to_numpy()})


def cluster(df: pd.DataFrame, fecha, sorteo) -> np.ndarray:
    r = _rasgos(df["fecha"])
    o = _rasgos(pd.Series([pd.Timestamp(fecha)])).iloc[0]
    m = (df["sorteo"].to_numpy() == sorteo) & (r["dow"].to_numpy() == o["dow"]) & (r["ultimo"].to_numpy() == o["ultimo"])
    m &= (r["n_dia"].to_numpy() == o["n_dia"]) if o["ultimo"] else (r["pos"].to_numpy() == o["pos"])
    return np.where(m)[0]


def pronosticar(df: pd.DataFrame, fecha, sorteo, top: int = 5, peso_general: float = 0.1):
    x = df["numero"].to_numpy()
    idx = cluster(df, fecha, sorteo)
    c = np.bincount(x[idx], minlength=100).astype(float)
    # Filtro validado en el cluster: ¿el resultado cae en la decena del sorteo anterior?
    prev_ok = [t for t in idx if t > 0]
    mismas = sum(x[t] // 10 == x[t - 1] // 10 for t in prev_ok)
    mitad = len(prev_ok) // 2
    m1 = sum(x[t] // 10 == x[t - 1] // 10 for t in prev_ok[:mitad])
    m2 = mismas - m1
    filtro = len(prev_ok) >= 20 and m1 <= 0.5 * mitad / 10 and m2 <= 0.5 * (len(prev_ok) - mitad) / 10
    general = np.bincount(x, minlength=100) / len(x) * 100  # veces por cada 100 sorteos
    score = c + peso_general * general / general.max()
    if filtro:
        score[np.arange(100) // 10 == x[-1] // 10] = -1
    orden = np.argsort(-score, kind="stable")[:top]
    return orden, c, idx, filtro, (mismas, len(prev_ok))
