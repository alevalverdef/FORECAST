"""Minería de patrones: cientos de reglas deterministas 'número = f(historia, fecha)'.

Responde a la pregunta "¿qué tendría que haber hecho para acertar el resultado X?":
se generan todas las reglas, se ve cuáles habrían acertado cada sorteo, y luego se
comprueba si las reglas que mejor 'explicaban' el pasado siguen acertando en el futuro.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _inv(x):
    return (x % 10) * 10 + x // 10


def reglas(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Cada regla devuelve, para cada sorteo t, un número 0-99 usando SOLO sorteos < t y la fecha t."""
    n = len(df)
    y = df["numero"].to_numpy()
    f = df["fecha"]
    dia, mes, anio = f.dt.day.to_numpy(), f.dt.month.to_numpy(), f.dt.year.to_numpy()
    doy, sem = f.dt.dayofyear.to_numpy(), f.dt.isocalendar().week.to_numpy().astype(int)

    def lag(k):
        out = np.full(n, -1)
        out[k:] = y[:-k]
        return out

    R: dict[str, np.ndarray] = {}
    p1, p2, p3 = lag(1), lag(2), lag(3)
    for k in range(1, 21):
        R[f"repite el de hace {k} sorteos"] = lag(k)
    for c in range(1, 100):
        R[f"anterior + {c}"] = np.where(p1 >= 0, (p1 + c) % 100, -1)
    for k in range(1, 6):
        R[f"invertido del de hace {k}"] = np.where(lag(k) >= 0, _inv(lag(k)), -1)
    ok12 = (p1 >= 0) & (p2 >= 0)
    R["suma de los 2 anteriores"] = np.where(ok12, (p1 + p2) % 100, -1)
    R["resta de los 2 anteriores"] = np.where(ok12, abs(p1 - p2), -1)
    R["producto de los 2 anteriores"] = np.where(ok12, (p1 * p2) % 100, -1)
    R["suma de los 3 anteriores"] = np.where(ok12 & (p3 >= 0), (p1 + p2 + p3) % 100, -1)
    R["99 - anterior (complemento)"] = np.where(p1 >= 0, 99 - p1, -1)
    R["decena del anterior + unidad del previo"] = np.where(ok12, (p1 // 10) * 10 + p2 % 10, -1)
    R["unidad del anterior + decena del previo"] = np.where(ok12, (p1 % 10) * 10 + p2 // 10, -1)
    for c in range(0, 100):
        R[f"día del mes + {c}"] = (dia + c) % 100
        R[f"día del año + {c}"] = (doy + c) % 100
    R["día y mes (dd·m)"] = (dia * 10 + mes) % 100
    R["mes y día (m·dd)"] = (mes * 10 + dia) % 100
    R["día × mes"] = (dia * mes) % 100
    R["día + mes"] = (dia + mes) % 100
    R["día + mes + año(2 díg)"] = (dia + mes + anio % 100) % 100
    R["suma de dígitos de la fecha"] = np.array([sum(map(int, f"{d:02d}{m:02d}{a}")) for d, m, a in
                                                 zip(dia, mes, anio)]) % 100
    R["semana del año"] = sem % 100
    R["año (2 dígitos)"] = anio % 100
    R["día invertido"] = _inv(dia % 100)
    # Mismo día del año pasado / mismo sorteo hace 52 semanas
    s = pd.Series(y, index=f)
    for nombre, delta in (("mismo día hace 1 año (±3 días)", 365), ("hace 52 semanas exactas", 364)):
        out = np.full(n, -1)
        for i, fi in enumerate(f):
            ventana = s[(s.index >= fi - pd.Timedelta(days=delta + 3)) &
                        (s.index <= fi - pd.Timedelta(days=delta - 3))]
            if len(ventana):
                out[i] = ventana.iloc[-1]
        R[nombre] = out
    # Frecuencia en ventanas
    for w in (10, 20, 50, 100, 10_000):
        mas, menos = np.full(n, -1), np.full(n, -1)
        for t in range(10, n):
            c = np.bincount(y[max(0, t - w):t], minlength=100)
            mas[t], menos[t] = int(np.argmax(c)), int(np.argmin(c))
        etiqueta = "toda la historia" if w == 10_000 else f"últimos {w}"
        R[f"más frecuente ({etiqueta})"] = mas
        R[f"menos frecuente ({etiqueta})"] = menos
    atras = np.full(n, -1)
    ult = np.full(100, -1)
    for t in range(n):
        if t > 0:
            atras[t] = int(np.argmin(ult))
        ult[y[t]] = t
    R["el más atrasado"] = atras
    return R


def tabla_aciertos(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz booleana sorteos × reglas: True si la regla habría acertado ese sorteo."""
    y = df["numero"].to_numpy()
    return pd.DataFrame({k: v == y for k, v in reglas(df).items()}, index=df.index)


def validar(df: pd.DataFrame, corte: float = 0.6, top: int = 15):
    """Selecciona las reglas con más aciertos en el primer 60% y las prueba en el 40% restante."""
    A = tabla_aciertos(df)
    n_ent = int(len(df) * corte)
    ent, pru = A.iloc[20:n_ent], A.iloc[n_ent:]
    r = pd.DataFrame({
        "aciertos_entrenamiento_%": 100 * ent.mean(),
        "aciertos_prueba_%": 100 * pru.mean(),
    }).sort_values("aciertos_entrenamiento_%", ascending=False)
    return A, r.head(top), len(ent), len(pru), r
