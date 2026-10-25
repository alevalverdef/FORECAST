"""Modelo de DÍAS ANÁLOGOS: para el sorteo a pronosticar busca en todo el histórico los sorteos
más parecidos según muchos factores (tipo de sorteo, mes, semana del mes, último sorteo del mes,
n.º de sorteo, día, qué venía saliendo antes, qué salió antes en el mismo mes...). Cada sorteo
pasado "vota" por su número con un peso que crece con su parecido. Cuánto pesa cada factor NO
se fija a mano: se aprende maximizando la verosimilitud en 2021-2024 y se valida en 2025-2026."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

FACTORES = ["mismo tipo de sorteo", "mismo mes", "misma semana del mes", "último sorteo de su tipo en el mes",
            "mismo n.º de sorteo de su tipo en el mes", "mismo día de la semana", "misma quincena",
            "anterior en la misma decena", "anterior: mismo alto/bajo", "anterior del mismo tipo: misma decena",
            "decenas ya salidas este mes (coincidencias)", "mismo año"]


def rasgos(df: pd.DataFrame) -> dict:
    f, x, s = df["fecha"], df["numero"].to_numpy(), df["sorteo"].to_numpy()
    n = len(df)
    ym = (f.dt.year * 100 + f.dt.month).to_numpy()
    tipo_n = df.groupby([f.dt.year, f.dt.month, df["sorteo"]]).cumcount().to_numpy()
    ultimo = np.zeros(n, int)
    for i in range(n):  # ¿no hay otro sorteo de su tipo después en el mismo mes? (calendario: +7 días)
        ultimo[i] = int((f.iloc[i] + pd.Timedelta(days=7)).month != f.iloc[i].month)
    prev = np.r_[-1, x[:-1]]
    prev_tipo = np.full(n, -1)
    ult = {}
    for i in range(n):
        prev_tipo[i] = ult.get(s[i], -1)
        ult[s[i]] = x[i]
    dec_mes = []
    for i in range(n):
        j = np.where((ym[:i] == ym[i]))[0]
        dec_mes.append(set(x[j] // 10))
    return dict(s=s, mes=f.dt.month.to_numpy(), sem=((f.dt.day - 1) // 7).to_numpy(), ultimo=ultimo, tipo_n=tipo_n,
                dow=f.dt.dayofweek.to_numpy(), q=(f.dt.day >= 15).to_numpy(), pdec=np.where(prev >= 0, prev // 10, -1),
                palto=np.where(prev >= 0, prev >= 50, -1), ptdec=np.where(prev_tipo >= 0, prev_tipo // 10, -1),
                decmes=dec_mes, anio=f.dt.year.to_numpy(), x=x)


def matriz_parecido(R: dict, t: int, hasta: int) -> np.ndarray:
    """(n_factores, hasta): 1 si el sorteo s<hasta coincide con t en cada factor."""
    i = np.arange(hasta)
    M = [R["s"][i] == R["s"][t], R["mes"][i] == R["mes"][t], R["sem"][i] == R["sem"][t],
         (R["ultimo"][i] == 1) & (R["ultimo"][t] == 1), R["tipo_n"][i] == R["tipo_n"][t], R["dow"][i] == R["dow"][t],
         R["q"][i] == R["q"][t], (R["pdec"][i] == R["pdec"][t]) & (R["pdec"][t] >= 0),
         R["palto"][i] == R["palto"][t], (R["ptdec"][i] == R["ptdec"][t]) & (R["ptdec"][t] >= 0),
         np.array([len(R["decmes"][k] & R["decmes"][t]) for k in i]) / 3.0, R["anio"][i] == R["anio"][t]]
    return np.array(M, float)


def distribucion(R, t, w, beta0, mezcla):
    M = matriz_parecido(R, t, t)
    peso = np.exp(beta0 + w @ M)
    c = np.bincount(R["x"][:t], weights=peso, minlength=100)
    return (1 - mezcla) * c / c.sum() + mezcla / 100


def ajustar(R, desde, hasta):
    Ms = [matriz_parecido(R, t, t) for t in range(desde, hasta)]
    ys = R["x"][desde:hasta]

    def nll(th):
        w, mez = th[:-1], 1 / (1 + np.exp(-th[-1]))
        tot = 0.0
        for M, t, y in zip(Ms, range(desde, hasta), ys):
            pe = np.exp(w @ M)
            p = (1 - mez) * pe[R["x"][:t] == y].sum() / pe.sum() + mez / 100
            tot -= np.log(p)
        return tot + 0.5 * np.sum(w ** 2)  # regularización: factores sin evidencia se quedan en 0

    r = minimize(nll, np.r_[np.zeros(len(FACTORES)), 0.0], method="L-BFGS-B")
    return r.x[:-1], 1 / (1 + np.exp(-r.x[-1]))
