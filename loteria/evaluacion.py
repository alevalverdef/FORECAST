"""Backtest walk-forward (origen móvil): cada sorteo de prueba se pronostica usando
SOLO los sorteos anteriores, exactamente como ocurriría en la vida real."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats

from .datos import N_NUMEROS
from .modelos import Clasificador, Ensamble, Regresor

LOG_UNIF = np.log(1.0 / N_NUMEROS)


def _es_costoso(m) -> bool:
    if isinstance(m, Ensamble):
        return any(_es_costoso(x) for x in m.miembros)
    return isinstance(m, (Clasificador, Regresor))


def _actualizar_sin_reentrenar(m, hist):
    """Entre reentrenamientos, los modelos de ML solo actualizan sus rezagos."""
    if isinstance(m, Ensamble):
        for x in m.miembros:
            _actualizar_sin_reentrenar(x, hist)
    elif isinstance(m, (Clasificador, Regresor)):
        m.hist = hist
    else:
        m.fit(hist)


def walk_forward(df: pd.DataFrame, modelos: list, n_prueba: int | None = None,
                 reentrenar_cada: int = 25, minimo_entrenamiento: int = 100, verbose: bool = True):
    n = len(df)
    if n_prueba is None:
        n_prueba = min(400, max(30, int(n * 0.3)))
    inicio = max(minimo_entrenamiento, n - n_prueba)
    if inicio >= n:
        raise ValueError(f"Se necesitan más de {minimo_entrenamiento} sorteos para evaluar (hay {n}).")
    y = df["numero"].to_numpy()
    registros = []
    for m in modelos:
        t0 = time.time()
        logp, rangos, mayores, empates = [], [], [], []
        for i, t in enumerate(range(inicio, n)):
            hist = df.iloc[:t]
            if i % reentrenar_cada == 0 or not _es_costoso(m):
                m.fit(hist)
            else:
                _actualizar_sin_reentrenar(m, hist)
            p = m.predict_proba(df.iloc[t:t + 1])[0]
            logp.append(np.log(p[y[t]]))
            # rango del número real (1 = el más probable); empates -> rango promedio
            g, e = np.sum(p > p[y[t]]), np.sum(p == p[y[t]])
            rangos.append(1 + g + 0.5 * (e - 1))
            mayores.append(g)
            empates.append(e)
        logp, rangos = np.array(logp), np.array(rangos)
        registros.append((m, logp, rangos, np.array(mayores), np.array(empates)))
        if verbose:
            print(f"  {m.nombre:<62s} {time.time() - t0:6.1f}s")
    return inicio, registros


def resumen(registros) -> pd.DataFrame:
    filas = []
    for m, logp, rangos, g, e in registros:
        # P(el real cae en el top k) desempatando al azar
        top = lambda k: np.clip((k - g) / e, 0, 1)
        n = len(logp)
        dif = logp - LOG_UNIF
        if np.allclose(dif, 0):
            p_t = 1.0
        else:
            p_t = stats.ttest_1samp(dif, 0.0, alternative="greater").pvalue
        top10 = int(round(top(10).sum()))
        filas.append({
            "modelo": m.nombre,
            "familia": m.familia,
            "log_score": logp.mean(),
            "ganancia_vs_azar_bits": dif.mean() / np.log(2),
            "acierto_top1_%": 100 * top(1).mean(),
            "acierto_top5_%": 100 * top(5).mean(),
            "acierto_top10_%": 100 * top(10).mean(),
            "rango_medio_real": rangos.mean(),
            "p_valor_vs_azar": p_t,
            "p_binomial_top10": stats.binomtest(top10, n, 0.10, alternative="greater").pvalue,
        })
    r = pd.DataFrame(filas)
    # Holm: al probar ~30 modelos, alguno "gana" por pura suerte
    p = r["p_valor_vs_azar"].to_numpy()
    orden = np.argsort(p)
    ajust, acum = np.empty(len(p)), 0.0
    for i, idx in enumerate(orden):
        acum = max(acum, min(1.0, (len(p) - i) * p[idx]))
        ajust[idx] = acum
    r["p_holm"] = ajust
    r["supera_azar"] = np.where(r["p_holm"] < 0.05, "SÍ", "no")
    return r.sort_values("log_score", ascending=False).reset_index(drop=True)
