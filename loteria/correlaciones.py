"""Barrido masivo de correlaciones: cientos de condiciones (rachas, posición del sorteo en el mes,
dígitos del anterior, mismo día de la semana pasada, etc.) contra varios objetivos (alto/bajo,
par/impar, decena, unidad, número exacto). Cada correlación se descubre con 2021-2024 y se
valida en 2025-2026, que no se usaron para encontrarla."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _racha(v: np.ndarray) -> np.ndarray:
    """Cuántos sorteos seguidos (hasta t-1) tienen el mismo valor que t-1."""
    r = np.zeros(len(v), int)
    for t in range(1, len(v)):
        r[t] = 1 + (r[t - 1] if t >= 2 and v[t - 1] == v[t - 2] else 0)
    return r


def condiciones(df: pd.DataFrame) -> pd.DataFrame:
    """Variables conocidas ANTES de cada sorteo (solo usan el pasado o el calendario)."""
    x = df["numero"].to_numpy()
    f = df["fecha"]
    alto, impar = (x >= 50).astype(int), (x % 2).astype(int)
    s = lambda v, k=1: pd.Series(v).shift(k).to_numpy()
    C = pd.DataFrame(index=df.index)
    C["sorteo"] = df["sorteo"].to_numpy()
    C["dia_semana"] = f.dt.dayofweek.to_numpy()
    C["mes"] = f.dt.month.to_numpy()
    C["dia_mes"] = f.dt.day.to_numpy()
    C["semana_del_mes"] = ((f.dt.day - 1) // 7 + 1).to_numpy()
    C["n_sorteo_del_mes"] = df.groupby([f.dt.year, f.dt.month]).cumcount().to_numpy() + 1
    C["mes×n_sorteo_del_mes"] = C["mes"].astype(str) + "-" + C["n_sorteo_del_mes"].astype(str)
    C["n_sorteo_del_mes_tipo"] = df.groupby([f.dt.year, f.dt.month, df["sorteo"]]).cumcount().to_numpy() + 1
    C["fin_de_mes"] = (f.dt.days_in_month - f.dt.day < 3).astype(int).to_numpy()
    C["anio"] = f.dt.year.to_numpy()
    # rachas del pasado inmediato
    ra, ri = _racha(alto), _racha(impar)
    C["racha_alto_bajo"] = np.where(s(alto) == 1, "altos", "bajos") + "×" + np.minimum(ra, 4).astype(str)
    C["racha_par_impar"] = np.where(s(impar) == 1, "impares", "pares") + "×" + np.minimum(ri, 4).astype(str)
    C["bajos_en_ult5"] = pd.Series(1 - alto).shift(1).rolling(5).sum().to_numpy()
    C["altos_en_ult3"] = pd.Series(alto).shift(1).rolling(3).sum().to_numpy()
    for k in (1, 2, 3):
        p = s(x, k)
        C[f"decena_hace{k}"] = p // 10
        C[f"unidad_hace{k}"] = p % 10
    C["anterior"] = s(x)
    C["suma_digitos_ant"] = (s(x) // 10 + s(x) % 10)
    C["dif_digitos_ant"] = np.abs(s(x) // 10 - s(x) % 10)
    C["ant_sube_o_baja"] = np.sign(s(x) - s(x, 2))
    C["ant_es_doble"] = (s(x) // 10 == s(x) % 10).astype(float)
    C["salto_ant"] = pd.cut(np.abs(s(x) - s(x, 2)), [-1, 9, 29, 49, 99], labels=False)
    C["decena_ant×decena_hace2"] = (s(x) // 10).astype(str) + "-" + (s(x, 2) // 10).astype(str)
    return C


def objetivos(df: pd.DataFrame) -> dict:
    x = df["numero"].to_numpy()
    return {"alto(50-99)": (x >= 50).astype(int), "impar": x % 2, "decena": x // 10,
            "unidad": x % 10, "numero": x}


def barrer(df: pd.DataFrame, corte: str = "2025-01-01", minimo: int = 15):
    C, Y = condiciones(df), objetivos(df)
    ent = (df["fecha"] < corte).to_numpy()
    filas = []
    for var in C.columns:
        v = C[var].astype(str).to_numpy()
        valido = ~pd.isna(C[var]).to_numpy()
        for obj, y in Y.items():
            clases = np.unique(y)
            base = 1.0 / len(clases) if obj != "alto(50-99)" else 0.5
            for nivel in np.unique(v[valido & ent]):
                m = valido & (v == nivel)
                me, mp = m & ent, m & ~ent
                if me.sum() < minimo:
                    continue
                for c in clases:
                    ke = int((y[me] == c).sum())
                    if ke == 0 or ke / me.sum() <= base:
                        continue  # solo correlaciones positivas (la clase sale MÁS de lo normal)
                    pe = stats.binomtest(ke, int(me.sum()), base, alternative="greater").pvalue
                    kp, n_p = int((y[mp] == c).sum()), int(mp.sum())
                    filas.append({"condicion": f"{var} = {nivel}", "objetivo": f"{obj} = {c}",
                                  "n_ent": int(me.sum()), "%_ent": 100 * ke / me.sum(),
                                  "p_ent": pe, "n_prueba": n_p,
                                  "%_prueba": 100 * kp / n_p if n_p else np.nan,
                                  "%_azar": 100 * base, "kp": kp})
    r = pd.DataFrame(filas)
    r["p_prueba"] = [stats.binomtest(int(k), int(n), b / 100, alternative="greater").pvalue if n else np.nan
                     for k, n, b in zip(r["kp"], r["n_prueba"], r["%_azar"])]
    # q-valor (Benjamini-Hochberg) en entrenamiento
    p = r["p_ent"].to_numpy()
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    r["q_ent"] = np.minimum(q, 1)
    return r.sort_values("p_ent").drop(columns="kp").reset_index(drop=True)
