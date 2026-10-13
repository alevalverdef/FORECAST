"""Carga y normalización del histórico de resultados de lotería (números 00-99)."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

N_NUMEROS = 100  # 00..99


def _norm(col: str) -> str:
    col = unicodedata.normalize("NFKD", str(col)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", col.lower())


_ALIAS_FECHA = {"fecha", "date", "dia", "fechasorteo"}
_ALIAS_NUMERO = {"numero", "num", "number", "ganador", "numeroganador", "premiomayor", "resultado"}
_ALIAS_SERIE = {"serie", "series"}
_ALIAS_SORTEO = {"sorteo", "tipo", "tiposorteo", "loteria", "juego", "horario", "turno"}


def _buscar(cols: dict[str, str], alias: set[str]) -> str | None:
    for norm, original in cols.items():
        if norm in alias:
            return original
    return None


def cargar_historico(ruta: str | Path) -> pd.DataFrame:
    """Lee CSV/Excel y devuelve un DataFrame con columnas: fecha, numero, [serie], [sorteo].

    Acepta nombres de columna en español o inglés, con o sin tildes.
    """
    ruta = Path(ruta)
    if ruta.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(ruta)
    else:
        df = pd.read_csv(ruta, sep=None, engine="python", dtype=str)

    cols = {_norm(c): c for c in df.columns}
    c_fecha = _buscar(cols, _ALIAS_FECHA)
    c_num = _buscar(cols, _ALIAS_NUMERO)
    if c_fecha is None or c_num is None:
        raise ValueError(
            f"No encontré columnas de fecha y número. Columnas recibidas: {list(df.columns)}"
        )
    c_serie = _buscar(cols, _ALIAS_SERIE)
    c_sorteo = _buscar(cols, _ALIAS_SORTEO)

    out = pd.DataFrame()
    out["fecha"] = pd.to_datetime(df[c_fecha], dayfirst=True, errors="coerce")
    out["numero"] = pd.to_numeric(df[c_num], errors="coerce")
    if c_serie is not None:
        out["serie"] = pd.to_numeric(df[c_serie], errors="coerce")
    out["sorteo"] = df[c_sorteo].astype(str).str.strip() if c_sorteo is not None else "unico"

    malos = out["fecha"].isna() | out["numero"].isna()
    if malos.any():
        print(f"[aviso] {int(malos.sum())} filas descartadas por fecha/número inválido")
    out = out[~malos].copy()
    out["numero"] = out["numero"].astype(int)
    fuera = ~out["numero"].between(0, N_NUMEROS - 1)
    if fuera.any():
        raise ValueError(f"Hay números fuera de 00-99: {out.loc[fuera, 'numero'].unique()[:10]}")

    orden_sorteo = {s: i for i, s in enumerate(sorted(out["sorteo"].unique()))}
    out["_o"] = out["sorteo"].map(orden_sorteo)
    out = out.sort_values(["fecha", "_o"]).drop(columns="_o").reset_index(drop=True)
    return out


def generar_sintetico(n: int = 1500, semilla: int = 0) -> pd.DataFrame:
    """Histórico SINTÉTICO (uniforme e independiente) solo para probar el pipeline."""
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range("2010-01-03", periods=n, freq="W-SUN")
    return pd.DataFrame(
        {
            "fecha": fechas,
            "numero": rng.integers(0, N_NUMEROS, n),
            "serie": rng.integers(0, 1000, n),
            "sorteo": "unico",
        }
    )
