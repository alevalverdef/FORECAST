"""Catálogo de modelos probabilísticos. Cada modelo devuelve una distribución sobre 00-99.

Interfaz común:
    m.fit(hist)                 # hist: DataFrame con fecha, numero, sorteo (solo pasado)
    m.predict_proba(ctx) -> (n, 100)   # ctx: filas a pronosticar (fecha, sorteo)

Se evalúan distribuciones completas (no un único número) porque con un proceso
uniforme lo único medible honestamente es cuánta probabilidad se le asigna al real.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .datos import N_NUMEROS

warnings.filterwarnings("ignore")
K = N_NUMEROS


def _normalizar(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, None)
    return p / p.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------- features
def calendario(df: pd.DataFrame) -> pd.DataFrame:
    f = df["fecha"]
    return pd.DataFrame(
        {
            "dia_semana": f.dt.dayofweek,
            "mes": f.dt.month,
            "dia_mes": f.dt.day,
            "semana_anio": f.dt.isocalendar().week.astype(int).to_numpy(),
            "dia_anio": f.dt.dayofyear,
            "quincena": (f.dt.day >= 15).astype(int),
            "fin_de_mes": (f.dt.days_in_month - f.dt.day < 3).astype(int),
            "anio": f.dt.year,
        },
        index=df.index,
    )


def features(hist: pd.DataFrame, ctx: pd.DataFrame | None = None, sorteos: list[str] | None = None):
    """Matriz de features para hist (+ctx al final). Los rezagos solo usan sorteos anteriores."""
    base = hist if ctx is None else pd.concat([hist, ctx], ignore_index=True)
    X = calendario(base)
    sorteos = sorteos or sorted(hist["sorteo"].unique())
    X["sorteo"] = base["sorteo"].map({s: i for i, s in enumerate(sorteos)}).fillna(-1).astype(int)
    num = base["numero"].astype(float)
    for lag in (1, 2, 3):
        prev = num.shift(lag)
        X[f"prev{lag}"] = prev
        X[f"prev{lag}_dec"] = prev // 10
        X[f"prev{lag}_uni"] = prev % 10
    # Si hay varios pasos futuros en ctx, los rezagos desconocidos se rellenan con el último real
    X = X.ffill().fillna(-1)
    # Senos/cosenos para estacionalidad
    X["sin_doy"] = np.sin(2 * np.pi * X["dia_anio"] / 365.25)
    X["cos_doy"] = np.cos(2 * np.pi * X["dia_anio"] / 365.25)
    X["sin_dow"] = np.sin(2 * np.pi * X["dia_semana"] / 7)
    X["cos_dow"] = np.cos(2 * np.pi * X["dia_semana"] / 7)
    n = len(hist)
    return X.iloc[:n].to_numpy(float), X.iloc[n:].to_numpy(float), list(X.columns)


# --------------------------------------------------------------------------- modelos de conteo
class Uniforme:
    nombre = "Uniforme (azar puro, 1/100)"
    familia = "referencia"

    def fit(self, hist):
        return self

    def predict_proba(self, ctx):
        return np.full((len(ctx), K), 1.0 / K)


class FrecuenciaGlobal:
    familia = "frecuencia"

    def __init__(self, alfa: float = 1.0):
        self.alfa = alfa
        self.nombre = f"Frecuencia histórica (Laplace a={alfa})"

    def fit(self, hist):
        c = np.bincount(hist["numero"], minlength=K) + self.alfa
        self.p = c / c.sum()
        return self

    def predict_proba(self, ctx):
        return np.tile(self.p, (len(ctx), 1))


class FrecuenciaReciente:
    """'Números calientes': conteo con decaimiento exponencial."""

    familia = "frecuencia"

    def __init__(self, vida_media: int = 100):
        self.h = vida_media
        self.nombre = f"Números calientes (vida media {vida_media} sorteos)"

    def fit(self, hist):
        n = len(hist)
        w = 0.5 ** ((n - 1 - np.arange(n)) / self.h)
        c = np.bincount(hist["numero"], weights=w, minlength=K) + 0.5
        self.p = c / c.sum()
        return self

    def predict_proba(self, ctx):
        return np.tile(self.p, (len(ctx), 1))


class Atrasados:
    """'Números fríos/atrasados': más probabilidad a los que llevan más sorteos sin salir."""

    familia = "frecuencia"

    def __init__(self, beta: float = 1.0):
        self.beta = beta
        self.nombre = f"Números atrasados (gap^{beta})"

    def fit(self, hist):
        n = len(hist)
        ult = np.full(K, -1)
        for i, x in enumerate(hist["numero"].to_numpy()):
            ult[x] = i
        gap = np.where(ult < 0, n + 1, n - ult).astype(float)
        self.p = _normalizar(gap**self.beta)
        return self

    def predict_proba(self, ctx):
        return np.tile(self.p, (len(ctx), 1))


class Condicional:
    """Frecuencia en 'escenarios similares' (mismo día de semana, mes, ...),
    con encogimiento bayesiano hacia la frecuencia global para no sobreajustar."""

    familia = "contextual"

    def __init__(self, variables: tuple[str, ...], fuerza: float = 50.0):
        self.vars = variables
        self.a = fuerza
        self.nombre = f"Escenarios similares [{', '.join(variables)}] (a={fuerza:g})"

    def _llave(self, df):
        cal = calendario(df)
        cal["sorteo"] = df["sorteo"].to_numpy()
        return cal[list(self.vars)].astype(str).agg("|".join, axis=1)

    def fit(self, hist):
        g = np.bincount(hist["numero"], minlength=K) + 1.0
        self.prior = g / g.sum()
        llave = self._llave(hist)
        self.tabla = {
            k: np.bincount(grp, minlength=K) for k, grp in hist["numero"].groupby(llave.to_numpy())
        }
        return self

    def predict_proba(self, ctx):
        out = []
        for k in self._llave(ctx):
            c = self.tabla.get(k, np.zeros(K))
            out.append((c + self.a * self.prior) / (c.sum() + self.a))
        return np.array(out)


class NaiveBayesContextual:
    """Combina varios contextos multiplicando sus 'lifts' respecto a la frecuencia global."""

    familia = "contextual"
    nombre = "Naive Bayes de contextos (día semana × mes × día mes × sorteo)"

    def __init__(self):
        self.partes = [Condicional((v,), 100.0) for v in ("dia_semana", "mes", "dia_mes", "sorteo")]

    def fit(self, hist):
        for m in self.partes:
            m.fit(hist)
        self.prior = self.partes[0].prior
        return self

    def predict_proba(self, ctx):
        logp = np.log(self.prior)[None, :].repeat(len(ctx), 0)
        for m in self.partes:
            logp += np.log(m.predict_proba(ctx)) - np.log(self.prior)
        logp -= logp.max(1, keepdims=True)
        return _normalizar(np.exp(logp))


class Markov:
    """P(número | número anterior) con suavizado hacia la frecuencia global."""

    familia = "secuencial"

    def __init__(self, fuerza: float = 20.0):
        self.a = fuerza
        self.nombre = f"Cadena de Markov orden 1 (a={fuerza:g})"

    def fit(self, hist):
        x = hist["numero"].to_numpy()
        g = np.bincount(x, minlength=K) + 1.0
        self.prior = g / g.sum()
        self.T = np.zeros((K, K))
        np.add.at(self.T, (x[:-1], x[1:]), 1)
        self.ultimo = x[-1]
        return self

    def predict_proba(self, ctx):
        c = self.T[self.ultimo]
        p = (c + self.a * self.prior) / (c.sum() + self.a)
        return np.tile(p, (len(ctx), 1))


class Digitos:
    """Modela decena y unidad por separado (cada una con contexto día de semana) y las combina."""

    familia = "contextual"
    nombre = "Dígitos independientes (decena × unidad | día semana)"

    def fit(self, hist):
        dow = hist["fecha"].dt.dayofweek.to_numpy()
        self.t = {}
        for nom, v in (("d", hist["numero"].to_numpy() // 10), ("u", hist["numero"].to_numpy() % 10)):
            g = np.bincount(v, minlength=10) + 1.0
            g = g / g.sum()
            tab = {}
            for d in range(7):
                c = np.bincount(v[dow == d], minlength=10)
                tab[d] = (c + 30 * g) / (c.sum() + 30)
            self.t[nom] = tab
        return self

    def predict_proba(self, ctx):
        out = []
        for d in ctx["fecha"].dt.dayofweek:
            out.append(np.outer(self.t["d"][d], self.t["u"][d]).ravel())
        return np.array(out)


# --------------------------------------------------------------------------- machine learning
class Clasificador:
    """Envuelve un clasificador sklearn-compatible de 100 clases."""

    familia = "machine learning"

    def __init__(self, nombre: str, fabrica, escalar: bool = False):
        self.nombre = nombre
        self.fabrica = fabrica
        self.escalar = escalar

    def fit(self, hist):
        from sklearn.preprocessing import StandardScaler

        self.sorteos = sorted(hist["sorteo"].unique())
        X, _, _ = features(hist, sorteos=self.sorteos)
        X, y = X[3:], hist["numero"].to_numpy()[3:]  # los 3 primeros no tienen rezagos
        self.sc = StandardScaler().fit(X) if self.escalar else None
        if self.sc is not None:
            X = self.sc.transform(X)
        # Algunas librerías exigen clases contiguas 0..k-1
        self.clases = np.unique(y)
        self.mapa = {c: i for i, c in enumerate(self.clases)}
        self.m = self.fabrica().fit(X, np.vectorize(self.mapa.get)(y))
        self.hist = hist
        return self

    def predict_proba(self, ctx):
        _, Xc, _ = features(self.hist, ctx, sorteos=self.sorteos)
        if self.sc is not None:
            Xc = self.sc.transform(Xc)
        pr = self.m.predict_proba(Xc)
        out = np.full((len(ctx), K), 1e-6)
        out[:, self.clases[self.m.classes_]] = pr
        return _normalizar(out)


class Regresor:
    """Trata el número como variable continua (regresión) y convierte la predicción
    en distribución con una normal centrada en el valor predicho. Se incluye porque
    se pidió, aunque el número de lotería es una etiqueta, no una magnitud."""

    familia = "regresión"

    def __init__(self, nombre: str, fabrica):
        self.nombre = nombre
        self.fabrica = fabrica

    def fit(self, hist):
        self.sorteos = sorted(hist["sorteo"].unique())
        X, _, _ = features(hist, sorteos=self.sorteos)
        X, y = X[3:], hist["numero"].to_numpy()[3:].astype(float)
        self.m = self.fabrica().fit(X, y)
        self.sd = max(float(np.std(y - self.m.predict(X))), 5.0)
        self.hist = hist
        return self

    def predict_proba(self, ctx):
        from scipy.stats import norm

        _, Xc, _ = features(self.hist, ctx, sorteos=self.sorteos)
        mu = self.m.predict(Xc)
        grid = np.arange(K)
        return _normalizar(norm.pdf(grid[None, :], mu[:, None], self.sd))


class Ensamble:
    familia = "ensamble"

    def __init__(self, nombre, miembros):
        self.nombre = nombre
        self.miembros = miembros

    def fit(self, hist):
        for m in self.miembros:
            m.fit(hist)
        return self

    def predict_proba(self, ctx):
        # Promedio geométrico (log-pooling): estándar para combinar pronósticos probabilísticos
        lp = np.mean([np.log(m.predict_proba(ctx)) for m in self.miembros], axis=0)
        return _normalizar(np.exp(lp - lp.max(1, keepdims=True)))


# --------------------------------------------------------------------------- catálogo
def catalogo(rapido: bool = False) -> list:
    from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingRegressor,
                                  HistGradientBoostingClassifier, RandomForestClassifier,
                                  RandomForestRegressor)
    from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier

    modelos = [
        Uniforme(),
        FrecuenciaGlobal(1.0),
        FrecuenciaGlobal(20.0),
        FrecuenciaReciente(30),
        FrecuenciaReciente(150),
        Atrasados(0.5),
        Atrasados(1.0),
        Markov(20.0),
        Condicional(("dia_semana",)),
        Condicional(("mes",)),
        Condicional(("dia_mes",)),
        Condicional(("quincena",)),
        Condicional(("sorteo",)),
        Condicional(("dia_semana", "mes")),
        Condicional(("mes", "dia_mes"), 30.0),
        NaiveBayesContextual(),
        Digitos(),
        Regresor("Regresión lineal simple/múltiple", LinearRegression),
        Regresor("Regresión Poisson", lambda: PoissonRegressor(alpha=1.0, max_iter=300)),
        Regresor("Random Forest (regresión)",
                 lambda: RandomForestRegressor(200, min_samples_leaf=20, n_jobs=-1, random_state=0)),
        Regresor("Gradient Boosting (regresión)",
                 lambda: GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                                   random_state=0)),
        Clasificador("Regresión logística multinomial (L2)",
                     lambda: LogisticRegression(C=0.05, max_iter=500), escalar=True),
        Clasificador("Random Forest (clasificación)",
                     lambda: RandomForestClassifier(300, min_samples_leaf=15, max_features="sqrt",
                                                    n_jobs=-1, random_state=0)),
        Clasificador("Extra Trees",
                     lambda: ExtraTreesClassifier(300, min_samples_leaf=15, n_jobs=-1, random_state=0)),
        Clasificador("K vecinos más cercanos (k=50)",
                     lambda: KNeighborsClassifier(50, weights="distance"), escalar=True),
        Clasificador("Naive Bayes gaussiano", GaussianNB, escalar=True),
        Clasificador("Red neuronal (MLP)",
                     lambda: MLPClassifier((32,), alpha=1.0, max_iter=300, early_stopping=True,
                                           random_state=0), escalar=True),
    ]
    if not rapido:
        modelos.append(
            Clasificador("Gradient Boosting histograma (sklearn)",
                         lambda: HistGradientBoostingClassifier(max_iter=40, max_depth=3,
                                                                learning_rate=0.05,
                                                                l2_regularization=5.0, random_state=0)))
        try:
            import lightgbm as lgb

            modelos.append(Clasificador("LightGBM", lambda: lgb.LGBMClassifier(
                n_estimators=60, num_leaves=7, learning_rate=0.03, min_child_samples=20,
                reg_lambda=5.0, verbose=-1, random_state=0)))
        except ImportError:
            pass
        try:
            import xgboost as xgb

            modelos.append(Clasificador("XGBoost", lambda: xgb.XGBClassifier(
                n_estimators=60, max_depth=2, learning_rate=0.05, reg_lambda=5.0,
                tree_method="hist", n_jobs=-1, random_state=0)))
        except ImportError:
            pass
    modelos.append(Ensamble("Ensamble (contexto + Markov + Random Forest)", [
        NaiveBayesContextual(), Markov(20.0),
        Clasificador("RF", lambda: RandomForestClassifier(300, min_samples_leaf=15, n_jobs=-1,
                                                          random_state=0)),
    ]))
    return modelos
