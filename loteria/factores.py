"""Modelo de factores por número (logit condicional): para cada sorteo y CADA número candidato
se calculan decenas de factores (qué salió en el mismo sorteo de años anteriores, en el mismo
mes de años anteriores, en el mismo contexto, repeticiones, atrasos, vecinos, espejos, etc.).
P(número) ∝ exp(Σ β·factor). Los β pueden ser positivos o negativos. Selección de factores
paso a paso: solo entra un factor si mejora la predicción en un año de validación que no se usó
para estimarlo; los que no aportan se eliminan. La prueba final es en sorteos no vistos."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

K = 100


def construir(df: pd.DataFrame):
    f, x, s = df["fecha"].reset_index(drop=True), df["numero"].to_numpy(), df["sorteo"].to_numpy()
    n = len(df)
    nums = np.arange(K)
    doy, mes, dia = f.dt.dayofyear.to_numpy(), f.dt.month.to_numpy(), f.dt.day.to_numpy()
    anio, dow = f.dt.year.to_numpy(), f.dt.dayofweek.to_numpy()
    sem = (dia - 1) // 7
    ordn = df.groupby([f.dt.year, f.dt.month, df["sorteo"]]).cumcount().to_numpy()
    nombres, F = [], []

    def add(nombre, M):
        nombres.append(nombre)
        F.append(M.astype(np.float32))

    # --- mismo sorteo en años anteriores (el más cercano a la misma fecha, mismo tipo)
    for L in (1, 2, 3):
        M = np.zeros((n, K)); Md = np.zeros((n, K)); Mu = np.zeros((n, K))
        for t in range(n):
            c = np.where((anio[:t] == anio[t] - L) & (s[:t] == s[t]))[0]
            if len(c):
                j = c[np.argmin(np.abs(doy[c] - doy[t]))]
                M[t, x[j]] = 1; Md[t] = nums // 10 == x[j] // 10; Mu[t] = nums % 10 == x[j] % 10
        add(f"= número del mismo sorteo hace {L} año(s)", M)
        add(f"misma decena que el mismo sorteo hace {L} año(s)", Md)
        add(f"misma unidad que el mismo sorteo hace {L} año(s)", Mu)
    # --- mismo mes / mismo n.º de sorteo / misma semana en años anteriores
    M1 = np.zeros((n, K)); M2 = np.zeros((n, K)); M3 = np.zeros((n, K)); M4 = np.zeros((n, K)); M5 = np.zeros((n, K))
    for t in range(n):
        prev_y = anio[:t] < anio[t]
        a = np.where(prev_y & (mes[:t] == mes[t]))[0]; M1[t] = np.bincount(x[a], minlength=K)
        b = np.where(prev_y & (mes[:t] == mes[t]) & (s[:t] == s[t]) & (ordn[:t] == ordn[t]))[0]; M2[t] = np.bincount(x[b], minlength=K)
        c = np.where(prev_y & (mes[:t] == mes[t]) & (sem[:t] == sem[t]))[0]; M3[t] = np.bincount(x[c], minlength=K)
        d = np.where(prev_y & (np.abs(doy[:t] - doy[t]) <= 7))[0]; M4[t] = np.bincount(x[d], minlength=K)
        e = np.where(prev_y & (mes[:t] == mes[t]))[0]; M5[t] = np.bincount(x[e] // 10, minlength=10)[nums // 10]
    add("veces en el mismo mes de años anteriores", M1)
    add("veces en el mismo n.º de sorteo del mismo mes, años anteriores", M2)
    add("veces en la misma semana del mismo mes, años anteriores", M3)
    add("veces a ±7 días de la misma fecha, años anteriores", M4)
    add("veces de su decena en el mismo mes de años anteriores", M5 / 10)
    # --- contexto en toda la historia
    for nom, v in (("mismo tipo de sorteo", s), ("mismo día de la semana", dow), ("mismo día del mes", dia),
                   ("misma semana del mes", sem), ("mismo n.º de sorteo de su tipo", ordn)):
        M = np.zeros((n, K))
        for t in range(n):
            M[t] = np.bincount(x[:t][v[:t] == v[t]], minlength=K)
        add(f"veces en el {nom} (historia)", M / max(1, 1))
    # --- dinámica reciente
    cnt = np.zeros((n, K)); gap = np.zeros((n, K)); last = np.full(K, -1)
    c30 = np.zeros((n, K)); run = np.zeros(K)
    for t in range(n):
        cnt[t] = run; gap[t] = np.where(last < 0, t + 1, t - last)
        c30[t] = np.bincount(x[max(0, t - 30):t], minlength=K)
        run[x[t]] += 1; last[x[t]] = t
    add("veces en toda la historia", cnt / np.maximum(1, np.arange(n))[:, None] * K)
    add("log(sorteos sin salir)", np.log(gap))
    add("veces en los últimos 30", c30)
    for L in range(1, 8):
        M = np.zeros((n, K))
        for t in range(L, n):
            M[t, x[t - L]] = 1
        add(f"= número de hace {L} sorteo(s)", M)
    P = np.r_[-1, x[:-1]]
    vec = lambda cond: np.array([cond(nums, p) if p >= 0 else np.zeros(K, bool) for p in P])
    add("misma decena que el anterior", vec(lambda a, p: a // 10 == p // 10))
    add("misma unidad que el anterior", vec(lambda a, p: a % 10 == p % 10))
    add("espejo del anterior (dígitos invertidos)", vec(lambda a, p: a == (p % 10) * 10 + p // 10))
    add("vecino ±1 del anterior", vec(lambda a, p: np.abs(a - p) == 1))
    add("mismo alto/bajo que el anterior", vec(lambda a, p: (a >= 50) == (p >= 50)))
    add("misma paridad que el anterior", vec(lambda a, p: a % 2 == p % 2))
    add("decena ya salió este mes", np.array([np.isin(nums // 10, x[:t][(anio[:t] == anio[t]) & (mes[:t] == mes[t])] // 10) for t in range(n)]))
    add("número ya salió este mes", np.array([np.isin(nums, x[:t][(anio[:t] == anio[t]) & (mes[:t] == mes[t])]) for t in range(n)]))
    add("es doble (11, 22, ...)", np.tile(nums // 10 == nums % 10, (n, 1)))
    add("es alto (50-99)", np.tile(nums >= 50, (n, 1)))
    add("es impar", np.tile(nums % 2 == 1, (n, 1)))
    X = np.stack(F, axis=2)  # (n, K, F)
    # estandarizar cada factor
    mu, sd = X.reshape(-1, X.shape[2]).mean(0), X.reshape(-1, X.shape[2]).std(0) + 1e-9
    return (X - mu) / sd, x, nombres


def nll(beta, X, y, l2=1.0):
    z = X @ beta
    z -= z.max(1, keepdims=True)
    lse = np.log(np.exp(z).sum(1))
    return -(z[np.arange(len(y)), y] - lse).sum() + l2 * (beta ** 2).sum()


def ajustar(X, y, cols, l2=1.0):
    Xs = X[:, :, cols]
    def f(b):
        z = Xs @ b; z -= z.max(1, keepdims=True); e = np.exp(z); p = e / e.sum(1, keepdims=True)
        g = -(Xs[np.arange(len(y)), y] - (p[:, :, None] * Xs).sum(1)).sum(0) + 2 * l2 * b
        return nll(b, Xs, y, l2), g
    return minimize(f, np.zeros(len(cols)), jac=True, method="L-BFGS-B").x


def logscore(beta, X, y, cols):
    z = X[:, :, cols] @ beta; z -= z.max(1, keepdims=True)
    return (z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean() - np.log(1 / K)


def seleccion(X, y, ent, val, maxf=15):
    """Selección hacia adelante: agrega el factor que más mejora la validación; para cuando ninguno mejora."""
    elegidos, mejor = [], 0.0
    while len(elegidos) < maxf:
        cand = []
        for j in range(X.shape[2]):
            if j in elegidos:
                continue
            c = elegidos + [j]
            b = ajustar(X[ent], y[ent], c)
            cand.append((logscore(b, X[val], y[val], c), j))
        sc, j = max(cand)
        if sc <= mejor + 1e-4:
            break
        elegidos.append(j); mejor = sc
    return elegidos, mejor
