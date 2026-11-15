"""Prueba global de permutación: ¿hay ALGÚN patrón aprovechable, visto o no visto?
Se aplica una batería amplia de reglas de pronóstico a la serie real y a cientos de copias de
la misma serie con los resultados barajados (mismas fechas, mismos números, orden al azar).
Barajar destruye cualquier patrón ligado al orden, la fecha o el contexto. Si la serie real
no rinde mejor que sus copias barajadas, no queda patrón de ese tipo por encontrar."""
import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from loteria.datos import cargar_historico

df = cargar_historico("data/historico.csv")
f = df["fecha"]
N = len(df)
sor = (df["sorteo"] == "Lotería Nacional").to_numpy().astype(int)
dow, mes, dia = f.dt.dayofweek.to_numpy(), f.dt.month.to_numpy(), f.dt.day.to_numpy()
pos = (dia - 1) // 7
CTX = {"mismo tipo de sorteo": sor, "mismo día y posición en el mes": dow * 10 + pos, "mismo mes": mes,
       "mismo día del mes": dia, "mismo tipo y mes": sor * 100 + mes}
INI = 300


def bateria(x, top=3, detalle=None):
    """Aciertos de top 3 de cada regla en los sorteos INI..N (cada pronóstico usa solo el pasado)."""
    h = {}
    cnt = np.zeros(100); last = np.full(100, -1); rec = np.zeros(100)
    ctxc = {k: {} for k in CTX}
    trans = np.zeros((100, 100))
    rng = np.random.default_rng(0)
    def hit(nombre, picks, y):
        h[nombre] = h.get(nombre, 0) + (y in picks)
        if detalle is not None:
            detalle.setdefault(nombre, []).append(bool(y in picks))
    for t in range(N):
        y = x[t]
        if t >= INI:
            ruido = rng.random(100) * 1e-6
            hit("más salidos (historia)", np.argsort(-(cnt + ruido))[:top], y)
            hit("menos salidos (historia)", np.argsort(cnt + ruido)[:top], y)
            hit("más salidos (últimos ~50)", np.argsort(-(rec + ruido))[:top], y)
            hit("más atrasados", np.argsort(last + ruido)[:top], y)
            hit("recién salidos", np.argsort(-(last + ruido))[:top], y)
            hit("repetir hace 1, 2 y 3", x[t - 3:t], y)
            hit("repetir hace 3, 6 y 9", x[[t - 3, t - 6, t - 9]], y)
            hit("sucesores del anterior (Markov)", np.argsort(-(trans[x[t - 1]] + cnt * 1e-3 + ruido))[:top], y)
            hit("vecinos del anterior (±1, espejo)", [(x[t-1]+1) % 100, (x[t-1]-1) % 100, (x[t-1] % 10) * 10 + x[t-1] // 10], y)
            hit("aritmética de fecha", [(dia[t] + mes[t]) % 100, dia[t], (dia[t] * mes[t]) % 100], y)
            for k, v in CTX.items():
                c = ctxc[k].get(v[t])
                sc = (c if c is not None else np.zeros(100)) + cnt * 1e-3 + ruido
                hit("más salidos en " + k, np.argsort(-sc)[:top], y)
        cnt[y] += 1; last[y] = t; rec *= 0.986; rec[y] += 1
        if t: trans[x[t - 1], y] += 1
        for k, v in CTX.items():
            ctxc[k].setdefault(v[t], np.zeros(100))[y] += 1
    return h


def extras(x):
    """Estadísticos generales de dependencia."""
    z = x - x.mean()
    ac = max(abs(np.corrcoef(z[:-L], z[L:])[0, 1]) for L in range(1, 11))
    chi = ((np.bincount(x, minlength=100) - N / 100) ** 2 / (N / 100)).sum()
    return ac, chi


if __name__ == "__main__":
    x = df["numero"].to_numpy()
    real = bateria(x); ac_r, chi_r = extras(x)
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = np.random.default_rng(1)
    sims, mx, tot, acs = {k: [] for k in real}, [], [], []
    for _ in range(R):
        xs = rng.permutation(x)
        b = bateria(xs)
        for k in b: sims[k].append(b[k])
        mx.append(max(b.values())); tot.append(sum(b.values())); acs.append(extras(xs)[0])
    n = N - INI
    print(f"{n} sorteos pronosticados | azar: {0.03 * n:.1f} aciertos por regla | {R} barajadas\n")
    print(f"{'regla':45s} {'real':>5s} {'barajado (prom.)':>17s} {'% barajadas ≥ real':>19s}")
    for k, v in sorted(real.items(), key=lambda kv: -kv[1]):
        s = np.array(sims[k]); print(f"{k:45s} {v:5d} {s.mean():17.1f} {100 * (s >= v).mean():18.0f}%")
    print(f"\nMEJOR regla:   real {max(real.values())} | barajado prom. {np.mean(mx):.1f} | {100 * np.mean(np.array(mx) >= max(real.values())):.0f}% de las barajadas igualan o superan")
    print(f"SUMA de todas: real {sum(real.values())} | barajado prom. {np.mean(tot):.1f} | {100 * np.mean(np.array(tot) >= sum(real.values())):.0f}% de las barajadas igualan o superan")
    print(f"Autocorrelación máx. (rezagos 1-10): real {ac_r:.3f} | barajado prom. {np.mean(acs):.3f} | {100 * np.mean(np.array(acs) >= ac_r):.0f}% igualan o superan")
    from scipy import stats
    print(f"Uniformidad (chi² 99 gl): {chi_r:.1f}, p = {1 - stats.chi2.cdf(chi_r, 99):.3f}")
