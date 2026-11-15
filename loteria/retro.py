"""Retrospectiva: para cada resultado real, ¿qué reglas lo tenían en su top 10 ANTES del sorteo?
Y luego: la regla (o reglas) que mejor 'explican' un bloque de sorteos, ¿siguen acertando en el
bloque siguiente? Uso: python loteria/retro.py [n_ultimos]"""
import sys
sys.path.insert(0, ".")
import numpy as np
from loteria.prueba_global import bateria, df, N, INI

n = int(sys.argv[1]) if len(sys.argv) > 1 else 14
x = df["numero"].to_numpy()
det = {}
bateria(x, top=10, detalle=det)   # reglas de 3 números fijos (repeticiones, vecinos, fecha) siguen con 3
reglas = list(det)
M = np.array([det[r] for r in reglas])          # (reglas, sorteos INI..N)
fechas = df["fecha"].dt.strftime("%d/%m").to_numpy()[INI:]
print(f"Últimos {n} sorteos: reglas que tenían el número real en su top 10\n")
for j in range(M.shape[1] - n, M.shape[1]):
    ok = [reglas[i] for i in range(len(reglas)) if M[i, j]]
    print(f"  {fechas[j]} salió {x[INI + j]:02d}: " + ("; ".join(ok) if ok else "ninguna"))
print(f"\nAciertos de cada regla en esos {n} sorteos vs. los {n} anteriores y todo el periodo:")
T = M.shape[1]
for i in np.argsort(-M[:, -n:].sum(1)):
    print(f"  {reglas[i]:45s} últimos {n}: {M[i, -n:].sum():2d} | {n} anteriores: {M[i, -2*n:-n].sum():2d} | tasa total {100 * M[i].mean():.1f}%")
# ¿Sirve 'usar la regla que mejor explicó los últimos n'?  Se repite a lo largo de toda la historia.
g = []; base = []
for b in range(n, T - n + 1, n):
    mejor = int(np.argmax(M[:, b - n:b].sum(1)))
    g.append(M[mejor, b:b + n].mean()); base.append(M[:, b:b + n].mean())
print(f"\nEstrategia 'usar la regla que mejor explicó los últimos {n} sorteos' en los {n} siguientes "
      f"({len(g)} bloques): acierta {100 * np.mean(g):.1f}% | promedio de todas las reglas: {100 * np.mean(base):.1f}%")
