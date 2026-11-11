import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from scipy import stats
from loteria.datos import cargar_historico
from loteria.patron_dia import _rasgos
df=cargar_historico('data/historico.csv'); x=df.numero.values; s=df.sorteo.values; N=len(x)
R=_rasgos(df.fecha); dow,pos,ndia,ult=(R[c].to_numpy() for c in ("dow","pos","n_dia","ultimo"))
dm=((df.fecha.dt.day+df.fecha.dt.month)%100).to_numpy()
def cl(t,modo):
    m=(s[:t]==s[t])&(dow[:t]==dow[t])
    if modo=="actual": m&=(ult[:t]==ult[t]); m&= (ndia[:t]==ndia[t]) if ult[t] else (pos[:t]==pos[t])
    elif modo=="posicion": m&=(pos[:t]==pos[t])
    elif modo=="solo_tipo": pass
    return m
def top(t,modo,peso,usar_dm,k=3):
    c=np.bincount(x[:t][cl(t,modo)],minlength=100).astype(float); gen=np.bincount(x[:t],minlength=100)
    sc=c+peso*gen/gen.max(); o=list(np.argsort(-sc,kind="stable"))
    if usar_dm: o=[i for i in o if i!=dm[t]][:k-1]+[dm[t]]
    return o[:k]
def corre(rng,modo,peso,usar_dm): return sum(x[t] in top(t,modo,peso,usar_dm) for t in rng)
f=df.fecha
cal=[t for t in range(N) if pd.Timestamp('2023-01-01')<=f[t]<pd.Timestamp('2025-01-01')]
tst=[t for t in range(N) if f[t]>=pd.Timestamp('2025-01-01')]
print(f"Calibración 2023-24: {len(cal)} sorteos | Prueba 2025-26: {len(tst)} sorteos | azar top3: {0.03*len(cal):.1f} / {0.03*len(tst):.1f}\n")
rows=[]
for modo in ("actual","posicion","solo_tipo"):
    for peso in (0.0,0.1,1.0,3.0):
        for udm in (False,True):
            rows.append((modo,peso,udm,corre(cal,modo,peso,udm),corre(tst,modo,peso,udm)))
r=pd.DataFrame(rows,columns=["cluster","peso_frec_general","3.º=día+mes","aciertos_2023_24","aciertos_2025_26"])
print(r.to_string(index=False))
oficial=r[(r.cluster=="actual")&(r.peso_frec_general==0.1)&(r["3.º=día+mes"])].iloc[0]
best=r.sort_values("aciertos_2023_24",ascending=False).iloc[0]
print("\nOFICIAL:",dict(oficial)); print("MEJOR EN CALIBRACIÓN:",dict(best))
print("p (oficial 2025-26 vs azar):", round(stats.binomtest(int(oficial.aciertos_2025_26),len(tst),0.03,alternative='greater').pvalue,3))
