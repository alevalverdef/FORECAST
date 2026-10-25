import sys, time, warnings; sys.path.insert(0,'.'); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from loteria.datos import cargar_historico
from loteria.modelos import *
from loteria.correlaciones import condiciones
df=cargar_historico('data/historico.csv'); y=df.numero.values; N=len(df)
ms=[Uniforme(),FrecuenciaGlobal(20.0),FrecuenciaReciente(30),FrecuenciaReciente(150),Atrasados(0.5),FriosConteo(5.0),
    Repeticion(),RepeticionPorRezago(),Markov(20.0),Condicional(("dia_semana",)),Condicional(("mes",)),Condicional(("mes","dia_mes"),30.0),
    Condicional(("sorteo",)),Digitos(),NaiveBayesContextual(),CorrelacionesBarrido()]
ini=100; T=N-ini; PR=np.zeros((len(ms),T,100))
t0=time.time()
for j,t in enumerate(range(ini,N)):
    h=df.iloc[:t]; c=df.iloc[t:t+1]
    for i,m in enumerate(ms): PR[i,j]=m.fit(h).predict_proba(c)[0]
print("precálculo",round(time.time()-t0),"s, modelos:",len(ms))

yy=y[ini:]; L=np.log(PR[:,np.arange(T),yy])  # (M,T)
C=condiciones(df).iloc[ini:].reset_index(drop=True)
ctx={"día de la semana":C.dia_semana.astype(str),"tipo de sorteo":C.sorteo,"mes":C.mes.astype(str),
     "n.º de sorteo del mes":C.n_sorteo_del_mes.astype(str),"semana del mes":C.semana_del_mes.astype(str),
     "racha alto/bajo":C.racha_alto_bajo.astype(str),"decena del anterior":C.decena_hace1.astype(str)}
test=216; s0=T-test
def evalua(choose):
    lp=[];t3=[]
    for j in range(s0,T):
        k=choose(j); p=PR[k,j]; lp.append(np.log2(p[yy[j]]*100))
        g=np.sum(p>p[yy[j]]); e=np.sum(p==p[yy[j]]); t3.append(np.clip((3-g)/e,0,1))
    return np.mean(lp),100*np.mean(t3)
print(f"\n{'Regla para escoger modelo':52s} {'bits':>8s} {'top3':>6s}")
for i,m in enumerate(ms[:1]+ms[6:8]):
    k=ms.index(m); print(f"  siempre {m.nombre[:43]:43s} {evalua(lambda j:k)[0]:+8.4f} {evalua(lambda j:k)[1]:5.1f}%")
print(f"  {'el que acertó mejor el sorteo anterior':50s} %+8.4f %5.1f%%"%evalua(lambda j:int(np.argmax(L[:,j-1]))))
print(f"  {'el mejor de los últimos 10 sorteos':50s} %+8.4f %5.1f%%"%evalua(lambda j:int(np.argmax(L[:,j-10:j].sum(1)))))
for nom,v in ctx.items():
    v=v.to_numpy()
    def ch(j,v=v):
        m=(v[:j]==v[j]); 
        return int(np.argmax(L[:,:j][:,m].sum(1))) if m.sum()>=5 else 0
    print(f"  {'el mejor históricamente en el mismo '+nom:50s} %+8.4f %5.1f%%"%evalua(ch))
print("\n(azar: 0 bits, top3 3.0%)")
