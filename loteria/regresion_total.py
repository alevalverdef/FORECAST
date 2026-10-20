import sys, warnings; sys.path.insert(0,'.'); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import lightgbm as lgb
from loteria.datos import cargar_historico
from loteria.correlaciones import condiciones
df=cargar_historico('data/historico.csv'); x=df.numero.values
C=condiciones(df).drop(columns=["anterior"]).astype(str)
X=pd.get_dummies(C).to_numpy(float); print("variables (todas las condiciones en one-hot):", X.shape[1])
N=len(x); ini=N-216; y={"alto":(x>=50).astype(int),"impar":x%2,"decena":x//10,"unidad":x%10}
def wf(make, yy, k):
    lp=[]; acc=[]
    for t in range(ini,N,25):
        m=make().fit(X[150:t],yy[150:t]); fin=min(t+25,N)
        P=m.predict_proba(X[t:fin]); cls=list(m.classes_)
        for i,tt in enumerate(range(t,fin)):
            p=np.full(k,1e-9); p[cls]=P[i]; p/=p.sum(); lp.append(np.log2(p[yy[tt]]*k)); acc.append(np.argmax(p)==yy[tt])
    return np.mean(lp), np.mean(acc)
mods={"Regresión logística L1 (C por validación cruzada)":lambda: GridSearchCV(LogisticRegression(penalty="l1",solver="saga",max_iter=2000),{"C":[0.003,0.01,0.03,0.1,0.3]},cv=TimeSeriesSplit(4),scoring="neg_log_loss"),
      "LightGBM con todas las variables":lambda: lgb.LGBMClassifier(n_estimators=80,num_leaves=7,learning_rate=0.03,min_child_samples=25,reg_lambda=5,verbose=-1)}
for obj,k in (("alto",2),("impar",2),("decena",10),("unidad",10)):
    for nom,mk in mods.items():
        b,a=wf(mk,y[obj],k); print(f"  {obj:7s} {nom:50s} ganancia {b:+.4f} bits | acierto {100*a:.1f}% (azar {100/k:.0f}%)")
