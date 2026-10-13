"""CLI del laboratorio de pronóstico.

    python -m loteria analizar     --datos data/historico.csv
    python -m loteria evaluar      --datos data/historico.csv
    python -m loteria pronosticar  --datos data/historico.csv --fecha 2026-10-18 [--sorteo X]
    python -m loteria registrar    --datos data/historico.csv --fecha 2026-10-18 --numero 47 [--serie 512]
    python -m loteria sintetico    --salida data/sintetico.csv     # datos falsos para probar
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import aleatoriedad, evaluacion
from .datos import N_NUMEROS, cargar_historico, generar_sintetico
from .modelos import catalogo

RES = Path("resultados")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 60)


def _contexto(df, fecha, sorteo):
    if sorteo is None:
        sorteo = df["sorteo"].iloc[-1] if df["sorteo"].nunique() == 1 else df["sorteo"].mode()[0]
    fecha = pd.to_datetime(fecha, dayfirst=True)
    return pd.DataFrame({"fecha": [fecha], "numero": [np.nan], "sorteo": [sorteo]})


DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def cmd_analizar(a):
    df = cargar_historico(a.datos)
    print(f"\n{len(df)} sorteos | {df.fecha.min():%d/%m/%Y} → {df.fecha.max():%d/%m/%Y} | "
          f"sorteos: {sorted(df.sorteo.unique())}\n")
    r = aleatoriedad.analizar(df)
    print(r.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    n_senal = (r["senal"] == "SÍ").sum()
    print("\nConclusión:", "hay" if n_senal else "NO hay",
          "evidencia estadística (corregida por pruebas múltiples) de que el resultado dependa de "
          "la fecha, el día, el sorteo o los resultados anteriores.")
    RES.mkdir(exist_ok=True)
    r.to_csv(RES / "aleatoriedad.csv", index=False)


def cmd_evaluar(a):
    df = cargar_historico(a.datos)
    modelos = catalogo(rapido=a.rapido)
    print(f"\nWalk-forward con {len(modelos)} modelos sobre {len(df)} sorteos...")
    inicio, reg = evaluacion.walk_forward(df, modelos, n_prueba=a.n_prueba,
                                          reentrenar_cada=a.reentrenar_cada)
    r = evaluacion.resumen(reg)
    print(f"\nPeriodo de prueba: {df.fecha.iloc[inicio]:%d/%m/%Y} → {df.fecha.iloc[-1]:%d/%m/%Y} "
          f"({len(df) - inicio} sorteos, nunca vistos al entrenar)")
    print("Azar puro esperado: top1=1%, top5=5%, top10=10%, rango medio=50.5, ganancia=0 bits\n")
    cols = ["modelo", "log_score", "ganancia_vs_azar_bits", "acierto_top1_%", "acierto_top5_%",
            "acierto_top10_%", "rango_medio_real", "p_valor_vs_azar", "p_holm", "supera_azar"]
    print(r[cols].to_string(float_format=lambda x: f"{x:.3f}"))
    mejor = r.iloc[0]
    print(f"\nMejor ajuste fuera de muestra: {mejor.modelo}  "
          f"(ganancia {mejor.ganancia_vs_azar_bits:+.4f} bits, p Holm = {mejor.p_holm:.3f})")
    if mejor.supera_azar == "no":
        print("→ Ningún modelo supera al azar de forma estadísticamente significativa. "
              "Las diferencias entre modelos son ruido.")
    RES.mkdir(exist_ok=True)
    r.to_csv(RES / "evaluacion.csv", index=False)


def cmd_pronosticar(a):
    df = cargar_historico(a.datos)
    ctx = _contexto(df, a.fecha, a.sorteo)
    # Calendario regular JPS: martes y viernes (Chances), domingo (Lotería Nacional)
    entre = pd.date_range(df.fecha.max() + pd.Timedelta(days=1), ctx.fecha[0] - pd.Timedelta(days=1))
    pendientes = [d for d in entre if d.dayofweek in (1, 4, 6)]
    if pendientes and not a.permitir_pendientes:
        print(f"\n[alto] Faltan resultados reales de sorteos anteriores a {ctx.fecha[0]:%d/%m/%Y}: "
              + ", ".join(f"{DIAS[d.dayofweek]} {d:%d/%m}" for d in pendientes)
              + ".\nRegistre primero esos resultados con 'registrar' (o use --permitir-pendientes).")
        return
    ev_path = RES / "evaluacion.csv"
    ranking = pd.read_csv(ev_path) if ev_path.exists() else None
    modelos = {m.nombre: m for m in catalogo(rapido=a.rapido)}
    if ranking is not None:
        nombres = [n for n in ranking["modelo"] if n in modelos and not n.startswith("Uniforme")][: a.modelos]
    else:
        print("[aviso] No hay resultados/evaluacion.csv; ejecute 'evaluar' primero. "
              "Se usan modelos por defecto.")
        nombres = [n for n in modelos if not n.startswith("Uniforme")][: a.modelos]
    print(f"\nPronóstico para {DIAS[ctx.fecha[0].dayofweek]} {ctx.fecha[0]:%d/%m/%Y} (sorteo: {ctx.sorteo[0]}) "
          f"con {len(df)} sorteos de historia\n")
    tabla = {}
    for nom in nombres:
        p = modelos[nom].fit(df).predict_proba(ctx)[0]
        top = np.argsort(-p)[: a.top]
        tabla[nom] = p
        print(f"■ {nom}")
        if ranking is not None:
            fila = ranking.set_index("modelo").loc[nom]
            print(f"  (backtest: top10 {fila['acierto_top10_%']:.1f}% vs 10% azar; "
                  f"supera azar: {fila['supera_azar']})")
        print("  " + "  ".join(f"{n:02d} ({100 * p[n]:.2f}%)" for n in top))
    # Consenso: promedio de las distribuciones de los modelos mostrados
    cons = np.mean(list(tabla.values()), axis=0)
    top = np.argsort(-cons)[: a.top]
    print("\n■ CONSENSO de los modelos anteriores")
    print("  " + "  ".join(f"{n:02d} ({100 * cons[n]:.2f}%)" for n in top))
    print(f"\nReferencia: cualquier número tiene 1.00% por azar puro. Los modelos dicen que este top "
          f"{a.top} acumula {100 * cons[top].sum():.1f}% (vs {a.top}% del azar), pero eso es lo que el "
          f"modelo cree, no lo que demostró: vea la columna 'supera azar' del backtest.")
    RES.mkdir(exist_ok=True)
    out = pd.DataFrame({"numero": [f"{i:02d}" for i in range(N_NUMEROS)]})
    for nom, p in tabla.items():
        out[nom] = p
    out["consenso"] = cons
    out.sort_values("consenso", ascending=False).to_csv(
        RES / f"pronostico_{ctx.fecha[0]:%Y%m%d}.csv", index=False)


def cmd_registrar(a):
    """Agrega el resultado real y hace el 'post-mortem': qué probabilidad y rango le dio
    cada modelo (entrenado sin ese dato) al número que realmente salió."""
    ruta = Path(a.datos)
    df = cargar_historico(ruta)
    ctx = _contexto(df, a.fecha, a.sorteo)
    print(f"\nPost-mortem del {ctx.fecha[0]:%d/%m/%Y}: salió el {a.numero:02d}\n")
    filas = []
    for m in catalogo(rapido=a.rapido):
        p = m.fit(df).predict_proba(ctx)[0]
        g, e = np.sum(p > p[a.numero]), np.sum(p == p[a.numero])
        filas.append((m.nombre, 100 * p[a.numero], 1 + g + 0.5 * (e - 1)))
    r = pd.DataFrame(filas, columns=["modelo", "prob_asignada_%", "rango_del_real"])
    print(r.sort_values("rango_del_real").to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n(rango 1 = era el número favorito del modelo; por azar el rango esperado es ~50)")

    crudo = pd.read_csv(ruta, sep=None, engine="python", dtype=str)
    cols = {c.lower(): c for c in crudo.columns}
    nueva = {c: "" for c in crudo.columns}
    for k, v in (("fecha", ctx.fecha[0].strftime("%d/%m/%Y")), ("numero", f"{a.numero:02d}"),
                 ("número", f"{a.numero:02d}"), ("serie", "" if a.serie is None else str(a.serie)),
                 ("sorteo", ctx.sorteo[0])):
        if k in cols:
            nueva[cols[k]] = v
    crudo = pd.concat([crudo, pd.DataFrame([nueva])], ignore_index=True)
    crudo.to_csv(ruta, index=False)
    print(f"\nResultado agregado a {ruta}. Ejecute 'evaluar' para recalcular el ranking de modelos.")


def cmd_patrones(a):
    from .patrones import validar

    df = cargar_historico(a.datos)
    A, top, ne, npr, todas = validar(df)
    print(f"\n{A.shape[1]} reglas | entrenamiento {ne} sorteos | prueba {npr} sorteos (azar = 1%)\n")
    print(top.to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"\nPromedio de las {len(top)} mejores reglas en prueba: {top['aciertos_prueba_%'].mean():.2f}% "
          f"| promedio de todas: {todas['aciertos_prueba_%'].mean():.2f}%")
    if a.fecha:
        t = df.index[df.fecha == pd.to_datetime(a.fecha, dayfirst=True)]
        if len(t):
            t = t[0]
            hit = A.columns[A.loc[t]]
            print(f"\n{df.fecha[t]:%d/%m/%Y} salió {df.numero[t]:02d}; reglas que lo habrían acertado:")
            for h in hit:
                print(f"  - {h:45s} acierto en toda la historia: {100 * A[h].iloc[20:].mean():.2f}%")


def cmd_consolidar(a):
    from .consolidar import consolidar

    base, conflictos = consolidar(Path(a.carpeta))
    if len(conflictos):
        print("[aviso] Fechas con números distintos entre archivos (se usó el archivo específico):")
        print(conflictos.to_string(index=False))
    Path(a.salida).parent.mkdir(exist_ok=True, parents=True)
    base.assign(fecha=pd.to_datetime(base["fecha"]).dt.strftime("%d/%m/%Y"),
                numero=base["numero"].map("{:02d}".format)).to_csv(a.salida, index=False)
    print(f"{len(base)} sorteos únicos → {a.salida}")
    print(base.groupby(["sorteo", "fuente"]).size().to_string())


def main():
    ap = argparse.ArgumentParser(prog="loteria")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def comun(p):
        p.add_argument("--datos", default="data/historico.csv")
        p.add_argument("--rapido", action="store_true", help="omite boosting (más lento)")
        return p

    comun(sub.add_parser("analizar")).set_defaults(f=cmd_analizar)
    p = comun(sub.add_parser("evaluar"))
    p.add_argument("--n-prueba", type=int, default=None)
    p.add_argument("--reentrenar-cada", type=int, default=25)
    p.set_defaults(f=cmd_evaluar)
    p = comun(sub.add_parser("pronosticar"))
    p.add_argument("--fecha", required=True)
    p.add_argument("--sorteo", default=None)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--modelos", type=int, default=5, help="cuántos de los mejores modelos mostrar")
    p.add_argument("--permitir-pendientes", action="store_true",
                   help="pronosticar aunque falten resultados de sorteos intermedios")
    p.set_defaults(f=cmd_pronosticar)
    p = comun(sub.add_parser("registrar"))
    p.add_argument("--fecha", required=True)
    p.add_argument("--numero", type=int, required=True)
    p.add_argument("--serie", type=int, default=None)
    p.add_argument("--sorteo", default=None)
    p.set_defaults(f=cmd_registrar)
    p = comun(sub.add_parser("patrones", help="minería de reglas y validación fuera de muestra"))
    p.add_argument("--fecha", default=None, help="sorteo a 'explicar' (dd/mm/aaaa)")
    p.set_defaults(f=cmd_patrones)
    p = sub.add_parser("consolidar", help="une los 3 Excel de la JPS en data/historico.csv")
    p.add_argument("--carpeta", default=".")
    p.add_argument("--salida", default="data/historico.csv")
    p.set_defaults(f=cmd_consolidar)
    p = sub.add_parser("sintetico")
    p.add_argument("--salida", default="data/sintetico.csv")
    p.add_argument("--n", type=int, default=1500)
    p.set_defaults(f=lambda a: (Path(a.salida).parent.mkdir(exist_ok=True, parents=True),
                                generar_sintetico(a.n).assign(
                                    fecha=lambda d: d.fecha.dt.strftime("%d/%m/%Y"))
                                .to_csv(a.salida, index=False), print(f"Escrito {a.salida}")))
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
