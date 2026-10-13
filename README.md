# Laboratorio de pronóstico — Lotería de Costa Rica (número 00–99)

Banco de pruebas para pronósticos de **alta aleatoriedad y pocas variables**, como la
demanda intermitente. Usa como caso de estudio el número del premio mayor (00–99).

## Flujo de trabajo

```bash
pip install -r requirements.txt

# 0a. Con los Excel de la JPS en la raíz (Histórico general / loteria / Chances):
python -m loteria consolidar        # → data/historico.csv, un resultado por fecha

# 0b. O coloque directamente el histórico en data/historico.csv
#    Columnas mínimas: fecha, numero    (opcionales: serie, sorteo)
#    Fechas en formato dd/mm/aaaa. Ej.:
#      fecha,numero,serie,sorteo
#      06/10/2026,47,512,Nacional

# 1. ¿Hay alguna señal en los datos? (pruebas de aleatoriedad)
python -m loteria analizar

# 2. Competencia de modelos con backtest walk-forward → elige el de mejor ajuste
python -m loteria evaluar

# 3. Pronóstico para una fecha
python -m loteria pronosticar --fecha 18/10/2026 [--sorteo Nacional]

# 4. Cuando salga el resultado real: post-mortem + se agrega al histórico
python -m loteria registrar --fecha 18/10/2026 --numero 47 --serie 512
python -m loteria evaluar     # reajusta el ranking con el dato nuevo
```

Para probar sin datos reales: `python -m loteria sintetico --salida data/sintetico.csv`
y luego `--datos data/sintetico.csv` (datos **falsos**, uniformes).

## Resultado con los datos reales (722 sorteos, 30/04/2021 → 11/10/2026)

- **Pruebas de aleatoriedad:** ninguna variable (día de semana, mes, quincena, año, tipo
  de sorteo, número anterior) muestra relación con el resultado tras la corrección de Holm.
- **Backtest (216 sorteos, 07/01/2025 → 11/10/2026):** ningún modelo supera al azar. El
  mejor log-score fue el uniforme; el más cercano (frecuencia histórica suavizada) acertó
  el top 10 en 14.4% vs 10%, diferencia compatible con suerte entre 31 modelos probados.
- **Minería de patrones (353 reglas, `python -m loteria patrones --fecha 11/10/2026`):** las
  15 reglas que más acertaban en 2021–2024 (hasta 2.9%) acertaron 0.81% en 2024–2026 (azar 1%).
  Con datos 100% aleatorios la mejor regla también "acierta" 2.2–3.4% en entrenamiento.
- **Sobreajuste:** árbol y Random Forest sin límite aciertan 100% el pasado y ~1% el futuro.
- **Series de tiempo:** ARIMA(1,0,1), ARIMA(2,1,2) y suavizamiento exponencial, por debajo del azar.
- `pronosticar` se niega a pronosticar una fecha si faltan resultados de sorteos intermedios
  (martes, viernes, domingo).
- Hueco conocido en los datos: no hay Chances entre 11/04/2025 y 29/08/2025.

## Modelos que compiten (≈30)

| Familia | Modelos |
|---|---|
| Referencia | Uniforme (1/100) — el rival a vencer |
| Frecuencia | Frecuencia histórica (2 suavizados), números "calientes" (decaimiento exponencial), números "atrasados" |
| Escenarios similares | Frecuencia condicional por día de semana, mes, día del mes, quincena, tipo de sorteo y combinaciones, con encogimiento bayesiano; Naive Bayes de contextos; dígitos (decena × unidad) por día |
| Secuencial | Cadena de Markov (número anterior → siguiente) |
| Regresión | Lineal, Poisson, Random Forest y Gradient Boosting de regresión |
| Machine learning | Logística multinomial, Random Forest, Extra Trees, KNN, Naive Bayes, red neuronal MLP, HistGradientBoosting, LightGBM, XGBoost |
| Ensamble | Promedio logarítmico de contexto + Markov + Random Forest |

Variables (features): día de semana, mes, día del mes, semana y día del año (con
senos/cosenos), quincena, fin de mes, año, tipo de sorteo y los 3 números anteriores
(con sus decenas y unidades).

## Cómo se decide el "mejor ajuste"

- **Walk-forward**: cada sorteo de prueba se pronostica solo con información anterior.
  Ajustar el modelo para que "explique" un resultado ya conocido no cuenta: eso es
  sobreajuste y siempre se puede lograr con ruido.
- **Métrica principal: log-score** (probabilidad asignada al número real). Además:
  aciertos en top 1/5/10, rango medio del número real.
- **Significancia** vs. azar (prueba t sobre la ganancia en log-score, binomial del top 10)
  con **corrección de Holm**, porque al comparar 30 modelos alguno gana por suerte.

## Expectativa honesta

En un sorteo bien hecho (bolitas/tómbola auditada), el resultado es independiente de la
fecha y de los sorteos anteriores: **ninguna** variable aporta información y el techo
teórico es 1% por número (10% para un top 10). El laboratorio está construido para
detectar señal si existiera (p. ej. un sesgo físico del equipo) y, si no existe, para
demostrarlo con números en vez de suponerlo. Esa misma disciplina es la que conviene
para la demanda: medir contra un modelo ingenuo y no aceptar mejoras que no sean
estadísticamente significativas fuera de muestra.
