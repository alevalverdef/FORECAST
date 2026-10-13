# Contexto para continuar la conversación (traspaso de sesión)

Este archivo resume la sesión anterior para que la nueva continúe sin que el usuario
tenga que repetir nada. Léelo completo antes de responder.

## Quién es el usuario y cómo trabajar con él
- Alejandro. Escribe en español de Costa Rica y trata de "usted/ud": responder igual.
- Quiere respuestas **directas y cortas**, sin jerga técnica (no hablar de "push", "403",
  "tokens"). Si algo falla, decir qué pasa, qué hacer y en pocas líneas.
- Espera que Claude actúe como experto en estadística, matemáticas y forecast, y lidere.
- Formato que pidió para pronósticos: **top 3, el 1 es el más probable**.
- Le molestó que se pronosticara el domingo 18/10 sin tener los resultados del martes
  13/10 y viernes 16/10. Regla: **pronosticar un sorteo a la vez**, después de registrar
  el resultado real del anterior (la herramienta ya lo bloquea).

## Objetivo del proyecto
Banco de pruebas para pronosticar demanda con alta aleatoriedad y pocas variables, usando
como caso el número (00–99) de la JPS de Costa Rica: Lotería Nacional (domingo) y Chances
(martes y viernes). El usuario dice que ambos juegos se juegan igual.

## Datos
- `Histórico general.xlsx`, `Histórico loteria.xlsx`, `Histórico Chances.xlsx` (en la raíz,
  subidos por el usuario a `main`). `python -m loteria consolidar` los une en
  `data/historico.csv`: **722 sorteos únicos, 30/04/2021 → 11/10/2026**.
- 24 fechas repetidas entre archivos, todas con el mismo número (sin conflictos).
- En Chances las columnas D–F son copia de A–C. Fechas mezcladas (fecha Excel y texto
  "28 Aug, 2026"), validadas contra el día de la semana: sin errores.
- Sorteos en días especiales (lunes 02/02/2026, algunos sábados, un miércoles): se conservan.
- Hueco: no hay Chances entre 11/04/2025 y 29/08/2025.
- Último resultado registrado: **domingo 11/10/2026 → 84** (Lotería Nacional).

## Hallazgos (ya comunicados al usuario)
- Pruebas de aleatoriedad (`analizar`): ninguna variable (día, mes, quincena, año, tipo de
  sorteo, número anterior) tiene relación con el resultado tras corrección de Holm.
- Backtest walk-forward de 31 modelos (`evaluar`, 216 sorteos 07/01/2025 → 11/10/2026):
  el mejor log-score fue el azar uniforme; ninguno lo supera significativamente. El mejor
  no-uniforme es "Frecuencia histórica (Laplace a=20)" con top-10 de 14.4% vs 10%.
- Además: ARIMA(1,0,1), ARIMA(2,1,2) y suavizamiento exponencial, por debajo del azar;
  árbol y Random Forest sin límite aciertan 100% el pasado y ~1% el futuro.
- Minería de 353 reglas (`patrones`): las 15 mejores de 2021–2024 (hasta 2.9%) aciertan
  0.81% en 2024–2026. Para el 84 del 11/10 cuatro reglas lo "explicaban" (repetir el de
  hace 7 sorteos, anterior 61 + 23, día del año 284, día del mes + 73); todas ~1% en el
  histórico.

## Estado actual / pendientes
- **Pronóstico entregado para Chances martes 13/10/2026: 1) 44, 2) 73, 3) 60**
  (modelo Frecuencia histórica a=20).
- Esperando del usuario el **resultado real del martes 13/10**. Al recibirlo:
  `python -m loteria registrar --fecha 13/10/2026 --numero NN --sorteo Chances`,
  luego dar el top 3 del **viernes 16/10 (Chances)** con
  `python -m loteria pronosticar --fecha 16/10/2026 --sorteo Chances --modelos 1 --top 3`.
  Opcional: volver a correr `evaluar` (~5 min) para actualizar el ranking.
- La sesión anterior no podía subir cambios a GitHub (rechazo de escritura); por eso se
  abrió esta. Desarrollar en la rama `claude/lottery-prediction-model-wh4avp`.

## Notas técnicas
- Instalar dependencias: `pip install --break-system-packages -r requirements.txt`. Si
  falla por "hash mismatch" (descarga cortada de xgboost), instalar por partes y
  reintentar xgboost.
- `resultados/` (salidas de evaluar/pronosticar) está en `.gitignore`; si no existe
  `resultados/evaluacion.csv`, correr `python -m loteria evaluar` antes de `pronosticar`.
