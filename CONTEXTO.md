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
- Último resultado registrado: **domingo 18/10/2026 → 75** (Lotería Nacional). 725 sorteos.

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
- 13/10 salió **47** (se había pronosticado 44, 73, 60). Post-mortem: solo los modelos de
  regresión lo tenían arriba (regresión lineal en el puesto 1), pero porque siempre favorecen
  números cercanos al promedio (~47–51); en el backtest aciertan el top 10 solo 5–6%, peor que
  el azar. El 47 había salido 3 veces en 722 (esperado 7.2): de los más "fríos". Reglas que lo
  "explicaban": anterior + 63, día del mes + 34, día del año + 61 (todas ~1% histórico).
- Se re-corrió `evaluar` con el dato nuevo: ranking prácticamente igual; no se recalibró nada.
- El usuario exige que Claude decida como experto y siempre use el mejor modelo, sin plantearle
  propuestas. Se agregaron 2 modelos: "Repetición reciente" (p ∝ 1 + w·[salió en los últimos k],
  k y w por máxima verosimilitud; hoy k=3, w=0.5) y "Mezcla experta" (stacking con pesos EM).
  Backtest 216 sorteos: Repetición reciente quedó 1.º (+0.007 bits, top1 2.0%, top5 8.0%,
  top10 12.8%; p=0.15, no significativo tras Holm). Mezcla experta 3.º.
- El usuario objetó (con razón) poner el 84 —salió hace 2 sorteos—. Datos por rezago exacto:
  repetir el de hace 1: 1.39%, hace 2: 0.69%, hace 3: 2.08% (p=0.01 sin corregir), hace 4: 0.97%,
  hace 5: 0.70%, hace 6: 1.53%, hace 7: 0.70%. Hace 3 y hace 6 ≈ mismo día de la semana 1 y 2
  semanas antes. Se creó "Repetición por rezago" (tasa por rezago con encogimiento Beta hacia 1%):
  backtest +0.004 bits, top1 2.8%, top5 7.7%, top10 12.6% (p=0.35, no significativo).
- MÉTODO FIJO (explicado al usuario): registrar → evaluar (walk-forward 216 sorteos, 34 modelos)
  → pronosticar con el consenso de los 2 mejores (`--modelos 2 --top 3`). La bitácora
  `data/bitacora.csv` guarda cada top 3 entregado; `registrar` muestra el marcador acumulado.
  El usuario NO quiere propuestas: Claude decide y entrega.
- Ranking actual: 1) Repetición reciente, 2) Repetición por rezago (top 3 en backtest 6.1% y
  5.6% vs 3% azar). Frecuencia histórica (usada el 13/10) solo 3.1%.
- 16/10: pronóstico 61, 47, 80 → salió **57** (fallo). Marcador bitácora: 0 de 2 en top 3.
  Post-mortem: solo "Números atrasados" lo tenía alto (puesto 4, 3.6%); el 57 llevaba 371 sorteos
  sin salir. Pero en 574 sorteos, el ganador está en el 5% más atrasado solo 3.1% de las veces
  (azar 5%): el atraso NO predice; 47 y 57 seguidos fue coincidencia. Atrasados sigue en el
  puesto 16 y 31 del ranking (top10 7.4% vs 10%). No se ajustó hacia esa regla.
- Recalibración (evaluar con 724 sorteos): mismo top 2 (repetición reciente y por rezago).
- **Pronóstico Lotería Nacional domingo 18/10/2026: 1) 84, 2) 57, 3) 97** (84 y 97 = los
  dos domingos anteriores, rezagos 3 y 6; 57 = el último).
- 18/10: pronóstico 84, 57, 97 → salió **75**. Marcador bitácora: 0 de 3 (con 3% por sorteo,
  0 de 3 pasa el 91% de las veces). 47, 57 y 75 seguidos estaban entre los que MENOS han salido.
- Calibración nueva: "Sistema adaptativo" (promedio bayesiano de 11 modelos con olvido 0.98;
  el peso de cada modelo se actualiza con cada sorteo). Backtest: 2.º en log-score (+0.004 bits),
  mejor top 10 de todos (15.2%), top 3 5.5% (últimos 30: 6.7%). Pesos hoy: fríos por conteo 43%,
  repetición 17%+19%, uniforme 17%. Es el MODELO OFICIAL desde ahora:
  `python -m loteria pronosticar --fecha DD/MM/AAAA --sorteo X --modelo "Sistema adaptativo" --top 5`
- El usuario confirmó que el 00 (y el 99) existen en estos sorteos. El 00 no ha salido en 725.
- El usuario pide buscar TODAS las correlaciones posibles. Se creó `loteria/correlaciones.py`:
  8871 correlaciones (rachas alto/bajo y par/impar, n.º de sorteo del mes, mes × n.º de sorteo,
  dígitos de los 3 anteriores, saltos, etc.) contra alto/bajo, par/impar, decena, unidad y número.
  Descubiertas en 2021-24, validadas en 2025-26: 0 sobreviven a pruebas múltiples; de las 300
  más fuertes solo 22% superan el azar en 2025-26 (promedio −0.79 puntos). Tras 2 bajos
  seguidos, el siguiente es bajo 46.1% (p=0.35). El 5.º sorteo de agosto nunca repitió número.
- Modelo "Correlaciones del barrido completo": tenía al 75 en el puesto 2 el 18/10, pero en
  216 sorteos queda por debajo del azar (top 10 7.4%). Entra al sistema adaptativo, que hoy le da 0%.
- **Pronóstico Chances martes 20/10/2026: 1) 00, 2) 62, 3) 47.**
- Siguiente: registrar 20/10 y pronosticar viernes 23/10 (Chances).

## Notas técnicas
- Instalar dependencias: `pip install --break-system-packages -r requirements.txt`. Si
  falla por "hash mismatch" (descarga cortada de xgboost), instalar por partes y
  reintentar xgboost.
- `resultados/` (salidas de evaluar/pronosticar) está en `.gitignore`; si no existe
  `resultados/evaluacion.csv`, correr `python -m loteria evaluar` antes de `pronosticar`.
