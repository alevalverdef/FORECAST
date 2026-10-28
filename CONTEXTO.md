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

- REGLA DEL USUARIO: nunca usar ni mencionar criterios basados en promedios (regresiones que
  favorecen números cercanos a 50). Parte de que todos los números tienen la misma probabilidad.
  Los 4 modelos de regresión de cantidad se retiraron del catálogo. Ningún pronóstico oficial los usó.

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
- Último resultado registrado: **martes 27/10/2026 → 39** (Chances, serie 843). 729 sorteos.
  (La JPS publica 1.º, 2.º y 3.er lugar; el histórico solo tiene el 1.º. 20/10: 85, 38, 64.
  23/10: 20, 63, 61 — el 61 del pronóstico salió en 3.er lugar.)

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
- El usuario cuestionó (con razón) poner el 00 "porque nunca ha salido". Error de Claude: el
  sistema adaptativo le dio 43% de peso a "fríos por conteo" por solo 3 sorteos recientes, aunque
  en 216 sorteos ese modelo está por debajo del azar. Se descarta el sistema adaptativo como
  oficial. REGLA: el pronóstico usa solo modelos que le ganan al azar en el backtest de 216
  sorteos (hoy: repetición reciente + repetición por rezago, consenso 50/50).
- **Pronóstico CORREGIDO Chances martes 20/10/2026: 1) 47, 2) 75, 3) 28.** Claude se mantiene en
  él (dejó de cambiar de modelo tras cada fallo, que era perseguir ruido).
- Regresión con TODOS los factores (467 variables del barrido, `python loteria/regresion_total.py`):
  logística L1 con penalización elegida por validación cruzada y LightGBM, walk-forward 216
  sorteos, para alto/bajo, par/impar, decena y unidad. Ninguna supera al azar; la L1 elimina
  todas las variables (alto/bajo exactamente 50%).
- 20/10: pronóstico 47, 75, 28 → salió **85** (número común: 7 veces antes, justo el promedio).
  Marcador: 0 de 4. Usuario muy molesto. Claude le dijo con claridad que ningún modelo puede
  pronosticar este sorteo y que no va a seguir cambiando de modelo tras cada fallo.
- Pronóstico viernes 23/10 (mismos 2 modelos validados): 57, 85, 61 (guardado en bitácora).
- El usuario insiste en frecuencia POR CONTEXTO (p. ej. "3.er sorteo de X mes"). Se mostró para el
  85 del 20/10 en 12 contextos (mes, día del mes, día semana, sorteo, semana del mes, n.º sorteo del
  mes, mes×n.º sorteo, mes×día semana, mes×semana, mes×día×semana, mes×sorteo). Contextos
  específicos tienen solo 5 sorteos previos (9.º sorteo de octubre: 5). Regla "jugar los 3 más
  frecuentes del contexto", validada 2023-26 (≈500 sorteos): entre 1.7% y 4.0% (azar 3%); los
  contextos más específicos son los PEORES (mes×n.º sorteo 2.0%, mes×día×semana 1.7%).
- Para el 23/10, el modelo que combina TODOS los contextos (Correlaciones del barrido completo)
  da 73, 33, 75 (2.96%, 2.79%, 2.56%); su historial en 216 sorteos: top 3 3.7%, top 10 7.4%.
  Se le mostró al usuario como "la respuesta por contextos"; el oficial sigue siendo 57, 85, 61.
- 23/10: salió **20** (1.er lugar). Marcador 1.er lugar: 0 de 5. El usuario dice que el sistema está
  "en un loop" con 47, 57, 75, 85 y que eso son promedios: es la regla de repetición de recientes
  (diseño del modelo), no promedios; su ventaja en backtest NO es significativa (p=0.15).
- Idea del usuario: "qué modelo aplicar según el día". Probado (`loteria/seleccion_por_contexto.py`,
  16 modelos, 9 reglas de selección, 216 sorteos de prueba): la mejor regla es escoger el modelo
  con mejor historial en el MISMO TIPO DE SORTEO: +0.0195 bits, top 3 5.4%, p=0.032 (sin corregir
  por las 9 reglas). Chances → Repetición por rezago; Lotería Nacional → Frecuencia histórica a=20
  (la repetición NO funciona en Lotería Nacional: −0.012 bits). NUEVA REGLA OFICIAL.
- Pronóstico Lotería Nacional domingo 25/10/2026 (corregido con la nueva regla): 44, 73, 60.
  CAMBIADO a pedido del usuario a **09, 67, 11**: los más salidos en el cluster "LN, último
  domingo de un mes con 4 domingos" (40 sorteos; 44 nunca salió ahí). Validación LN 2023-26:
  cluster 7/188 vs frecuencia global 10/188 (azar ≈5.6): sin diferencia real. Claude dijo que
  le daría un solo top 3 oficial por sorteo, sin mezclar resultados de experimentos.
  Análisis dentro del cluster (40): decena/unidad ya salidas en el mes = lo esperado (26 vs 24.3;
  23 vs 23.5; el 65% "crudo" era porque a fin de mes ya hubo más sorteos). Misma decena que el
  sorteo anterior: 0/40 (esperado 4, p=0.03; 0/20 en cada mitad; control 5 domingos 1/21) → único
  patrón aplicable: excluir la decena del anterior (20 → 20-29). Misma unidad que el anterior
  7/40 (p=0.11, no). Últimos domingos de octubre 2021-25: 26, 29, 84, 83, 77 (sin repetición).
  Diferencias vs anterior: ninguna se repite más de 2 veces (normal). Top 3 se mantiene 09, 67, 11.
  Son los más salidos en TODA la historia (escenario que el usuario llama "la estupidez más grande").
  Él quiere "el más salido en el contexto idéntico". Validado solo en domingos LN 2023-26 (≈189):
  frecuencia global 10/188 (5.3%); LN×octubre×último domingo 9/189 (4.8%, pero solo 5 sorteos
  previos: 26, 29, 77, 83, 84, una vez cada uno); LN×mes 8/189; LN×último domingo 8/189;
  LN×semana 5/189; LN×n.º sorteo 4/189. Azar ≈5.7/189. Todo dentro del ruido.
- Pedido del usuario: "días análogos" con muchos factores iterados. Construido `loteria/analogos.py`:
  cada sorteo pasado vota por su número con peso exp(Σ w·coincidencias) en 12 factores (tipo,
  mes, semana del mes, último del mes, n.º de sorteo, día, quincena, decena/alto del anterior,
  decena del anterior del mismo tipo, decenas ya salidas en el mes, año); pesos aprendidos por
  máxima verosimilitud en 2021-24. Prueba 2025-26 (223 sorteos): −0.014 bits, top3 2.7%
  (azar 3%), p=0.86; solo LN: 3.3%. No supera al azar. Para 25/10 daba 09, 67, 73.
  Ajustado para "haber predicho" 2021-24: 5.4% top3 en esos mismos sorteos vs 2.7% en 2025-26.
  Reajustando pesos cada 10 sorteos con resultados reales: 3.6% top3, −0.012 bits. Puesto que dio
  al real en los últimos 6 (ajustado justo antes): 20, 83, 64, 98, 61, 46.
- Pedido: incluir años anteriores, correlaciones positivas y negativas, seleccionar factores e
  iterar. `loteria/factores.py`: logit condicional por número con 40 factores (mismo sorteo hace
  1-3 años: número/decena/unidad; mismo mes/n.º de sorteo/semana/±7 días en años anteriores;
  frecuencia por contexto; repeticiones hace 1-7; atraso; últimos 30; vecinos, espejo, paridad,
  decena/número ya salido en el mes...). Entrenamiento 2022-23, validación 2024, prueba 2025-26.
  Selección hacia adelante eligió 15 factores (+0.059 bits en validación) → prueba −0.051 bits,
  top3 3.1%, p=0.94. Los 9 factores consistentes en 2022-23 y 2024 juntos: −0.043 bits en prueba.
  Cada factor solo en prueba: entre −0.048 y +0.011 bits. Varios cambian de signo entre periodos.

## MÉTODO OFICIAL DESDE 25/10 (definido por el usuario)
- `loteria/patron_dia.py`: 1) cluster de días similares al sorteo (mismo tipo, mismo día de la
  semana, misma posición en el mes / último + cantidad de ese día en el mes); 2) puntaje = veces en
  el cluster; 3) filtros que se sostienen en ambas mitades del cluster (hoy: no repetir la decena
  del anterior); 4) la frecuencia general (p. ej. 44) SOLO como ajuste/desempate, nunca como base.
  El usuario NO quiere que se elija por "aciertos históricos"; quiere el patrón del día. Top 3-10 a
  criterio de Claude.
- 25/10 (cluster 40): 09, 67, 11, 73, 35 → salió **40** (2.º 95, 3.º 97). El 40 nunca salió en el
  cluster y solo 3 veces en la historia: puesto 85 con el método. Marcador 1.er lugar: 0 de 6.
- 27/10 Chances (último martes de un mes con 4 martes; cluster 30; filtro de decena NO aplica,
  8/30): 91, 22, 73, 07, 31 → salió **39** (2.º 84, 3.º 19). Marcador 1.er lugar: 0 de 7.
- Se empezó a guardar 1.º/2.º/3.er lugar con series en `data/premios_2y3.csv` (desde 20/10).
- 30/10 Chances (último viernes de un mes con 5 viernes; cluster 18): 30, 94, 34, 44, 31.

## Notas técnicas
- Instalar dependencias: `pip install --break-system-packages -r requirements.txt`. Si
  falla por "hash mismatch" (descarga cortada de xgboost), instalar por partes y
  reintentar xgboost.
- `resultados/` (salidas de evaluar/pronosticar) está en `.gitignore`; si no existe
  `resultados/evaluacion.csv`, correr `python -m loteria evaluar` antes de `pronosticar`.
