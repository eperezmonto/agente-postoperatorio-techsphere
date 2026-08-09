# Informe final

**Tech Sphere Challenge 2026** · Erwin Perez Montoya
Agente de voz para seguimiento postoperatorio

Repositorio: https://github.com/eperezmonto/agente-postoperatorio-techsphere

---

## 1. Declaracion del modelo

**Modelo usado: `llama3.2:3b`, ejecutado localmente via Ollama.**
Pertenece a la lista permitida por la ficha tecnica del reto (compuerta G3).
No se uso ningun otro modelo de lenguaje en ninguna parte del sistema.

Voz: Web Speech API y Web Speech Synthesis API del navegador (`es-CO`).
Recuperacion: BM25 implementado en Python puro, sin modelo de embeddings.

**El proyecto no usa claves de API.** Ver `.env.example`.

### Por que este modelo y no otro

La eleccion no se hizo por reputacion sino midiendo. Se construyo un banco propio
(`bench.py`, incluido en el repositorio) que ejecuto **108 inferencias** sobre el hardware
de desarrollo comparando los dos candidatos locales de la lista permitida.

| Metrica medida | Phi-3.5 Mini | **Llama 3.2 3B** |
|---|---|---|
| Residencia en GPU (`num_ctx=2048`) | 74% | **100%** |
| Velocidad de generacion | 25.4 tok/s | **65.6 tok/s** |
| Latencia total P95 | 1751 ms | **1109 ms** |
| Tokens del mismo prompt en espanol | 1384 | **1216** |

Un segundo banco (`calidad.py`) midio la calidad de extraccion sobre **casos reales del
dataset del reto**, capa ruidosa:

| | Phi-3.5 Mini | **Llama 3.2 3B** |
|---|---|---|
| JSON valido | 3/5 | **5/5** |
| Campo `herida` correcto | 3/5 | **4/5** |
| Campo `movilidad` correcto | 3/5 | **5/5** |
| Resiste inyeccion de prompt | Si | Si |

Phi-3.5 fallo el parseo en dos de cinco casos, **uno de ellos rojo**. Un fallo de parseo en
un caso critico deja al pipeline sin datos justo donde no puede permitirselo.

La causa de la diferencia de velocidad es fisica: la RTX 3050 tiene 4 GB nominales, pero
Windows reserva alrededor de 1 GB. Quedan ~3 GB utiles. Llama 3.2 3B cabe entero; Phi-3.5
no, y las capas expulsadas a CPU cruzan el bus PCIe en cada token.

**Se predijo que Phi-3.5 seria superior en extraccion estructurada. La medicion dijo lo
contrario y mando la medicion.**

---

## 2. Arquitectura y por que

![Arquitectura](diagrama.svg)

Cinco capas con una **regla invariante: cualquier capa puede subir la criticidad; ninguna
puede bajarla.**

El principio de diseno es el mismo de todo el proyecto: **quitarle capacidad al modelo en
vez de instruirlo para que no la use mal.**

### El modelo de lenguaje no decide la gravedad

El LLM hace exactamente dos cosas: extraer sintomas estructurados del lenguaje ambiguo del
paciente, y redactar prosa empatica breve. La clasificacion verde/amarillo/rojo la produce
un motor deterministico con reglas explicitas y auditables.

Esto se decidio midiendo, no por preferencia. Con un prompt que pedia los cuatro campos a
la vez, ante la frase real *"Pues ahi mas o menos doctora, un poquito molesto no mas, uno
aguanta"*, el modelo devolvio:

```json
{"dolor_nrs": 2, "fiebre_c": null, "herida": "normal", "movilidad": "limitada_esperada"}
```

**Tres de cuatro campos alucinados.** El mas peligroso es `herida: "normal"`: afirma que
una herida esta bien cuando nadie la miro. Si esa herida tuviera secrecion purulenta,
acabamos de fabricar un falso negativo con apariencia de dato.

### Correccion: un campo por turno

Rediseñado para extraer **un solo campo por invocacion**, con ejemplos explicitos de
cuando abstenerse, el mismo modelo devuelve `null` correctamente. En una prueba de cuatro
casos acerto 3, y **el unico error fue por sobre-abstencion** (devolvio `null` donde habia
un numero). Ese error dispara una repregunta; el error contrario fabrica un dato falso.

### Validacion estricta de enumerados

Todo valor fuera del enumerado se descarta y se marca como faltante. Nunca se corrige ni
se adivina. Esta capa existe por un fallo medido: el modelo devolvio
`"herida": "amorillito"` —un valor inexistente— en el unico caso del dataset con
secrecion purulenta, el signo de alarma mas claro que existe.

### Fail-safe: la ausencia de dato nunca empuja a verde

Si falta un dato critico, se escala. Si faltan dos o mas, el cuadro se declara
**no evaluable de forma remota** y se alerta a personal humano.

Esta regla nacio de una medicion incomoda, descrita en la seccion 4.

### El agente persigue lo que no le dicen

La extraccion posterior a la llamada **no puede recuperar lo que la conversacion nunca
capturo**. En el caso `caso_tray_pac_42_00017_7` del dataset, la paciente tiene dolor
**9/10** y fiebre 37.9 °C. Lo que dice es:

> *"Ay, no, tranquila doctora, un poquito molesto no mas, nada del otro mundo, uno aguanta."*

El 9 no esta en ninguna parte del texto. Ningun modelo puede extraerlo, porque no existe.
La reparacion no es mejor extraccion ni mejor prompt: es **preguntar durante la llamada**.

> *"Necesito el numero. Si 10 es el peor dolor que ha sentido en su vida, ¿donde esta hoy?"*

Maximo 2 reintentos por campo. Agotados, se escala por dato faltante. Tope duro de 14
turnos por llamada para que la conversacion nunca se atasque.

---

## 3. Recuperacion (RAG) y trazabilidad

### Por que BM25 y no embeddings

La ficha tecnica sugiere BGE-M3, y con razon: entiende sinonimos medicos que la
coincidencia lexica no captura.

Se midio antes de descartarlo. `bge-m3` via Ollama proyecta **5 minutos** para indexar los
5697 fragmentos del corpus en el hardware de desarrollo. Sumado a la descarga del modelo
LLM (~2 GB), la del modelo de embeddings (1.2 GB), las dependencias y la ingesta de PDFs,
el arranque total queda entre 12 y 14 minutos.

**La compuerta G2 es de 15 minutos y es eliminatoria.** Cambiar algo eliminatorio por algo
puntuable es un mal negocio. Se descarto.

### El problema real que BM25 tiene, y como se cubrio

Un paciente colombiano no dice *"presento eritema periincisional"*. Dice *"la heridita esta
como rojita"*. BM25 no conecta `rojita` con `eritema`.

Se construyo un **lexico de traduccion coloquial → clinico** extraido de los 1920 turnos de
paciente del propio dataset. Frecuencias reales observadas: *rojita/rojito* aparece en 136
turnos, *calorcito/acalorado* en 38, *hinchado* en 29.

Efecto medido sobre la consulta *"La heridita esta como rojita en el borde"*:

| | Score | Fragmento recuperado |
|---|---|---|
| Sin lexico | 5.75 | resonancia magnetica (irrelevante) |
| **Con lexico** | **30.63** | *"inspeccionar el sitio de incision por si hubiera eritema..."* |

Score **5.3× mayor** y el fragmento correcto. **Latencia adicional: cero. Descargas: cero.**

Honestidad sobre el alcance: el lexico mejoro 1 de 3 consultas probadas. Las otras dos
fallan por una razon distinta —**el corpus no contiene material de educacion al paciente
sobre esos signos**, son papers academicos— y eso no lo arregla ningun recuperador.
De ahi la siguiente decision.

### Filtro por procedimiento

BM25 sin filtrar era inutilizable. Consultas sobre colecistectomia devolvian guias de
reemplazo de cadera en las dos primeras posiciones: esas guias estan escritas en espanol
llano y actuan como imanes de palabras clave.

La solucion es clinica, no tecnica: **a un paciente de vesicula no se le responde con
protocolos de cadera**. El procedimiento se conoce desde la ficha, asi que la recuperacion
se acota antes de rankear.

### Umbral de fundamento

Si ningun fragmento supera un score minimo, el agente **declara que no tiene informacion**
en vez de responder con lo primero que salio. La interfaz lo muestra explicitamente:
*"Sin fuentes por encima del umbral de fundamento"*.

### Trazabilidad

Cada respuesta clinica registra en SQLite que fragmento y que documento la sustentan, con
su score. El resumen final lista los documentos citados. Verificable en la tabla `citas`.

---

## 4. El hallazgo mas importante del proceso

El motor de triaje se valido contra los **160 casos con ground truth** del dataset:

| | |
|---|---|
| Exactitud | 81.2% |
| **Falsos negativos (rojo perdido)** | **0 de 12** |
| Amarillo degradado a verde | 0 de 25 |

Resultado excelente. Y **enganoso**.

Al probar la **cadena completa** —LLM extrae de la conversacion, luego el motor decide—
sobre los tres casos rojos reales, el resultado fue:

**3 falsos negativos de 3.**

El motor era correcto. El eslabon que lo alimenta no. Se habia validado el motor con datos
perfectos de una hoja de calculo, datos que en produccion nunca existen.

### Que se corrigio y cuanto sirvio

Con validacion de enumerados y fail-safe, uno de los tres casos pasa a rojo correctamente:
el validador rechaza `"amorillito"`, quedan dos campos criticos ausentes, y la regla de
fail-safe escala con el motivo *"cuadro no evaluable de forma remota"*.

Los otros dos **no se pueden arreglar con reglas**. En uno, el modelo extrajo dolor 6 y
fiebre 37.5 cuando los reales eran 9 y 38.0: datos equivocados sin ninguna senal de
ausencia. Ninguna regla deterministica detecta eso.

Por eso la reparacion es conversacional. El sistema no clasifica mejor lo que le dicen:
**persigue lo que no le dicen**.

### Una metrica propia que resulto inservible

Se reporto inicialmente *"detecta incertidumbre en casos rojos: 3/3"*. Al revisar la salida
cruda, el campo `confianza` valia `"baja"` en **los cinco casos**, incluido el sano.

Era una constante disfrazada de senal. Se descarto como criterio de escalacion.

---

## 5. Auditoria del corpus entregado

El corpus tiene cuatro problemas. Los cuatro se detectan y se declaran.

| Hallazgo | Como se detecta | Que hace el sistema |
|---|---|---|
| PDF escaneado sin capa de texto | Densidad de texto por pagina | Marca `escaneado`, no indexa, lo declara en la consola |
| PDF cifrado con AES | Excepcion de `pypdf` | `cryptography` en `requirements.txt` |
| Carpeta `breast_cancer` mal etiquetada | Auditoria manual del contenido | Declara sin cobertura y escala |
| Documento duplicado | Comparacion de contenido | Identificado; dedup por sha no lo captura |

### El hallazgo con consecuencia clinica

La carpeta `breast_cancer/` contiene **19 PDFs sobre cancer de cuello uterino**. Ninguno
sobre mama: se conto por nombre de archivo, 9 mencionan explicitamente cervix o cuello
uterino, **cero** mencionan mama, seno o *breast*.

El dataset de pacientes incluye **8 pacientes con Mastectomia** — el 20% de la cohorte,
32 de los 160 casos.

**No existe corpus valido para ese procedimiento.**

Un sistema que confie en el nombre de la carpeta le respondera a esas 8 pacientes con
protocolos de cancer cervical. Citara fuente. Sonara seguro. Y estara clinicamente
equivocado.

Este sistema declara el limite y escala a personal humano. Verificable en la consola,
tabla **Cobertura por procedimiento**.

### Sobre el duplicado

`Recommendations for follow-up of colorectal cancer survivors.pdf` y
`ecommendations for follow-up of colorectal cancer survivors.pdf` —nota la letra
faltante— tienen el mismo DOI, 10 paginas cada uno y **99% de similitud textual**, pero
SHA-256 distinto. La deduplicacion por hash no los captura; ambos entran al indice con 51
fragmentos cada uno.

Se identifico y esta documentado. **La deduplicacion por similitud de contenido queda como
trabajo pendiente**, descrito en la seccion 7.

---

## 6. Metricas

Hardware: Windows 11 · RTX 3050 (4 GB VRAM, ~3 GB utiles) · 64 GB RAM · Python 3.12.10.
Configuracion: `num_ctx=2048`, `top-k=3`, 100% del modelo residente en GPU.

### Llamada completa medida

| Metrica | Valor |
|---|---|
| Turnos | 11 |
| Invocaciones al LLM | 11 |
| **Latencia LLM P50** | **2059 ms** |
| **Latencia LLM P95** | **3302 ms** |
| Tokens entrada / salida | 3724 / 322 |
| Documentos citados | 2 |
| Criticidad resultante | rojo |

**Estas cifras son solo del LLM.** La latencia percibida por el paciente incluye ademas el
reconocimiento de voz y el arranque del sintetizador del navegador, que no se instrumentaron.

### Consumo

Dos invocaciones al modelo por turno: extraccion y redaccion.
Promedio ~339 tokens de entrada y ~29 de salida por invocacion.

### Costo por llamada

**Costo real: 0 USD.** El modelo corre localmente sobre hardware propio.

Extrapolado a precios de API publicos para un modelo de escala equivalente
(~0.10 USD por millon de tokens de entrada, ~0.40 de salida):

```
(3724 / 1e6) × 0.10  +  (322 / 1e6) × 0.40  ≈  0.00050 USD
```

Aproximadamente **2000 llamadas por dolar**. Los tokens son medidos; los precios son
referencia publica y varian por proveedor.

### Corpus

107 PDFs · 106 indexados · 1 declarado sin capa de texto · 5697 fragmentos ·
ingesta ~80 s · indexado BM25 ~1.1 s.

---

## 7. Que haria con dos semanas mas

**Recuperacion hibrida.** Embeddings con BGE-M3 combinados con BM25, y el indice
vectorial **precomputado y persistido** en el repositorio para no pagar los 5 minutos en
el arranque. Es la limitacion mas clara del sistema actual.

**Deduplicacion por similitud de contenido.** MinHash sobre los fragmentos normalizados,
en lugar de SHA del archivo. Resolveria el caso del documento duplicado y evitaria que dos
copias del mismo texto ocupen dos de los tres espacios de contexto.

**ASR y TTS locales.** Whisper y Piper en vez de la Web Speech API. Elimina el envio de
audio a servidores de Google, que en un contexto clinico real es un problema de
privacidad, no un detalle. Se sacrificaria tiempo de arranque, aceptable fuera de un reto
con cronometro.

**OCR para escaneados.** Tesseract recuperaria el PDF que hoy se declara como no indexable.

**Instrumentacion de latencia extremo a extremo.** Medir desde que el paciente calla hasta
que suena el audio, no solo el LLM. Es la metrica que realmente describe la experiencia.

**Validacion del extractor a escala.** Los 3991 turnos del dataset permiten medir la tasa
de abstencion correcta e incorrecta con significancia estadistica. La prueba actual de
cuatro casos es orientativa, no concluyente.

---

## 8. Riesgos identificados

**El modelo de 3B es el eslabon debil.** Toda la arquitectura esta construida asumiendo
que se equivoca: validacion estricta, fail-safe, repregunta. Un modelo mayor reduciria la
frecuencia de error pero no cambiaria el diseno, porque la asimetria clinica seguiria
exigiendo que la decision sea deterministica.

**El ASR del navegador falla en espanol colombiano.** En pruebas reales *"muy mal"* se
transcribio como *"animal"*. El sistema lo absorbio sin degradar la decision: sin numero
explicito el extractor devuelve `null` y el agente repregunta. Un extractor que rellenara
huecos habria inventado un valor a partir de una palabra que el paciente nunca dijo.

**El corpus no cubre todo lo que el dataset requiere.** Documentado en la seccion 5.
El sistema lo declara en vez de improvisar.

**Sobre-escalacion.** 30 de 123 casos verdes se clasifican como amarillos. Es deliberado:
escalar de mas cuesta una llamada, escalar de menos cuesta un paciente. Se probo una
version con umbrales por dia postoperatorio que subia la exactitud a 86.9%, pero
introducia 3 amarillos degradados a verde. **Se descarto.**

**Cierre por signo de alarma: implementado, no verificado en ejecucion.** Cuando el triaje
detecta un signo de alarma real —fiebre >= 38, dolor >= 8, secrecion purulenta—, seguir el
guion es clinicamente incorrecto, y el sistema cierra la llamada y escala de inmediato.
Verificado en ejecucion: en una llamada por voz con dolor 10/10 y fiebre 39 grados,
el agente cerro la llamada de inmediato, informo que un profesional de salud contactaria
al paciente, y no continuo preguntando por herida ni movilidad.

---

## 9. Evidencia incluida

| Archivo | Que contiene |
|---|---|
| `bench.py` | Banco de latencia. Genera las metricas de la seccion 1 |
| `calidad.py` | Banco de calidad de extraccion sobre casos reales del dataset |
| `cal_llama.txt`, `cal_phi.txt` | Salidas crudas de la comparacion de modelos |
| `METRICAS.md` | Todas las mediciones consolidadas |
| `app/triaje.py` | Reglas del motor, auditables linea por linea |
| `app/lexico.py` | Lexico coloquial con frecuencias observadas en el dataset |

Los bancos son reproducibles: cualquiera puede ejecutarlos y obtener sus propios numeros
sobre su hardware.

---

## 10. Nota de metodo

Las cifras de este informe salen de mediciones ejecutadas sobre el hardware descrito. No
hay numeros estimados presentados como medidos.

Donde una prueba fue limitada, se dice. Donde una hipotesis no se confirmo, se marca como
hipotesis. Donde una metrica propia resulto inservible, se reporta el error en lugar de
omitirlo.

El proyecto se construyo en tres dias por un desarrollador. Varias decisiones estan
condicionadas por ese plazo, y donde ese es el caso, se indica.
