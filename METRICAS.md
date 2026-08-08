# Metricas medidas — Tech Sphere Challenge 2026

## Hardware
Windows 11 · RTX 3050 (4 GB VRAM, ~3 GB utiles) · 64 GB RAM · Python 3.12.10

## Modelo
llama3.2:3b via Ollama · num_ctx=2048 · top-k=3 · 100% residente en GPU

## Llamada completa medida (paciente pac_42_00017, Colecistectomia, dia 7)
| Metrica | Valor |
|---|---|
| Turnos | 11 |
| Invocaciones al LLM | 11 |
| Latencia LLM P50 | 2059 ms |
| Latencia LLM P95 | 3302 ms |
| Tokens entrada | 3724 |
| Tokens salida | 322 |
| Documentos citados | 2 |
| Criticidad resultante | rojo |

## Comparativa de modelos (banco propio, 108 inferencias)
| | Phi-3.5 Mini | Llama 3.2 3B |
|---|---|---|
| Residencia en GPU (ctx=2048) | 74% | 100% |
| Generacion | 25.4 tok/s | 65.6 tok/s |
| Latencia total P95 | 1751 ms | 1109 ms |
| Tokens del mismo prompt | 1384 | 1216 |
| JSON valido (5 casos reales) | 3/5 | 5/5 |
| Herida correcta | 3/5 | 4/5 |
| Movilidad correcta | 3/5 | 5/5 |

## Motor de triaje — 160 casos con ground truth
Exactitud 81.2% · Falsos negativos (rojo perdido): 0 de 12 ·
Amarillo degradado a verde: 0 de 25 · Verde sobre-escalado: 30 de 123

## Corpus
107 PDFs · 106 indexados · 1 sin capa de texto (declarado) ·
5697 fragmentos · ingesta ~80 s · indexado BM25 ~1.1 s

## Descartado con medicion
bge-m3: 5 min proyectados para indexar 5697 fragmentos = 1/3 del
presupuesto de la compuerta G2 (levantar en <=15 min).
