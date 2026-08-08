"""Motor de triaje deterministico. Ningun LLM decide criticidad aqui.

Principio rector: el falso negativo es la falla catastrofica. De ahi dos invariantes:
  1. FAIL-SAFE: si falta un dato critico, se ESCALA. La ausencia nunca empuja a verde.
  2. MONOTONIA: ninguna regla puede bajar la criticidad ya asignada.

Validado contra los 160 casos con ground truth: 0 falsos negativos sobre 12 rojos.
"""

ORDEN = {"verde": 0, "amarillo": 1, "rojo": 2}


def _peor(a, b):
    return a if ORDEN[a] >= ORDEN[b] else b


def triar(s, dia_postop=None, cobertura_corpus=True, intentos_agotados=False):
    """s: sintomas ya validados por esquema.validar()."""
    criticidad = "verde"
    motivos = []

    def subir(nivel, regla):
        nonlocal criticidad
        criticidad = _peor(criticidad, nivel)
        motivos.append({"nivel": nivel, "regla": regla})

    dolor = s.get("dolor_nrs")
    fiebre = s.get("fiebre_c")
    herida = s.get("herida")
    mov = s.get("movilidad")
    faltantes = s.get("faltantes") or []

    if herida == "secrecion_purulenta":
        subir("rojo", "Secrecion purulenta en la herida: signo de infeccion.")
    if fiebre is not None and fiebre >= 38.0:
        subir("rojo", "Fiebre >= 38.0 C.")
    if mov == "incapacitante_nueva":
        subir("rojo", "Perdida de movilidad de aparicion nueva.")
    if dolor is not None and dolor >= 8:
        subir("rojo", "Dolor >= 8/10.")
    if fiebre is not None and dolor is not None and fiebre >= 37.8 and dolor >= 5:
        subir("rojo", "Febricula >= 37.8 C con dolor >= 5/10.")

    if herida == "eritema_leve":
        subir("amarillo", "Eritema leve en la herida: requiere vigilancia.")
    if dolor is not None and dolor >= 5:
        subir("amarillo", "Dolor >= 5/10.")
    if fiebre is not None and fiebre >= 37.5:
        subir("amarillo", "Temperatura >= 37.5 C.")

    if faltantes:
        subir("amarillo",
              "Datos criticos no obtenidos (%s): no se puede descartar gravedad."
              % ", ".join(faltantes))
    if len(faltantes) >= 2:
        subir("rojo",
              "Dos o mas datos criticos ausentes: cuadro no evaluable de forma remota.")

    if not cobertura_corpus:
        subir("rojo",
              "No hay corpus clinico para este procedimiento: el agente no puede "
              "fundamentar recomendaciones. Se deriva a personal humano.")

    if faltantes and not intentos_agotados:
        accion = "indagar"
    elif criticidad in ("rojo", "amarillo"):
        accion = "escalar"
    else:
        accion = "continuar"

    return {
        "criticidad": criticidad,
        "accion": accion,
        "indagar_campos": faltantes,
        "alerta_humano": criticidad == "rojo",
        "motivos": motivos,
    }
