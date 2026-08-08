"""Validacion estricta del JSON de sintomas.

Motivacion medida: el LLM devolvio herida="amorillito" (valor inexistente) en un
caso ROJO con secrecion purulenta. El motor de triaje no reconocio el valor,
siguio de largo y clasifico amarillo. Falso negativo.

Regla: todo valor fuera del enumerado se DESCARTA y se marca como faltante.
Nunca se corrige, nunca se adivina.
"""

HERIDA = {"normal", "eritema_leve", "secrecion_purulenta"}
MOVILIDAD = {"normal", "limitada_esperada", "incapacitante_nueva"}
CONFIANZA = {"alta", "media", "baja"}

CRITICOS = ("dolor_nrs", "fiebre_c", "herida", "movilidad")


def _num(v, lo, hi, entero=False):
    if v is None or isinstance(v, bool):
        return None
    try:
        x = int(v) if entero else float(v)
    except (TypeError, ValueError):
        return None
    return x if lo <= x <= hi else None


def validar(bruto):
    """Devuelve (sintomas_validos, rechazos)."""
    rechazos = []
    if not isinstance(bruto, dict):
        vacio = {c: None for c in CRITICOS}
        vacio.update({"confianza": None, "requiere_indagar": [],
                      "faltantes": list(CRITICOS)})
        return vacio, [{"campo": "_raiz", "valor": repr(bruto)[:80],
                        "motivo": "no es objeto JSON parseable"}]

    out = {}

    d = _num(bruto.get("dolor_nrs"), 0, 10, entero=True)
    if bruto.get("dolor_nrs") is not None and d is None:
        rechazos.append({"campo": "dolor_nrs", "valor": repr(bruto.get("dolor_nrs"))[:40],
                         "motivo": "fuera de rango 0-10 o no numerico"})
    out["dolor_nrs"] = d

    f = _num(bruto.get("fiebre_c"), 34.0, 43.0)
    if bruto.get("fiebre_c") is not None and f is None:
        rechazos.append({"campo": "fiebre_c", "valor": repr(bruto.get("fiebre_c"))[:40],
                         "motivo": "fuera de rango 34-43 o no numerico"})
    out["fiebre_c"] = f

    for campo, permitidos in (("herida", HERIDA), ("movilidad", MOVILIDAD),
                              ("confianza", CONFIANZA)):
        v = bruto.get(campo)
        if isinstance(v, str) and v.strip().lower() in permitidos:
            out[campo] = v.strip().lower()
        else:
            out[campo] = None
            if v is not None:
                rechazos.append({"campo": campo, "valor": repr(v)[:40],
                                 "motivo": "valor fuera del enumerado"})

    ind = bruto.get("requiere_indagar")
    out["requiere_indagar"] = [c for c in ind if c in CRITICOS] if isinstance(ind, list) else []

    out["faltantes"] = [c for c in CRITICOS if out.get(c) is None]
    return out, rechazos
