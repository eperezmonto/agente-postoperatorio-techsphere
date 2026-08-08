"""Mapeo procedimiento -> area del corpus, y validacion de cobertura.

Hallazgo de auditoria: la carpeta 'breast_cancer' del corpus entregado contiene
documentacion de cancer de CUELLO UTERINO, no de mama. El dataset tiene 8
pacientes con Mastectomia. No existe corpus valido para ese procedimiento: el
agente debe declarar el limite y escalar, nunca responder con otra patologia.
"""

PROCEDIMIENTO_AREA = {
    "Apendicectomía":              "Appendicitis",
    "Colecistectomía":             "cholecystitis",
    "Colectomía":                  "colorectal cancer",
    "Reemplazo de cadera/rodilla": "total joint replacement",
    "Mastectomía":                 None,
}

AREAS_NO_CONFIABLES = {
    "breast_cancer": ("La carpeta contiene documentacion de cancer de cuello uterino, "
                      "no de mama. No es corpus valido para seguimiento de mastectomia."),
}

def area_de(procedimiento: str):
    return PROCEDIMIENTO_AREA.get(procedimiento)

def hay_cobertura(procedimiento: str):
    """(bool, motivo_o_area). False obliga a declarar el limite y escalar."""
    if procedimiento not in PROCEDIMIENTO_AREA:
        return False, "Procedimiento no reconocido: %r." % procedimiento
    area = PROCEDIMIENTO_AREA[procedimiento]
    if area is None:
        return False, ("No hay corpus clinico cargado para este procedimiento. "
                       + AREAS_NO_CONFIABLES.get("breast_cancer", ""))
    return True, area
