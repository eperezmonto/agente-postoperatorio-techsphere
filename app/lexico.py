"""Traduccion coloquial colombiano -> terminologia clinica.

Alternativa deterministica a los embeddings semanticos. Motivacion medida:
bge-m3 proyecta 5 minutos para indexar 5.697 fragmentos en la maquina de
desarrollo, lo que consume un tercio del presupuesto de la compuerta G2
(levantar en <=15 min). Este modulo cubre la misma brecha -el paciente dice
"rojita", el corpus dice "eritema"- con latencia cero y sin descargas.

Terminos extraidos de los 1.920 turnos de paciente del dataset del reto.
La frecuencia observada se anota para trazabilidad.
"""
import re
import unicodedata

# (patron, terminos clinicos, turnos donde aparece en el dataset)
MAPA = [
    (r"roj(?:it[ao]|[ao])\b",                  ["eritema", "enrojecimiento"],        136),
    (r"calorcito|acalorad[ao]",                ["febricula", "temperatura", "fiebre"], 38),
    (r"hinchad[ao]|inflamad[ao]",              ["edema", "inflamacion"],              29),
    (r"\b(?:materia|pus|amarillent[ao]|verdos[ao])\b",
                                               ["secrecion", "purulenta", "exudado"], 18),
    (r"puntada|punzada",                       ["dolor", "punzante"],                  4),
    (r"me da vueltas|maread[ao]|mareo",        ["mareo", "vertigo"],                   2),
    # Expresiones frecuentes sin conteo especifico, de vocabulario clinico general
    (r"aguadit[ao]|aguad[ao]",                 ["serosa", "secrecion"],                0),
    (r"desmadejad[ao]|sin fuerzas|debil|floj[ao]", ["astenia", "debilidad"],           0),
    (r"no me (?:pasa|baja) la comida|no puedo comer|sin apetito",
                                               ["intolerancia", "oral", "apetito"],    0),
    (r"heridit[ao]|cortadit[ao]|cicatrizit[ao]", ["herida", "incision", "sitio quirurgico"], 0),
    (r"me arde|ardor",                         ["dolor", "quemante"],                  0),
    (r"dolorcito|molestia",                    ["dolor"],                              47),
]


def _sin_tildes(t: str) -> str:
    t = unicodedata.normalize("NFD", t.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def expandir(consulta: str):
    """Devuelve (consulta_expandida, terminos_agregados).

    No reemplaza: AGREGA terminologia clinica a la consulta original, para no
    perder coincidencias exactas cuando el paciente ya usa el termino correcto.
    """
    base = _sin_tildes(consulta)
    agregados = []
    for patron, clinicos, _frec in MAPA:
        if re.search(patron, base):
            for c in clinicos:
                if c not in base and c not in agregados:
                    agregados.append(c)
    return (consulta + " " + " ".join(agregados)).strip(), agregados
