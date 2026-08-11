"""Extraccion deterministica de numeros del habla del paciente.

Motivacion medida: llama3.2:3b devolvia null ante respuestas como "9" o "un 9",
porque los ejemplos del prompt eran frases completas y un numero suelto no
coincide con el patron. El paciente daba el dato y el sistema lo perdia.

Un numero no necesita un modelo de lenguaje. Se extrae con reglas, sin latencia
y de forma auditable. El LLM queda como respaldo solo cuando no hay cifra clara.
"""
import re
import unicodedata

PALABRAS = {
    "cero": 0, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    "once": 11, "doce": 12,
    # temperaturas frecuentes en el habla
    "treinta": 30, "treinta y cinco": 35, "treinta y seis": 36,
    "treinta y siete": 37, "treinta y ocho": 38, "treinta y nueve": 39,
    "cuarenta": 40, "cuarenta y uno": 41,
}

# Expresiones que NIEGAN el dato: no se debe extraer numero de ellas
NEGACION = re.compile(
    r"no\s+(?:me\s+)?(?:lo\s+)?(?:s[ée]|tengo|me tom[ée]|he tomado|medi|revis)"
    r"|sin\s+term[oó]metro|no\s+tengo\s+term|ni\s+idea|no\s+sabr[ií]a",
    re.I)


def _norm(t):
    t = unicodedata.normalize("NFD", (t or "").lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def dolor(texto):
    """Devuelve entero 0-10 o None. Deterministico."""
    t = _norm(texto)
    if NEGACION.search(t):
        return None

    # Digito explicito, con o sin contexto: "9", "un 9", "9 de 10", "dolor 7"
    for m in re.finditer(r"\b(\d{1,2})\b", t):
        n = int(m.group(1))
        if 0 <= n <= 10:
            return n

    # "uno"/"una" son pronombres impersonales en el habla colombiana
    # ("uno aguanta", "uno se acostumbra"): no son cifras de dolor.
    t = re.sub(r"\b(?:uno|una)\s+(?:se\s+)?[a-z]+a\b", " ", t)

    # Numero en palabras: "nueve", "un ocho", "como siete"
    for palabra, valor in PALABRAS.items():
        if valor > 10:
            continue
        if re.search(r"\b%s\b" % re.escape(palabra), t):
            return valor
    return None


def temperatura(texto):
    """Devuelve float 34.0-43.0 o None. Deterministico."""
    t = _norm(texto)
    if NEGACION.search(t):
        return None

    # "38.5", "38,5", "38 5", "treinta y ocho cinco"
    m = re.search(r"\b(3[4-9]|4[0-3])\s*[.,]\s*(\d)\b", t)
    if m:
        return float("%s.%s" % (m.group(1), m.group(2)))

    m = re.search(r"\b(3[4-9]|4[0-3])\s+(?:con\s+|punto\s+)?(\d)\b", t)
    if m:
        return float("%s.%s" % (m.group(1), m.group(2)))

    m = re.search(r"\b(3[4-9]|4[0-3])\b", t)
    if m:
        return float(m.group(1))

    # En palabras: "treinta y nueve", opcionalmente + decimal
    for palabra in sorted(PALABRAS, key=len, reverse=True):
        v = PALABRAS[palabra]
        if not (34 <= v <= 43):
            continue
        mm = re.search(r"\b%s\b(?:\s+(?:con\s+|punto\s+)?(\w+))?" % re.escape(palabra), t)
        if mm:
            dec = PALABRAS.get(mm.group(1) or "", None)
            if dec is not None and 0 <= dec <= 9:
                return float("%d.%d" % (v, dec))
            return float(v)
    return None
