"""Extraccion y segmentacion del corpus clinico."""
import os, re, unicodedata
from pypdf import PdfReader

MIN_CHUNK, MAX_CHUNK = 400, 1100

def normalizar(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).replace("\x00", "")
    t = re.sub(r"-\n(?=[a-záéíóúñ])", "", t)
    t = re.sub(r"\n{2,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()

def segmentar(texto: str):
    """Corta en limites de parrafo/oracion, nunca a mitad de palabra."""
    parrafos = [p.strip() for p in re.split(r"\n\s*\n", texto) if len(p.strip()) > 40]
    chunks, buf = [], ""
    for p in parrafos:
        p = re.sub(r"\s*\n\s*", " ", p)
        if len(buf) + len(p) + 1 <= MAX_CHUNK:
            buf = (buf + " " + p).strip()
            continue
        if buf:
            chunks.append(buf); buf = ""
        if len(p) <= MAX_CHUNK:
            buf = p; continue
        oraciones = re.split(r"(?<=[.;:])\s+(?=[A-ZÁÉÍÓÚÑ0-9])", p)
        for o in oraciones:
            if len(buf) + len(o) + 1 <= MAX_CHUNK:
                buf = (buf + " " + o).strip()
            else:
                if len(buf) >= MIN_CHUNK: chunks.append(buf)
                buf = o[:MAX_CHUNK]
    if len(buf) >= MIN_CHUNK: chunks.append(buf)
    return [c for c in chunks if len(c) >= MIN_CHUNK]

def leer_pdf(ruta: str):
    """Devuelve (texto, n_paginas, escaneado)."""
    r = PdfReader(ruta)
    txt = normalizar("\n\n".join((p.extract_text() or "") for p in r.pages))
    escaneado = len(txt) < 200 * max(len(r.pages), 1) and len(txt) < 1000
    return txt, len(r.pages), escaneado
