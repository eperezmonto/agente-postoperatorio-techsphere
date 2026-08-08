"""Ingesta del corpus. Sirve para el corpus inicial y para carga en caliente."""
import hashlib, os, glob
from .chunking import leer_pdf, segmentar

def sha_archivo(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(65536), b""): h.update(b)
    return h.hexdigest()

def ingerir_pdf(con, ruta, area=None, origen="corpus"):
    """Idempotente por sha. Devuelve dict con el resultado."""
    nombre = os.path.basename(ruta)
    area = area or os.path.basename(os.path.dirname(ruta))
    sha = sha_archivo(ruta)
    row = con.execute("SELECT id FROM documentos WHERE sha=?", (sha,)).fetchone()
    if row:
        return {"estado": "duplicado", "documento_id": row["id"], "nombre": nombre, "fragmentos": 0}
    try:
        texto, paginas, escaneado = leer_pdf(ruta)
    except Exception as e:
        return {"estado": "error", "nombre": nombre, "detalle": str(e)[:200]}
    frags = segmentar(texto)
    if escaneado or not frags:
        cur = con.execute(
            "INSERT INTO documentos(nombre,area,origen,paginas,escaneado,sha) VALUES(?,?,?,?,1,?)",
            (nombre, area, origen, paginas, sha))
        con.commit()
        return {"estado": "sin_texto", "documento_id": cur.lastrowid, "nombre": nombre,
                "fragmentos": 0,
                "aviso": "PDF sin capa de texto extraible. NO indexado. Requiere OCR."}
    cur = con.execute(
        "INSERT INTO documentos(nombre,area,origen,paginas,escaneado,sha) VALUES(?,?,?,?,0,?)",
        (nombre, area, origen, paginas, sha))
    doc_id = cur.lastrowid
    con.executemany("INSERT INTO fragmentos(documento_id,ordinal,texto) VALUES(?,?,?)",
                    [(doc_id, i, t) for i, t in enumerate(frags)])
    con.commit()
    return {"estado": "indexado", "documento_id": doc_id, "nombre": nombre,
            "area": area, "paginas": paginas, "fragmentos": len(frags)}

def ingerir_corpus(con, base):
    return [ingerir_pdf(con, r) for r in
            sorted(glob.glob(os.path.join(base, "**", "*.pdf"), recursive=True))]

def cargar_indice(con, bm25):
    filas = con.execute("""
        SELECT f.id, f.texto, f.ordinal, d.nombre, d.area, d.id AS doc_id, d.origen
        FROM fragmentos f JOIN documentos d ON d.id=f.documento_id""").fetchall()
    bm25.indexar([{"id": r["id"], "texto": r["texto"], "documento": r["nombre"],
                   "area": r["area"], "documento_id": r["doc_id"],
                   "ordinal": r["ordinal"], "origen": r["origen"]} for r in filas])
    return len(filas)
