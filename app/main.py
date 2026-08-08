"""API del agente de seguimiento postoperatorio. Fase 0: corpus + recuperacion."""
import os, shutil, tempfile
from collections import Counter
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from .db import inicializar, conectar
from .bm25 import BM25
from .ingesta import ingerir_pdf, ingerir_corpus, cargar_indice
from .cobertura import hay_cobertura, PROCEDIMIENTO_AREA
from .esquema import validar
from .triaje import triar
from .orquestador import Llamada, PASOS
from .extractor import extraer_campo, redactar
from .lexico import expandir
from . import llamadas as LL

SESIONES = {}
UMBRAL_FUNDAMENTO = 8.0

app = FastAPI(title="Agente Postoperatorio", version="0.1")
con = inicializar()
indice = BM25()

def reindexar():
    return cargar_indice(con, indice)

@app.on_event("startup")
def arranque():
    n = reindexar()
    print(f"[arranque] fragmentos en indice: {n}")
    if n == 0:
        print("[arranque] indice vacio. POST /admin/corpus/cargar para ingerir el corpus.")

@app.get("/salud")
def salud():
    d = con.execute("SELECT COUNT(*) c FROM documentos").fetchone()["c"]
    f = con.execute("SELECT COUNT(*) c FROM fragmentos").fetchone()["c"]
    return {"estado": "ok", "documentos": d, "fragmentos": f, "indice": len(indice.docs)}

@app.post("/admin/corpus/cargar")
def cargar_corpus(ruta: str = Form("kit/dataset/textos")):
    if not os.path.isdir(ruta):
        raise HTTPException(400, f"No existe el directorio: {ruta}")
    res = ingerir_corpus(con, ruta)
    n = reindexar()
    return {"resumen": dict(Counter(r["estado"] for r in res)),
            "errores": [{"nombre": r["nombre"], "detalle": r.get("detalle", "")}
                        for r in res if r["estado"] == "error"],
            "sin_texto": [r["nombre"] for r in res if r["estado"] == "sin_texto"],
            "fragmentos_indexados": n}

@app.get("/admin/documentos")
def listar():
    filas = con.execute("""
      SELECT d.id, d.nombre, d.area, d.origen, d.paginas, d.escaneado,
             COUNT(f.id) fragmentos
      FROM documentos d LEFT JOIN fragmentos f ON f.documento_id=d.id
      GROUP BY d.id ORDER BY d.subido_en DESC, d.id DESC""").fetchall()
    return [dict(r) for r in filas]

@app.post("/admin/documentos")
async def subir(archivo: UploadFile = File(...), area: str = Form("cargado")):
    """G5: conocimiento vivo. Sube un PDF y queda disponible de inmediato."""
    if not archivo.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Solo se aceptan archivos PDF.")
    tmp = os.path.join(tempfile.gettempdir(), archivo.filename)
    with open(tmp, "wb") as f:
        shutil.copyfileobj(archivo.file, f)
    try:
        r = ingerir_pdf(con, tmp, area=area, origen="cargado")
    finally:
        os.unlink(tmp)
    r["fragmentos_indexados"] = reindexar()
    return r

@app.delete("/admin/documentos/{doc_id}")
def eliminar(doc_id: int):
    """G5: el agente olvida. Borra documento y sus fragmentos, y reindexa."""
    cur = con.execute("DELETE FROM documentos WHERE id=?", (doc_id,))
    con.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "Documento no encontrado.")
    return {"eliminado": doc_id, "fragmentos_indexados": reindexar()}

@app.get("/buscar")
def buscar(consulta: str, procedimiento: str = None, k: int = 3):
    """Recuperacion con validacion de cobertura por procedimiento."""
    if not indice.docs:
        reindexar()
    filtro = None
    aviso = None
    if procedimiento:
        ok, info = hay_cobertura(procedimiento)
        if not ok:
            return JSONResponse({"cobertura": False, "motivo": info,
                                 "resultados": [], "accion": "declarar_limite_y_escalar"})
        filtro = lambda d, a=info: d["area"] == a or d["origen"] == "cargado"
    else:
        aviso = "Sin procedimiento: la recuperacion no esta acotada clinicamente."
    res = indice.buscar(consulta, k=k, filtro=filtro)
    return {"cobertura": True, "aviso": aviso, "consulta": consulta,
            "resultados": [{"fragmento_id": r["id"], "documento": r["documento"],
                            "area": r["area"], "score": r["score"],
                            "texto": r["texto"]} for r in res]}

@app.get("/procedimientos")
def procedimientos():
    out = []
    for p in PROCEDIMIENTO_AREA:
        ok, info = hay_cobertura(p)
        out.append({"procedimiento": p, "cobertura": ok,
                    "area": info if ok else None, "motivo": None if ok else info})
    return out


# =====================================================================
#  Interfaz de llamada (G4) y ciclo conversacional
# =====================================================================

class InicioLlamada(BaseModel):
    paciente_id: str = "pac_42_00017"
    procedimiento: str = "Colecistectomía"
    dia_postop: int = 7


class TurnoPaciente(BaseModel):
    llamada_id: int
    texto: str


def _buscar_contexto(consulta, procedimiento):
    """Recuperacion con lexico coloquial y umbral de fundamento."""
    ok, info = hay_cobertura(procedimiento)
    if not ok:
        return [], False
    exp, _ = expandir(consulta)
    if not indice.docs:
        reindexar()
    res = indice.buscar(exp, k=3, filtro=lambda d, a=info: d["area"] == a or d["origen"] == "cargado")
    res = [r for r in res if r["score"] >= UMBRAL_FUNDAMENTO]
    return [{"fragmento_id": r["id"], "documento": r["documento"],
             "texto": r["texto"], "score": r["score"]} for r in res], True


@app.post("/llamada/iniciar")
def iniciar(b: InicioLlamada):
    ok, _ = hay_cobertura(b.procedimiento)
    lid = LL.abrir(con, b.paciente_id, b.procedimiento, b.dia_postop)
    L = Llamada(b.paciente_id, b.procedimiento, b.dia_postop, cobertura=ok)
    SESIONES[lid] = L

    if not ok:
        texto = ("Buenos días. Le llamo del seguimiento postoperatorio. "
                 "No tengo información clínica cargada para su procedimiento, "
                 "así que voy a comunicarlo con el personal de salud para que lo contacten.")
        L.registrar_agente(texto)
        LL.guardar_turno(con, lid, 0, "agente", texto)
        r = triar(L.snapshot(), b.dia_postop, cobertura_corpus=False, intentos_agotados=True)
        LL.guardar_decision(con, lid, r, L.snapshot())
        LL.cerrar(con, lid)
        return {"llamada_id": lid, "texto": texto, "termino": True,
                "cobertura": False, "triaje": r}

    ins = L.instruccion_siguiente()
    texto, met = redactar(
        "Saluda al paciente, identifícate como el seguimiento postoperatorio "
        "del día %d después de su %s, y luego: %s"
        % (b.dia_postop, b.procedimiento, ins), historial=None)
    if texto is None:
        raise HTTPException(503, "Ollama no responde: %s" % met.get("error"))
    L.registrar_agente(texto)
    LL.guardar_turno(con, lid, 0, "agente", texto, met)
    return {"llamada_id": lid, "texto": texto, "termino": False,
            "cobertura": True, "campo_en_curso": L.campo_actual(), "metricas": met}


@app.post("/llamada/turno")
def turno(b: TurnoPaciente):
    L = SESIONES.get(b.llamada_id)
    if L is None:
        raise HTTPException(404, "Llamada no encontrada o ya cerrada.")

    idx = len(L.historial)
    campo = L.campo_actual()

    # 1) EXTRAER solo el campo en curso
    met_ex = {}
    bruto = {}
    if campo:
        bruto_campo, met_ex = extraer_campo(campo, b.texto)
        if isinstance(bruto_campo, dict):
            bruto = bruto_campo
    # 2) VALIDAR contra el enumerado estricto
    validado, rechazos = validar(bruto)
    L.registrar_paciente(b.texto, validado, rechazos)
    tid = LL.guardar_turno(con, b.llamada_id, idx, "paciente", b.texto, met_ex)

    # 3) RECUPERAR contexto con trazabilidad
    contexto, cobertura = _buscar_contexto(b.texto, L.procedimiento)
    LL.guardar_citas(con, b.llamada_id, tid, contexto)

    # 4) TRIAR (deterministico, sin LLM)
    r = triar(L.snapshot(), L.dia_postop, cobertura_corpus=cobertura,
              intentos_agotados=L.intentos_agotados())
    LL.guardar_decision(con, b.llamada_id, r, L.snapshot())

    # 5) CIERRE CLINICO INMEDIATO
    # Si hay ROJO por un signo de alarma REAL (no por datos faltantes), seguir el
    # guion es clinicamente incorrecto: se cierra y se escala de inmediato.
    alarma_real = any(m["nivel"] == "rojo" and "ausentes" not in m["regla"]
                      and "corpus" not in m["regla"] for m in r["motivos"])

    ins = None if alarma_real else L.instruccion_siguiente()
    if alarma_real or ins is None or L.termino():
        if alarma_real:
            cierre = ("Cierra la llamada YA. Explica en una frase que lo que describe "
                      "requiere valoracion medica hoy mismo, sin alarmarlo. Indica que "
                      "un profesional de salud lo contactara de inmediato. No hagas mas preguntas.")
        elif r["alerta_humano"]:
            cierre = ("Cierra la llamada agradeciendo. Informa que un profesional de "
                      "salud lo contactara pronto.")
        else:
            cierre = ("Cierra la llamada agradeciendo. Indica que todo sigue su curso "
                      "y que llame si algo cambia.")
        texto, met = redactar(cierre, contexto=contexto, historial=L.historial)
        texto = texto or "Gracias por su tiempo. Que esté bien."
        L.registrar_agente(texto)
        LL.guardar_turno(con, b.llamada_id, len(L.historial), "agente", texto, met)
        LL.cerrar(con, b.llamada_id)
        SESIONES.pop(b.llamada_id, None)
        return {"texto": texto, "termino": True, "triaje": r,
                "cierre_por_alarma": alarma_real,
                "fuentes": contexto, "rechazos": rechazos,
                "resumen": LL.resumen(con, b.llamada_id)}

    texto, met = redactar(ins, contexto=contexto, historial=L.historial)
    if texto is None:
        raise HTTPException(503, "Ollama no responde: %s" % met.get("error"))
    L.registrar_agente(texto)
    LL.guardar_turno(con, b.llamada_id, len(L.historial), "agente", texto, met)
    return {"texto": texto, "termino": False, "triaje": r, "fuentes": contexto,
            "rechazos": rechazos, "campo_en_curso": L.campo_actual(),
            "sintomas": L.sintomas, "metricas": met}


@app.get("/llamada/{lid}/resumen")
def ver_resumen(lid: int):
    r = LL.resumen(con, lid)
    if r is None:
        raise HTTPException(404, "Llamada no encontrada.")
    return r


@app.get("/llamadas")
def listar_llamadas():
    filas = con.execute(
        "SELECT id,paciente_id,procedimiento,dia_postop,estado,criticidad,iniciada"
        " FROM llamadas ORDER BY id DESC LIMIT 50").fetchall()
    return [dict(f) for f in filas]


@app.get("/pacientes")
def pacientes():
    """Pacientes de demo. Cedula enmascarada: son datos sinteticos pero con forma de PII."""
    return [
        {"paciente_id": "pac_42_00017", "nombre": "Ramiro Antonio Guzmán",
         "cedula": "****2010", "eps": "Compensar EPS",
         "procedimiento": "Colecistectomía", "dia_postop": 7},
        {"paciente_id": "pac_42_00028", "nombre": "Luz Marina Ospina",
         "cedula": "****7331", "eps": "Sura EPS",
         "procedimiento": "Apendicectomía", "dia_postop": 3},
        {"paciente_id": "pac_42_00005", "nombre": "Gloria Esperanza Ruiz",
         "cedula": "****9789", "eps": "Salud Total EPS",
         "procedimiento": "Mastectomía", "dia_postop": 7},
    ]


@app.get("/")
def raiz():
    ruta = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if not os.path.exists(ruta):
        raise HTTPException(404, "Falta app/static/index.html")
    return FileResponse(ruta)
