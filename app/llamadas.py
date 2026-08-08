"""Capa de persistencia y ciclo de vida de llamadas."""
import json


def abrir(con, paciente_id, procedimiento, dia_postop):
    cur = con.execute(
        "INSERT INTO llamadas(paciente_id,procedimiento,dia_postop) VALUES(?,?,?)",
        (paciente_id, procedimiento, dia_postop))
    con.commit()
    return cur.lastrowid


def guardar_turno(con, llamada_id, idx, hablante, texto, met=None):
    met = met or {}
    cur = con.execute(
        "INSERT INTO turnos(llamada_id,idx,hablante,texto,latencia_ms,tokens_in,tokens_out)"
        " VALUES(?,?,?,?,?,?,?)",
        (llamada_id, idx, hablante, texto, met.get("latencia_ms"),
         met.get("tokens_in"), met.get("tokens_out")))
    con.commit()
    return cur.lastrowid


def guardar_decision(con, llamada_id, r, sintomas):
    reglas = " | ".join("[%s] %s" % (m["nivel"], m["regla"]) for m in r["motivos"])
    con.execute(
        "INSERT INTO decisiones(llamada_id,criticidad,motivo,sintomas_json,regla)"
        " VALUES(?,?,?,?,?)",
        (llamada_id, r["criticidad"], r.get("accion", ""),
         json.dumps(sintomas, ensure_ascii=False), reglas))
    con.execute("UPDATE llamadas SET criticidad=? WHERE id=?", (r["criticidad"], llamada_id))
    con.commit()


def guardar_citas(con, llamada_id, turno_id, resultados):
    if not resultados:
        return
    con.executemany(
        "INSERT INTO citas(llamada_id,turno_id,fragmento_id,documento_nombre,score)"
        " VALUES(?,?,?,?,?)",
        [(llamada_id, turno_id, r["fragmento_id"], r["documento"], r["score"])
         for r in resultados])
    con.commit()


def cerrar(con, llamada_id):
    con.execute("UPDATE llamadas SET estado='cerrada', cerrada=datetime('now') WHERE id=?",
                (llamada_id,))
    con.commit()


def resumen(con, llamada_id):
    """Resumen estructurado de la llamada, con trazabilidad completa."""
    ll = con.execute("SELECT * FROM llamadas WHERE id=?", (llamada_id,)).fetchone()
    if not ll:
        return None
    turnos = con.execute(
        "SELECT idx,hablante,texto,latencia_ms,tokens_in,tokens_out FROM turnos"
        " WHERE llamada_id=? ORDER BY idx", (llamada_id,)).fetchall()
    dec = con.execute(
        "SELECT criticidad,motivo,sintomas_json,regla,creado FROM decisiones"
        " WHERE llamada_id=? ORDER BY id DESC LIMIT 1", (llamada_id,)).fetchone()
    citas = con.execute(
        "SELECT DISTINCT documento_nombre, MAX(score) s FROM citas"
        " WHERE llamada_id=? GROUP BY documento_nombre ORDER BY s DESC",
        (llamada_id,)).fetchall()

    lat = [t["latencia_ms"] for t in turnos if t["latencia_ms"]]
    lat_ord = sorted(lat)
    def pctl(p):
        return lat_ord[min(int(len(lat_ord) * p), len(lat_ord) - 1)] if lat_ord else 0

    return {
        "llamada_id": llamada_id,
        "paciente_id": ll["paciente_id"],
        "procedimiento": ll["procedimiento"],
        "dia_postop": ll["dia_postop"],
        "estado": ll["estado"],
        "criticidad": dec["criticidad"] if dec else None,
        "accion": dec["motivo"] if dec else None,
        "sintomas": json.loads(dec["sintomas_json"]) if dec else {},
        "reglas_disparadas": (dec["regla"] or "").split(" | ") if dec and dec["regla"] else [],
        "turnos": [dict(t) for t in turnos],
        "fuentes_citadas": [{"documento": c["documento_nombre"], "score": c["s"]} for c in citas],
        "metricas": {
            "turnos_totales": len(turnos),
            "invocaciones_llm": len(lat),
            "latencia_llm_p50_ms": pctl(0.50),
            "latencia_llm_p95_ms": pctl(0.95),
            "tokens_entrada": sum(t["tokens_in"] or 0 for t in turnos),
            "tokens_salida": sum(t["tokens_out"] or 0 for t in turnos),
        },
    }
