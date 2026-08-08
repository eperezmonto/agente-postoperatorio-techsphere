"""Esquema SQLite. Cero instalacion, archivo unico."""
import sqlite3, os

RUTA = os.environ.get("DB_PATH", "datos.db")

ESQUEMA = """
CREATE TABLE IF NOT EXISTS documentos (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  nombre TEXT NOT NULL, area TEXT NOT NULL,
  origen TEXT NOT NULL DEFAULT 'corpus',
  paginas INTEGER, escaneado INTEGER DEFAULT 0,
  sha TEXT UNIQUE, subido_en TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS fragmentos (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  documento_id INTEGER NOT NULL REFERENCES documentos(id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL, texto TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_frag_doc ON fragmentos(documento_id);
CREATE TABLE IF NOT EXISTS llamadas (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  paciente_id TEXT, procedimiento TEXT, dia_postop INTEGER,
  estado TEXT DEFAULT 'en_curso', criticidad TEXT,
  iniciada TEXT DEFAULT (datetime('now')), cerrada TEXT
);
CREATE TABLE IF NOT EXISTS turnos (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  llamada_id INTEGER NOT NULL REFERENCES llamadas(id) ON DELETE CASCADE,
  idx INTEGER, hablante TEXT, texto TEXT,
  latencia_ms INTEGER, tokens_in INTEGER, tokens_out INTEGER,
  creado TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS decisiones (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  llamada_id INTEGER NOT NULL REFERENCES llamadas(id) ON DELETE CASCADE,
  criticidad TEXT NOT NULL, motivo TEXT NOT NULL,
  sintomas_json TEXT, regla TEXT, creado TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS citas (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  llamada_id INTEGER REFERENCES llamadas(id) ON DELETE CASCADE,
  turno_id INTEGER, fragmento_id INTEGER, documento_nombre TEXT, score REAL
);
"""

def conectar():
    c = sqlite3.connect(RUTA, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    return c

def inicializar():
    c = conectar(); c.executescript(ESQUEMA); c.commit(); return c
