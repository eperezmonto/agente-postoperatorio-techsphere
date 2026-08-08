"""Orquestador de la llamada. Guion adaptativo con repregunta en vivo.

Hallazgo que motiva el diseno: la extraccion posterior a la llamada NO puede
recuperar lo que la conversacion nunca capturo. Medido: la paciente del caso
rojo con dolor 9/10 dice "un poquito molesto"; ningun modelo saca un 9 de ahi.
La reparacion no es mejor extraccion, es PREGUNTAR durante la llamada.

Maximo 2 reintentos por campo. Si se agotan, se escala por dato faltante.
"""

MAX_REINTENTOS = 2
MAX_TURNOS = 14        # tope duro: la llamada nunca se atasca

# Guion base: un campo critico por paso.
PASOS = [
    ("dolor_nrs", "Pregunta por el nivel de dolor en escala de 0 a 10."),
    ("fiebre_c",  "Pregunta si ha tenido fiebre y cuanto marco el termometro."),
    ("herida",    "Pregunta como se ve la herida: enrojecimiento, secrecion, o normal."),
    ("movilidad", "Pregunta si puede moverse y caminar como esperaba."),
]

# Repreguntas que fuerzan un dato concreto ante respuestas evasivas.
REPREGUNTA = {
    "dolor_nrs": "El paciente no dio un numero. Insiste con amabilidad pero con firmeza: "
                 "si 10 es el peor dolor que ha sentido en su vida y 0 es ninguno, "
                 "pide que diga el numero de hoy. No aceptes 'poquito' como respuesta.",
    "fiebre_c":  "El paciente no dio una cifra de temperatura. Pregunta si tiene termometro "
                 "y pide que se tome la temperatura ahora, o que diga si sintio escalofrio.",
    "herida":    "El paciente no describio la herida con claridad. Pide que la mire ahora "
                 "y diga si hay enrojecimiento, si sale liquido y de que color.",
    "movilidad": "El paciente no fue claro sobre su movilidad. Pregunta si logra levantarse "
                 "y caminar hasta el bano sin ayuda.",
}


class Llamada:
    """Maquina de estados de una llamada. No persiste: eso lo hace la capa API."""

    def __init__(self, paciente, procedimiento, dia_postop, cobertura=True):
        self.paciente = paciente
        self.procedimiento = procedimiento
        self.dia_postop = dia_postop
        self.cobertura = cobertura
        self.historial = []          # [{"hablante","texto"}]
        self.sintomas = {c: None for c, _ in PASOS}
        self.intentos = {c: 0 for c, _ in PASOS}
        self.paso = 0
        self.cerrada = False
        self.rechazos = []           # auditoria de valores invalidos del LLM

    # --- estado ---
    def campo_actual(self):
        return PASOS[self.paso][0] if self.paso < len(PASOS) else None

    def faltantes(self):
        return [c for c, _ in PASOS if self.sintomas.get(c) is None]

    def intentos_agotados(self):
        return all(self.intentos[c] >= MAX_REINTENTOS for c in self.faltantes()) \
            if self.faltantes() else True

    def snapshot(self):
        s = dict(self.sintomas)
        s["faltantes"] = self.faltantes()
        return s

    # --- flujo ---
    def instruccion_siguiente(self):
        """Que debe decir el agente ahora. None si la llamada termino."""
        if len(self.historial) >= MAX_TURNOS:
            return None                    # tope duro de la conversacion
        for _ in range(len(PASOS) + 1):    # sin recursion: no puede colgarse
            campo = self.campo_actual()
            if campo is None:
                return None
            if self.sintomas.get(campo) is not None or self.intentos[campo] > MAX_REINTENTOS:
                self.paso += 1
                continue
            return dict(PASOS)[campo] if self.intentos[campo] == 0 else REPREGUNTA[campo]
        return None

    def registrar_agente(self, texto):
        self.historial.append({"hablante": "agente", "texto": texto})
        campo = self.campo_actual()
        if campo is not None:
            self.intentos[campo] += 1

    def registrar_paciente(self, texto, sintomas_validados, rechazos=None):
        """Integra lo extraido. Nunca sobrescribe un dato ya obtenido."""
        self.historial.append({"hablante": "paciente", "texto": texto})
        if rechazos:
            self.rechazos.extend(rechazos)
        for c, _ in PASOS:
            if self.sintomas.get(c) is None and sintomas_validados.get(c) is not None:
                self.sintomas[c] = sintomas_validados[c]
        campo = self.campo_actual()
        if campo is not None and (self.sintomas.get(campo) is not None
                                  or self.intentos[campo] > MAX_REINTENTOS):
            self.paso += 1

    def termino(self):
        return self.paso >= len(PASOS) or len(self.historial) >= MAX_TURNOS
