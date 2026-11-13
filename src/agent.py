"""Agente LLM: loop manual con tool calling (API compatible con OpenAI), RAG y T_max."""
import json, os
from openai import OpenAI
from .csp import solve
from .data import TEMP_L, TREND_L, SEASON_L

MAX_ITERS = 6
SYSTEM = """Eres un asistente agroclimatico para caficultores colombianos. Usa SIEMPRE las herramientas:
- get_risk para riesgo de roya y temperatura esperada, - schedule para planear labores, - search_docs para preguntas tecnicas.
Limitacion: el modelo solo usa TEMPERATURA de una estacion; la lluvia en schedule es un supuesto del usuario. Dilo si es relevante.
Cita las fuentes de search_docs asi: [fuente: archivo, p. N]. Si no hay documentos relevantes, dilo; no inventes.
Responde en espanol, breve. Eres una herramienta analitica, NO un consejo agronomico profesional: recomienda
consultar al extensionista del Comite de Cafeteros."""

TOOLS = [
    {"type": "function", "function": {"name": "get_risk", "description": "Probabilidad de roya (bajo/medio/alto) y de la temperatura de la proxima semana. Sin argumentos usa la ultima semana observada.",
     "parameters": {"type": "object", "properties": {"temp": {"type": "string", "enum": TEMP_L}, "trend": {"type": "string", "enum": TREND_L}, "season": {"type": "string", "enum": SEASON_L}}}}},
    {"type": "function", "function": {"name": "schedule", "description": "Calendario semanal de labores (CSP). rain_class es un SUPUESTO de lluvia (no hay datos); capacity = jornaleros por dia.",
     "parameters": {"type": "object", "properties": {"rain_class": {"type": "string", "enum": ["baja", "media", "alta"]}, "capacity": {"type": "integer", "minimum": 1, "maximum": 20}}}}},
    {"type": "function", "function": {"name": "search_docs", "description": "Busca en documentos tecnicos de Cenicafe/Federacion.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


class Toolbox:
    def __init__(self, weekly, disc, bn, retriever, capacity=5):
        self.w, self.d, self.bn, self.r, self.capacity = weekly, disc, bn, retriever, capacity

    def _last(self):
        t = self.w["temp"]
        return dict(temp=self.d.temp(t.iloc[-1]), trend=self.d.trend(t.iloc[-1] - t.iloc[-2]), season=self.d.season(self.w.index[-1]))

    def get_risk(self, temp=None, trend=None, season=None):
        ev = {**self._last(), **{k: v for k, v in dict(temp=temp, trend=trend, season=season).items() if v}}
        return {"evidencia": ev, "P_roya": self.bn.query("roya", ev), "P_temp_proxima_semana": self.bn.query("temp_next", ev),
                "nota": "Riesgo de roya basado en CPT experta solo de temperatura (supuesto)."}

    def schedule(self, rain_class="media", capacity=None):
        alto = bool(self.bn.query("roya", self._last()).get("alto", 0) > 0.4)
        return {"lluvia_asumida_por_usuario": rain_class, "roya_alto": alto, **solve(rain_class, capacity or self.capacity, alto)}

    def search_docs(self, query):
        return self.r.search(query) or {"error": "Sin documentos relevantes en el corpus."}


class CoffeeAgent:
    def __init__(self, toolbox):
        self.tb = toolbox
        self.client = OpenAI(base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
                             api_key=os.getenv("LLM_API_KEY", "missing"))
        self.model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
        self._picked = False

    def _pick_model(self):
        """Si el modelo no existe, elige uno disponible en la cuenta (los modelos de Groq cambian)."""
        import re
        ids = [m.id for m in self.client.models.list().data]
        pref = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "meta-llama/llama-4-maverick-17b-128e-instruct",
                "meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.3-70b-versatile"]
        ok = [i for i in pref if i in ids] or [i for i in ids if not re.search(r"whisper|guard|tts|orpheus|compound|safeguard|embed", i)]
        if not ok: raise RuntimeError(f"Sin modelos de chat disponibles: {ids}")
        self.model = ok[0]; self._picked = True; print(f"[agent] usando modelo {self.model}")

    def run(self, history, user_msg):
        msgs = [{"role": "system", "content": SYSTEM}] + history + [{"role": "user", "content": user_msg}]
        trace = []
        for _ in range(MAX_ITERS):
            try:
                m = self.client.chat.completions.create(model=self.model, messages=msgs, tools=TOOLS, temperature=0.2).choices[0].message
            except Exception as e:
                if not self._picked and ("404" in str(e) or "does not exist" in str(e)):
                    try: self._pick_model(); continue
                    except Exception as e2: e = e2
                return f"Error al llamar al LLM: {e}", trace
            if not m.tool_calls: return m.content or "", trace
            msgs.append(m.model_dump(exclude_none=True))
            for tc in m.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    out = getattr(self.tb, tc.function.name)(**args)
                except Exception as e:  # el error vuelve como observacion
                    args, out = {}, {"error": f"{type(e).__name__}: {e}"}
                trace.append({"tool": tc.function.name, "args": args})
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out, ensure_ascii=False, default=str)})
        return "No logre responder dentro del limite de pasos.", trace
