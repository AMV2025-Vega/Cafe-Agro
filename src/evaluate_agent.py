"""Evalua el agente con preguntas fijas: herramienta correcta + cita de fuente."""
import json, os, re, sys, csv
from .pipeline import load_all
from .rag import Retriever
from .agent import Toolbox, CoffeeAgent

path = os.getenv("DATA_PATH", "data/raw/ideam_station.csv")
w, d, s, bn, train, test = load_all(path)
agent = CoffeeAgent(Toolbox(w, d, bn, Retriever("corpus")))
qs = json.load(open("eval/preguntas.json", encoding="utf-8")); rows = []
for q in qs:
    ans, trace = agent.run([], q["pregunta"]); used = [t["tool"] for t in trace]
    if ans.startswith('Error al llamar'): print(ans)
    cited = bool(re.search(r"fuente:", ans, re.I))
    rows.append({"pregunta": q["pregunta"], "esperada": q["herramienta"], "usadas": "|".join(used),
                 "herramienta_ok": q["herramienta"] in used, "cito_fuente": cited, "respuesta": ans.replace("\n", " ")[:300]})
os.makedirs("results", exist_ok=True)
with open("results/evaluacion_agente.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
print(f"Herramienta correcta: {sum(r['herramienta_ok'] for r in rows)}/{len(rows)}")
print(f"Cito fuente (solo relevante en search_docs): {sum(r['cito_fuente'] for r in rows if r['esperada']=='search_docs')}/{sum(r['esperada']=='search_docs' for r in rows)}")
