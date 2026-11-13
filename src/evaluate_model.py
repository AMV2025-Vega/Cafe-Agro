"""Metrica Parte 1: temperatura (tercil) de la semana siguiente; validacion temporal 70/30."""
import json, os, sys
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from .pipeline import load_all

path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATA_PATH", "data/raw/ideam_station.csv")
w, d, s, bn, train, test = load_all(path)
y = test["temp_next"].tolist(); major = Counter(train["temp_next"]).most_common(1)[0][0]
preds = {"red_bayesiana": [bn.predict("temp_next", dict(temp=r.temp, trend=r.trend, season=r.season)) for r in test.itertuples()],
         "base_persistencia": test["temp"].tolist(), "base_mayoritaria": [major] * len(y)}
res = {k: {"accuracy": round(accuracy_score(y, p), 3), "f1_macro": round(f1_score(y, p, average="macro"), 3)} for k, p in preds.items()}
res["n_train"], res["n_test"], res["datos"] = len(train), len(y), path
os.makedirs("results", exist_ok=True); json.dump(res, open("results/metricas_modelo.json", "w"), indent=2)
print(json.dumps(res, indent=2))
