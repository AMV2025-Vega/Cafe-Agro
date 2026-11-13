"""Red bayesiana con eliminacion de variables (implementada desde cero).
TempNext | Temp, Trend, Season : aprendida con conteos + Laplace.
Roya | TempNext : CPT EXPERTA (supuesto a validar; no hay etiquetas reales ni lluvia/humedad)."""
from itertools import product
from .data import TEMP_L, TREND_L, SEASON_L

ROYA_L = ["bajo", "medio", "alto"]
DOM = {"temp": TEMP_L, "trend": TREND_L, "season": SEASON_L, "temp_next": TEMP_L, "roya": ROYA_L}


class Factor:
    def __init__(self, vars, table): self.vars, self.table = list(vars), dict(table)


def multiply(f, g):
    vs = f.vars + [v for v in g.vars if v not in f.vars]; t = {}
    for vals in product(*[DOM[v] for v in vs]):
        a = dict(zip(vs, vals))
        t[vals] = f.table[tuple(a[v] for v in f.vars)] * g.table[tuple(a[v] for v in g.vars)]
    return Factor(vs, t)


def sum_out(f, var):
    i = f.vars.index(var); t = {}
    for k, p in f.table.items():
        kk = k[:i] + k[i+1:]; t[kk] = t.get(kk, 0.0) + p
    return Factor(f.vars[:i] + f.vars[i+1:], t)


def restrict(f, var, val):
    if var not in f.vars: return f
    i = f.vars.index(var)
    return Factor(f.vars[:i] + f.vars[i+1:], {k[:i] + k[i+1:]: p for k, p in f.table.items() if k[i] == val})


class BayesNet:
    def __init__(self): self.nodes = {}

    def add(self, name, parents, cpt): self.nodes[name] = (parents, cpt)

    def _factor(self, name):
        parents, cpt = self.nodes[name]; t = {}
        for pv, dist in cpt.items():
            for v, p in dist.items(): t[pv + (v,)] = p
        return Factor(parents + [name], t)

    def query(self, var, evidence):
        factors = [self._factor(n) for n in self.nodes]
        for k, v in evidence.items(): factors = [restrict(f, k, v) for f in factors]
        for h in [n for n in self.nodes if n != var and n not in evidence]:
            rel = [f for f in factors if h in f.vars]; rest = [f for f in factors if h not in f.vars]
            if not rel: continue
            prod = rel[0]
            for f in rel[1:]: prod = multiply(prod, f)
            factors = rest + [sum_out(prod, h)]
        res = factors[0]
        for f in factors[1:]: res = multiply(res, f)
        z = sum(res.table.values())
        return {k[0]: float(p / z) for k, p in res.table.items()}

    def predict(self, var, evidence):
        d = self.query(var, evidence); return max(d, key=d.get)


ROYA_EXPERTA = {"fresca": [.60, .30, .10], "media": [.20, .40, .40], "calida": [.40, .40, .20]}


def build_network(states, alpha=1.0):
    bn = BayesNet(); n = len(states)
    for v in ["temp", "trend", "season"]:
        c = states[v].value_counts()
        bn.add(v, [], {(): {x: (c.get(x, 0) + alpha) / (n + alpha * len(DOM[v])) for x in DOM[v]}})
    cpt = {}
    for pv in product(TEMP_L, TREND_L, SEASON_L):
        sub = states[(states.temp == pv[0]) & (states.trend == pv[1]) & (states.season == pv[2])]
        c = sub["temp_next"].value_counts()
        cpt[pv] = {x: (c.get(x, 0) + alpha) / (len(sub) + alpha * 3) for x in TEMP_L}
    bn.add("temp_next", ["temp", "trend", "season"], cpt)
    bn.add("roya", ["temp_next"], {(t,): dict(zip(ROYA_L, p)) for t, p in ROYA_EXPERTA.items()})
    return bn
