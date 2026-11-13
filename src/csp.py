"""CSP de labores semanales: backtracking + MRV + comprobacion de restricciones."""
import random

DAYS = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]
TASKS = {"inspeccion_lotes": dict(w=1, dry=False), "control_roya": dict(w=2, dry=True),
         "fertilizacion": dict(w=3, dry=True), "poda": dict(w=2, dry=True), "cosecha": dict(w=4, dry=True)}


def wet_days(rain_class, seed=0):
    n = {"baja": 1, "media": 3, "alta": 5}[rain_class]
    return set(random.Random(seed).sample(range(7), n))


def solve(rain_class="media", capacity=5, roya_alto=False, tasks=None, seed=0):
    tasks = {k: TASKS[k] for k in (tasks or TASKS)}; wet = wet_days(rain_class, seed)
    dom = {}
    for t, a in tasks.items():
        ds = [d for d in range(7) if not (a["dry"] and d in wet) and a["w"] <= capacity]
        if roya_alto and t == "control_roya": ds = [d for d in ds if d <= 2]  # urgente
        dom[t] = ds
    stats = {"nodes": 0}

    def ok(t, d, asg):
        same = [x for x, dd in asg.items() if dd == d]
        if sum(tasks[x]["w"] for x in same) + tasks[t]["w"] > capacity: return False
        if {t, *same} >= {"fertilizacion", "cosecha"}: return False
        if t == "fertilizacion" and "control_roya" in asg and asg["control_roya"] > d: return False
        if t == "control_roya" and "fertilizacion" in asg and asg["fertilizacion"] < d: return False
        return True

    def bt(asg):
        if len(asg) == len(tasks): return asg
        t = min((x for x in tasks if x not in asg), key=lambda x: sum(ok(x, d, asg) for d in dom[x]))  # MRV
        for d in dom[t]:
            stats["nodes"] += 1
            if ok(t, d, asg):
                r = bt({**asg, t: d})
                if r: return r
        return None

    empty = [t for t in dom if not dom[t]]
    sol = None if empty else bt({})
    return {"feasible": sol is not None, "wet_days": [DAYS[d] for d in sorted(wet)], "nodes": stats["nodes"],
            "plan": {t: DAYS[d] for t, d in sorted((sol or {}).items(), key=lambda x: x[1])},
            "note": None if sol else "Sin plan factible" + (f"; sin dias posibles para: {empty}" if empty else "") + ". Sube jornaleros o reduce labores."}
