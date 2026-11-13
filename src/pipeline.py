from .data import load_weekly, Discretizer, to_states
from .bayes import build_network


def load_all(path, train_frac=0.7):
    w = load_weekly(path); cut = int(len(w) * train_frac)
    d = Discretizer().fit(w.iloc[:cut]); s = to_states(w, d)
    limite = w.index[cut - 1]  # fila t usa t+1: el entrenamiento no ve semanas de test
    train, test = s[s.index < limite], s[s.index >= limite]
    return w, d, s, build_network(train), train, test
