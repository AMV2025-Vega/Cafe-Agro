"""Formato CSV: separador ';', columnas fecha, valor. Motor Python robusto."""
import numpy as np, pandas as pd

TEMP_L = ["fresca", "media", "calida"]
TREND_L = ["baja", "estable", "sube"]
SEASON_L = ["ene-mar", "abr-jun", "jul-sep", "oct-dic"]


def load_weekly(path):
    chunks = []
    """Usamos engine='python' y chunksize para procesar de forma segura sin saturar la RAM"""
    for chunk in pd.read_csv(
        path, 
        sep=";", 
        usecols=["fecha", "valor"], 
        chunksize=50000, 
        engine='python', 
        encoding="latin-1", 
        on_bad_lines='skip'
    ):
        chunk["fecha"] = pd.to_datetime(chunk["fecha"], format='mixed', errors="coerce")
        
        """Limpieza de valores numéricos con posibles separadores extraños"""
        chunk["valor"] = (
            chunk["valor"]
            .astype(str)
            .str.replace('.', '', regex=False)
            .str.replace(',', '.', regex=False)
        )
        chunk["valor"] = pd.to_numeric(chunk["valor"], errors="coerce")
        chunk = chunk.dropna()
        chunk = chunk[chunk["valor"].between(0, 45)]  # descarta valores imposibles
        chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)
    
    w = pd.DataFrame({"temp": df.set_index("fecha")["valor"].resample("W").mean()})
    w = w.interpolate(limit=2).dropna()
    
    if len(w) < 40:
        raise ValueError(f"Solo {len(w)} semanas utiles; se necesita mas historia.")
    return w


class Discretizer:
    """Umbrales ajustados SOLO con entrenamiento (sin fuga de datos)."""
    def fit(self, w):
        self.tq = np.quantile(w["temp"], [1/3, 2/3])
        self.ts = 0.5 * float(w["temp"].diff().dropna().std())
        return self

    def temp(self, x): return TEMP_L[int(np.digitize(x, self.tq))]
    def trend(self, dx): return TREND_L[0 if dx < -self.ts else (2 if dx > self.ts else 1)]
    @staticmethod
    def season(ts): return SEASON_L[(ts.month - 1) // 3]


def to_states(w, d):
    diff = w["temp"].diff()
    s = pd.DataFrame({"temp": w["temp"].map(d.temp),
                      "trend": diff.map(lambda x: np.nan if pd.isna(x) else d.trend(x)),
                      "season": [d.season(t) for t in w.index]}, index=w.index)
    s["temp_next"] = s["temp"].shift(-1)
    return s.dropna()