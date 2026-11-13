"""Genera datos SINTETICOS (solo para probar el pipeline). Reemplazar por datos reales IDEAM."""
import numpy as np, pandas as pd
rng = np.random.default_rng(7)
dates = pd.date_range("2022-01-01", "2025-12-31", freq="D"); n = len(dates)
season = np.sin(2 * np.pi * 2 * dates.dayofyear / 365 - 1)  # regimen bimodal
wet = np.zeros(n, bool); p = 0.35
for i in range(n):
    pw = np.clip(0.35 + 0.25 * season[i] + (0.25 if i and wet[i-1] else -0.1), 0.05, 0.9)
    wet[i] = rng.random() < pw
rain = np.where(wet, rng.exponential(9, n), 0)
hum = np.clip(72 + 12 * wet + 4 * season + rng.normal(0, 3, n), 50, 98)
temp = 21.5 - 0.8 * wet + rng.normal(0, 1, n)
pd.DataFrame({"fecha": dates, "precipitacion": rain.round(1), "temperatura": temp.round(1), "humedad": hum.round(1)}
             ).to_csv("data/raw/demo_station.csv", index=False)
print("data/raw/demo_station.csv listo (SINTETICO)")
