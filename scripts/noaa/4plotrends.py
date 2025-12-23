import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path("noaa/outputs")

df = pd.read_csv(out_dir / "noaa_oni_features_clean.csv")
df["time"] = pd.to_datetime(df["time"])

plt.figure(figsize=(10,4))
plt.plot(df["time"], df["oni"], label="ONI")
plt.axhline(0.5, linestyle="--", color="red", label="El Niño threshold")
plt.axhline(-0.5, linestyle="--", color="blue", label="La Niña threshold")
plt.legend()
plt.title("NOAA Oceanic Niño Index (ONI)")
plt.show()
