import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("era5_features_clean.csv")

features = ["wind_speed", "t2m_c", "sshf", "net_radiation", "e"]

plt.figure(figsize=(12, 8))

for f in features:
    plt.plot(df["valid_time"], df[f], label=f)

plt.legend()
plt.title("ERA5 Climate Drivers – Monthly Trends (1981–2024)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
