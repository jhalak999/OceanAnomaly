import pandas as pd
import matplotlib.pyplot as plt

# loading scaled train set
df = pd.read_csv("train_features_scaled.csv", parse_dates=["valid_time"])
df = df.set_index("valid_time")

feature_cols = [
    "wind_speed",
    "t2m_c",
    "sshf",
    "net_radiation",
    "e"
]

plt.figure(figsize=(14, 6))

for col in feature_cols:
    plt.plot(df.index, df[col], label=col, linewidth=1)

plt.title("Standardized ERA5 Climate Features (Training Set)")
plt.xlabel("Time")
plt.ylabel("Standardized Value (z-score)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
