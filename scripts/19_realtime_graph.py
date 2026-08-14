import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "dataset" / "final"
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.pkl"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


model = joblib.load(MODEL_PATH)
X_test = pd.read_csv(DATA_DIR / "X_test.csv")


if "valid_time" in X_test.columns:
    X_test = X_test.drop(columns=["valid_time"])
if "oni" in X_test.columns:
    X_test = X_test.drop(columns=["oni"])

times = []

for i in range(100):   
    start = time.time()
    model.predict(X_test.iloc[:1])  
    end = time.time()
    times.append(end - start)

avg_time = sum(times) / len(times)

print(f"\nAverage prediction time per sample: {avg_time:.5f} sec")


plt.figure(figsize=(6,4))
plt.plot(times)
plt.title("Real-Time Prediction Latency per Sample")
plt.xlabel("Run Number")
plt.ylabel("Time (seconds)")
plt.grid(True)

save_path = RESULTS_DIR / "realtime_prediction_speed.png"
plt.savefig(save_path)
plt.show()

print("\nGraph saved at:", save_path)