import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "dataset" / "final"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results" / "shap"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


model = joblib.load(MODEL_DIR / "xgboost_model.pkl")
print("Loaded XGBoost model")


X_train = pd.read_csv(DATA_DIR / "X_train.csv")


for col in ["valid_time", "oni"]:
    if col in X_train.columns:
        X_train = X_train.drop(columns=[col])


expected_features = model.get_booster().feature_names
X_train = X_train[expected_features]

print("Aligned SHAP input features")
print("X_train shape:", X_train.shape)


explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)


plt.figure()
shap.summary_plot(shap_values, X_train, show=False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "shap_summary.png", dpi=300)
plt.close()

print("SHAP global summary saved")


idx = 80 

plt.figure()
shap.plots.waterfall(shap_values[idx], show=False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "shap_force_example.png", dpi=300)
plt.close()

print("SHAP local explanation saved")
