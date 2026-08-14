import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
data_dir = BASE_DIR / "dataset" / "final"
model_dir = BASE_DIR / "models"


X_train = pd.read_csv(data_dir / "X_train_ml.csv")


model = joblib.load(model_dir / "xgboost_model.pkl")


importance = model.feature_importances_
features = X_train.columns


assert len(features) == len(importance), "Feature length mismatch!"

imp_df = pd.DataFrame({
    "feature": features,
    "importance": importance
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(8, 5))
plt.barh(imp_df["feature"], imp_df["importance"])
plt.gca().invert_yaxis()
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
