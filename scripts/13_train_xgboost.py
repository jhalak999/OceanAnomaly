import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


BASE_DIR = Path(__file__).resolve().parents[1]
data_dir = BASE_DIR / "dataset" / "final"
model_dir = BASE_DIR / "models"
model_dir.mkdir(exist_ok=True)

X_train = pd.read_csv(data_dir / "X_train_ml.csv")
X_test  = pd.read_csv(data_dir / "X_test_ml.csv")
y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
y_test  = pd.read_csv(data_dir / "y_test.csv").values.ravel()

model = XGBRegressor(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nXGBoost Performance")
print(f"RMSE: {rmse:.3f}")
print(f"MAE : {mae:.3f}")
print(f"R²  : {r2:.3f}")


import joblib
from pathlib import Path

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

joblib.dump(model, model_dir / "xgboost_model.pkl")

print("XGBoost model saved to models/xgboost_model.pkl")
