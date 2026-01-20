import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parents[1]
final_dir = BASE_DIR / "dataset/final"

# Load data
X_train = pd.read_csv(final_dir / "X_train.csv")
X_test  = pd.read_csv(final_dir / "X_test.csv")
y_train = pd.read_csv(final_dir / "y_train.csv").values.ravel()
y_test  = pd.read_csv(final_dir / "y_test.csv").values.ravel()

# 🔥 KEEP ONLY NUMERIC FEATURES
X_train = X_train.select_dtypes(include=["number"])
X_test  = X_test.select_dtypes(include=["number"])

y_train = y_train.astype(float)
y_test = y_test.astype(float)


# Train simple model again (baseline)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
residuals = y_test - y_pred

# --------------------
# Residual plots
# --------------------
plt.figure(figsize=(10,4))
plt.plot(residuals)
plt.title("Residuals over time")
plt.xlabel("Time index")
plt.ylabel("Residual")
plt.show()

# Histogram
plt.figure(figsize=(6,4))
plt.hist(residuals, bins=30)
plt.title("Residual Distribution")
plt.show()

# ACF
sm.graphics.tsa.plot_acf(residuals, lags=40)
plt.show()

print("✅ Residual diagnostics completed")
