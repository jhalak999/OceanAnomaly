import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# loading ml ready data

X_train = pd.read_csv("dataset/final/X_train_ml.csv")
X_test  = pd.read_csv("dataset/final/X_test_ml.csv")
y_train = pd.read_csv("dataset/final/y_train.csv").values.ravel()
y_test  = pd.read_csv("dataset/final/y_test.csv").values.ravel()


# evaluation
def evaluate(model, name):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n {name}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"R²  : {r2:.3f}")


# 1.linear regression model

lr = LinearRegression()
lr.fit(X_train, y_train)
evaluate(lr, "Linear Regression")


# 2. random forest model

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)
rf.fit(X_train, y_train)
evaluate(rf, "Random Forest Regressor")

