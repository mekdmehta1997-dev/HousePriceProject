import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

print("Training Started...")

# ---- IMPORTANT ----
mlflow.set_experiment("house-price-prediction")
# -------------------

# Dummy example dataset
df = pd.read_csv("/app/data/california.csv")

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("MSE:", mse)
print("R2 :", r2)

with mlflow.start_run():
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(model, "model")

print("Training completed")
