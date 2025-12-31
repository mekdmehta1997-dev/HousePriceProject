import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

print("Training Started...")

# ================= Load Data =================
df = pd.read_csv("/app/data/california.csv")

# ================= Split X and y =================
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# ================= Columns =================
categorical_cols = ["ocean_proximity"]

numeric_cols = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income"
]

# ================= Preprocessor =================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# ================= Model =================
model = XGBRegressor(
    n_estimators=250,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", model)
])

# ================= Split =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= Train =================
pipeline.fit(X_train, y_train)

# ================= Predict =================
preds = pipeline.predict(X_test)

mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("MSE:", mse)
print("R2 :", r2)

# ================= MLflow =================

mlflow.set_experiment("house-price-prediction")

with mlflow.start_run():
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(pipeline, "model")


print("Training Completed Successfully!")

