import argparse
import os
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = os.path.join("data", "food_delivery_eta.csv")
MODEL_PATH = os.path.join("model", "model.joblib")
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "food-delivery-eta-prediction"

NUMERIC_FEATURES = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "distance_km",
    "Vehicle_condition",
    "multiple_deliveries",
]

CATEGORICAL_FEATURES = [
    "Weather_conditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
]

TARGET_COLUMN = "Time_taken_min"


def build_pipeline(
    model_type: str,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    alpha: float,
    random_state: int,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    if model_type == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        regressor = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
    elif model_type == "ridge":
        regressor = Ridge(alpha=alpha, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Food Delivery ETA regression model with MLflow tracking."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="random_forest",
        choices=["random_forest", "gradient_boosting", "ridge"],
        help="Regression algorithm to train (default: random_forest)",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of trees for ensemble models (default: 100)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=6,
        help="Maximum tree depth for ensemble models (default: 6)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate for gradient boosting (default: 0.1)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Regularization strength for Ridge regression (default: 1.0)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    run_name = (
        f"{args.model_type}_n{args.n_estimators}_d{args.max_depth}"
        if args.model_type != "ridge"
        else f"ridge_alpha{args.alpha}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("alpha", args.alpha)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        pipeline = build_pipeline(
            model_type=args.model_type,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            alpha=args.alpha,
            random_state=args.random_state,
        )

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        mlflow.log_metric("rmse", round(rmse, 4))
        mlflow.log_metric("mae", round(mae, 4))
        mlflow.log_metric("r2", round(r2, 4))

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)

        print(f"Run: {run_name}")
        print(f"Metrics -> RMSE: {rmse:.4f} min | MAE: {mae:.4f} min | R2: {r2:.4f}")
        print(f"Saved serialized model pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
