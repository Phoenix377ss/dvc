import os
from pathlib import Path
import pickle
import json
import argparse
import yaml
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn  # или mlflow.sklearn.autolog, если захочешь автологирование
from pathlib import Path



base_dir = Path(os.getcwd())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    return parser.parse_args()






if __name__ == "__main__":
    args = parse_args()

    # Путь к конфигу
    config_path = Path(args.config)

    # Корень проекта
    base_dir = Path(os.getcwd())

    data_initial_dir = base_dir / "data" / "initial_data"
    data_prepared_dir = base_dir / "data" / "prepared_data"
    models_dir = base_dir / "models"
    metrics_dir = base_dir / "metrics"

    params_dir = base_dir / "configs" / "best_params.json"
    mlruns_dir = base_dir / "mlruns"


    models_dir.mkdir(exist_ok=True, parents=True)
    metrics_dir.mkdir(exist_ok=True, parents=True)
    mlruns_dir.mkdir(exist_ok=True, parents=True)

    #mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    mlflow.set_experiment("model_training")  # любое имя эксперимента

    # 1) Читаем config
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)["train"]
    
    # Читаем параметры модели
    with open(params_dir, "r") as f:
        model_params = json.load(f)["params"]


    
    print(f"Params in this set up are {params}")
    print(f"Model params in this set up are {model_params}")

    SEED = params["seed"]
    VAL_SIZE = params["val_size"]
    TEST_SIZE = params["test_size"]

    # 2) Загружаем данные
    X = pd.read_csv(data_prepared_dir / "prepared_data.csv")
    y = pd.read_csv(data_initial_dir / "target.csv")["target"]

    np.random.seed(SEED)

    # 3) Сплиты
    train_idx, val_idx = train_test_split(
        X.index, test_size=VAL_SIZE, random_state=SEED
    )
    train_idx, test_idx = train_test_split(
        train_idx, test_size=TEST_SIZE, random_state=SEED
    )

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]
    X_val, y_val = X.loc[val_idx], y.loc[val_idx]

    
    with mlflow.start_run():
    # Можно залогировать параметры из конфига
        mlflow.log_params({
            "seed": SEED,
            "val_size": VAL_SIZE,
            "test_size": TEST_SIZE,
            **model_params
        })
    
    
        # 4) Модель
        model = RandomForestRegressor(
            **model_params,
            random_state=42,
        )
        model.fit(X_train, y_train)

        # 5) Метрики
        metrics = {
            "train_MSE": float(mean_squared_error(y_train, model.predict(X_train))),
            "test_MSE": float(mean_squared_error(y_test, model.predict(X_test))),
            "validation_MSE": float(mean_squared_error(y_val, model.predict(X_val)))
        }

        # Логируем метрики в MLflow
        mlflow.log_metrics(metrics)

        # 6) Сохраняем модель на диск (для DVC)
        model_path = models_dir / "best_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # И в MLflow как артефакт (по желанию)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # 7) Сохраняем метрики в файл как раньше
        metrics_path = metrics_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print("Using config:", config_path)
        print("Model saved:", model_path)
        print("Metrics:", metrics)