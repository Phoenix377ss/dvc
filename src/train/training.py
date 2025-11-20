import os
from pathlib import Path
import pickle
import json
import argparse
import yaml

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


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

    models_dir.mkdir(exist_ok=True, parents=True)
    metrics_dir.mkdir(exist_ok=True, parents=True)

    # 1) Читаем YAML
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)["train"]

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

    # 4) Модель
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5) Метрики
    metrics = {
        "train_MSE": float(mean_squared_error(y_train, model.predict(X_train))),
        "test_MSE": float(mean_squared_error(y_test, model.predict(X_test))),
        "validation_MSE": float(mean_squared_error(y_val, model.predict(X_val)))
    }

    # 6) Сохраняем модель
    model_path = models_dir / "linear_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # 7) Сохраняем метрики
    metrics_path = metrics_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Using config:", config_path)
    print("Model saved:", model_path)
    print("Metrics:", metrics)
