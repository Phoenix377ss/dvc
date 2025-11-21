import os
import json
import argparse
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn  # или mlflow.sklearn.autolog, если захочешь автологирование
from pathlib import Path
import json
import optuna
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split





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



def prepare_data_and_config():

    base_dir = Path(os.getcwd())
    data_initial_dir = base_dir / "data" / "initial_data"
    data_prepared_dir = base_dir / "data" / "prepared_data"


    
    ###    config part
    args = parse_args()
    config_path = Path(args.config)
    # 1) Читаем YAML
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)["tuning"]

    ###     data part  

    SEED = params["seed"]
    VAL_SIZE = params["val_size"]
    np.random.seed(SEED)

    # 2) Загружаем данные
    X = pd.read_csv(data_prepared_dir / "prepared_data.csv")
    y = pd.read_csv(data_initial_dir / "target.csv")["target"]

    X_tr, X_v, y_tr, y_v = train_test_split(
        X,
        y,
        test_size=VAL_SIZE,
        random_state=SEED,
    )

    return X_tr, X_v, y_tr, y_v, params



def objective(trial):
    global X_train, X_val, y_train, y_val

    # гиперпараметры
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }

    # nested run для каждого trial
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(params)

        model = RandomForestRegressor(
            **params,
            random_state=42,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        val_mse = mean_squared_error(y_val, y_pred)

        # логируем метрику
        mlflow.log_metric("val_MSE", val_mse)

        # при желании можно модель залогировать как артефакт trial-а
        #mlflow.sklearn.log_model(model, artifact_path="model")

        return val_mse  # Optuna минимизирует это
    



if __name__ == "__main__":

    #args = parse_args()
    #print(args)
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--n-trials", type=int, default=None)
    #args = parser.parse_args()

    # готовим данные один раз
    X_train, X_val, y_train, y_val, train_cfg = prepare_data_and_config()
    n_trials = train_cfg.get("n_trials", 10)

    np.random.seed(train_cfg["seed"])

    configs_dir = base_dir / "configs"
    configs_dir.mkdir(exist_ok=True, parents=True)

    # выбираем эксперимент
    mlflow.set_experiment("rf_tuning_dvc")

    # общий run "Hyperparameter Tuning"
    with mlflow.start_run(run_name="Hyperparameter Tuning") as parent_run:
        mlflow.log_param("n_trials", n_trials)

        # создаем Optuna study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_trial = study.best_trial

        # логируем лучшие параметры и итоговую метрику в родительский run
        mlflow.log_params({f"best_{k}": v for k, v in best_trial.params.items()})
        mlflow.log_metric("best_val_MSE", best_trial.value)

        # сохраняем лучший результат в файл (для DVC/тренировки)
        best_params = {
            "model": "RandomForestRegressor",
            "best_val_MSE": best_trial.value,
            "best_trial_number": best_trial.number,
            "params": best_trial.params,
        }

        out_path = configs_dir / "best_params.json"
        with open(out_path, "w") as f:
            json.dump(best_params, f, indent=4)

        print("Best params:", best_trial.params)
        print("Saved best params to:", out_path)