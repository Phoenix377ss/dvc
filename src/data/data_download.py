import os
from pathlib import Path
import pandas as pd
from sklearn import datasets
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Текущая рабочая директория (корень проекта)
if __name__ == "__main__":
    BASE_DIR = Path(os.getcwd())
    DATA_DIR = BASE_DIR / "data/initial_data" 

    
    os.makedirs(DATA_DIR, exist_ok= True)

    logger.info(f'Data downloaded in {DATA_DIR}')
    #print(DATA_DIR)
    #DATA_DIR.mkdir(exist_ok=True)

    dataset = datasets.load_diabetes()

    features = pd.DataFrame(
        data=dataset.data,
        columns=[f"feat{x}" for x in range(dataset.data.shape[1])]
    )
    target = pd.DataFrame(data=dataset.target, columns=["target"])

    features.to_csv(DATA_DIR / "initial_data.csv", index=False)
    target.to_csv(DATA_DIR / "target.csv", index=False)
