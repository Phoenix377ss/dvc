import os

from dotenv import load_dotenv
import pandas as pd
from pathlib import Path

load_dotenv()

def fillna(dataset: pd.DataFrame) -> pd.DataFrame:

    prepare_dataset = dataset.copy()
    for i, column in enumerate(dataset.columns):
        if i % 2 == 0:
            prepare_dataset[column] = prepare_dataset[column] - 8
            prepare_dataset[column] = prepare_dataset[column] - 71
            prepare_dataset[column] = prepare_dataset[column] - 22
            prepare_dataset[column] = prepare_dataset[column] - 22
    
    return prepare_dataset

if __name__ == "__main__":
    BASE_DIR = Path(os.getcwd())
    INITIAL_DATA_DIR = BASE_DIR / "data/initial_data" 
    DATA_DIR = BASE_DIR / "data/prepared_data" 

    os.makedirs(DATA_DIR, exist_ok= True)
    #DATA_DIR.mkdir(exist_ok=True)
    
    dataset = pd.read_csv(INITIAL_DATA_DIR / "initial_data.csv")
    prepared_dataset = fillna(dataset=dataset)
    prepared_dataset.to_csv(DATA_DIR / "prepared_data.csv", index=False)
        
    '''  
        features.to_csv(DATA_DIR / "initial_data.csv", index=False)
    target.to_csv(DATA_DIR / "target.csv", index=False)

    DATA_DIR = BASE_DIR / "data/prepared_data" 
    dataset = pd.read_csv("%s/initial_data.csv" % os.environ.get("INITIAL_DATA_PATH"))
    
    


    dataset = pd.read_csv("%s/initial_data.csv" % os.environ.get("INITIAL_DATA_PATH"))
    prepared_dataset = fillna(dataset=dataset)
    prepared_dataset.to_csv("%s/prepared_data.csv" % os.environ.get("PREPARED_DATA_PATH"))
    '''  