import logging
from typing import Any

from box import ConfigBox

import pickle
from pathlib import Path
import pandas as pd

from settings import MODELS_FOLDER, DVC_PARAMS_FILE
from ruamel.yaml import YAML

from src.data_preparation.data_preparation import DataPreparation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

yaml = YAML(typ='safe')

params = ConfigBox(yaml.load(open(Path(DVC_PARAMS_FILE))))

categorical_columns = list(params.preprocess.categorical_columns)
numeric_categorical_columns = list(params.preprocess.numeric_categorical_columns)
money_columns = list(params.preprocess.money_columns)

models_folder = Path(MODELS_FOLDER)
bank_acronyms = ['a', 'b', 'c', 'd', 'e']

models_names = [f"random_forest_100_bank_{i}.pkl" for i in bank_acronyms]

bank_names = [f'bank_{i}' for i in bank_acronyms]

model_list = []
scaler_name = "scaler.pkl"
ohe_model = "ohe_model.pkl"

for name in models_names:
    with open(Path(MODELS_FOLDER, name), "rb") as model_file:
        model_list.append(pickle.load(model_file))

with open(Path(MODELS_FOLDER, scaler_name), "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open(Path(MODELS_FOLDER, ohe_model), "rb") as model_file:
    ohe_model = pickle.load(model_file)


def predict(data: dict) -> ValueError | dict[Any, dict[str, dict[str, Any] | Any]]:
    # Convert input data to a DataFrame
    input_data = pd.DataFrame([data])
    # input_data = input_data.drop('position', axis=1)
    dp = DataPreparation(df=input_data, ohe_model=ohe_model, scaler=scaler)
    dp.index_as_int()
    dp.columns_to_type(columns=numeric_categorical_columns, dtype='UInt8')
    dp.columns_to_type(columns=money_columns, dtype='int64')
    dp.transform_to_categorical(columns=categorical_columns + numeric_categorical_columns)
    dp.add_time_features()

    dp.transform_one_hot_encoder(is_new=False)

    dp.df[money_columns] = scaler.transform(dp.df[money_columns])
    dp.drop_columns(columns=['position'])
    # Make predictions using the trained model
    results = {}
    try:
        for model, bank_name in zip(model_list, bank_names):
            prediction = model.predict(dp.df)
            proba = model.predict_proba(dp.df)  # Assuming the model has a predict_proba method

            results[bank_name] = {
                "prediction": prediction[0],
                "probability": {"denied": proba[0][0], "success": proba[0][1]}
            }
    except ValueError as e:
        return e

    return results
