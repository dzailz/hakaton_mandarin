import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from box import ConfigBox
from ruamel.yaml import YAML

from settings import DVC_PARAMS_FILE, MODELS_FOLDER
from src.data_preparation.data_preparation import DataPreparation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

yaml = YAML(typ="safe")
params = ConfigBox(yaml.load(open(Path(DVC_PARAMS_FILE))))
models_folder = Path(MODELS_FOLDER)
bank_acronyms = ["a", "b", "c", "d", "e"]
models_names = [f"random_forest_100_bank_{i}.pkl" for i in bank_acronyms]
bank_names = [f"bank_{i}" for i in bank_acronyms]
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
    """
    Predict the bank decision based on the input data.

    This function takes a dictionary of input data, prepares it for prediction, and makes predictions using the trained models.
    The preparation steps include converting the input data to a DataFrame, applying various data preparation steps, and returning the prepared data as a DataPreparation object.
    The prediction steps include iterating over the list of trained models, making a prediction for each model, and storing the predictions in a dictionary.
    If an error occurs during the prediction, it returns the error.

    Args:
        data (dict): The input data to be prepared for prediction.

    Returns:
        ValueError | dict: The prediction results if the prediction is successful, or an error if an error occurs during the prediction.
    """
    dp = prepare_data(data)
    results = make_predictions(dp)
    return results


def prepare_data(data: dict) -> DataPreparation:
    """
    Prepare the input data for prediction.

    This function takes a dictionary of input data, converts it to a DataFrame, and applies various data preparation steps.
    These steps include converting the index to an integer, converting certain columns to specific types,
    transforming categorical columns, adding time features, applying one-hot encoding, scaling certain columns,
    and dropping unnecessary columns.
    The prepared data is returned as a DataPreparation object.

    Args:
        data (dict): The input data to be prepared for prediction.

    Returns:
        DataPreparation: The prepared data for prediction.
    """
    input_data = pd.DataFrame([data])
    dp = DataPreparation(df=input_data, ohe_model=ohe_model, scaler=scaler)
    dp.index_as_int()
    dp.columns_to_type(columns=params.preprocess.numeric_categorical_columns, dtype="UInt8")
    dp.columns_to_type(columns=params.preprocess.money_columns, dtype="int64")
    dp.transform_to_categorical(columns=params.preprocess.categorical_columns + params.preprocess.numeric_categorical_columns)
    dp.add_time_features()
    dp.transform_one_hot_encoder(is_new=False)
    dp.df[params.preprocess.money_columns] = scaler.transform(dp.df[params.preprocess.money_columns])
    dp.drop_columns(columns=["position"])
    return dp


def make_predictions(dp: DataPreparation) -> ValueError | dict[Any, dict[str, dict[str, Any] | Any]]:
    """
    Make predictions using the trained models.

    This function takes a DataPreparation object, which contains the prepared data for prediction.
    It iterates over the list of trained models and makes a prediction for each model.
    The predictions are stored in a dictionary, with the bank name as the key and the prediction results as the value.
    If an error occurs during the prediction, it returns the error.

    Args:
        dp (DataPreparation): The prepared data for prediction.

    Returns:
        ValueError | dict: The prediction results if the prediction is successful, or an error if an error occurs during the prediction.
    """
    results = {}
    try:
        for model, bank_name in zip(model_list, bank_names):
            prediction = model.predict(dp.df)
            proba = model.predict_proba(dp.df)
            results[bank_name] = {"prediction": prediction[0], "probability": {"denied": proba[0][0], "success": proba[0][1]}}
    except ValueError as e:
        return e
    return results
