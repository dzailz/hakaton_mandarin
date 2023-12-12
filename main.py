import pickle
from pathlib import Path

import pandas as pd
from box import ConfigBox
from fastapi import FastAPI

from src.api.data_structures.BankDecisionInput import BankDecisionInput
from src.data_preparation.data_preparation import DataPreparation
from settings import MODELS_FOLDER, DVC_PARAMS_FILE
from ruamel.yaml import YAML

yaml = YAML(typ='safe')

params = ConfigBox(yaml.load(open(Path(DVC_PARAMS_FILE))))

categorical_columns = list(params.preprocess.categorical_columns)
numeric_categorical_columns = list(params.preprocess.numeric_categorical_columns)
money_columns = list(params.preprocess.money_columns)

model_name = "random_forest_100_bank_d.pkl"
scaler_name = "scaler.pkl"
ohe_model = "ohe_model.pkl"

with open(Path(MODELS_FOLDER, model_name), "rb") as model_file:
    model = pickle.load(model_file)

with open(Path(MODELS_FOLDER, scaler_name), "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open(Path(MODELS_FOLDER, ohe_model), "rb") as model_file:
    ohe_model = pickle.load(model_file)


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict_bank_decision")
def predict_bank_decision(data: BankDecisionInput):
    # Convert input data to a DataFrame
    input_data = pd.DataFrame([data.model_dump()])
    # input_data = input_data.drop('position', axis=1)
    dp = DataPreparation(df=input_data, ohe_model=ohe_model)
    dp.index_as_int()
    dp.columns_to_type(columns=numeric_categorical_columns, dtype='UInt8')
    dp.columns_to_type(columns=money_columns, dtype='int64')
    dp.transform_to_categorical(columns=categorical_columns + numeric_categorical_columns)
    dp.add_time_features()

    dp.transform_one_hot_encoder(is_new=False)
    dp.df[money_columns] = scaler.fit_transform(dp.df[money_columns])
    dp.drop_columns(columns=['position'])
    # Make predictions using the trained model
    prediction = model.predict(dp.df)

    # Return the prediction as a response
    return {"prediction": prediction[0]}
