import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI

from src.api.data_structures.BankDecisionInput import BankDecisionInput
from src.data_preparation.data_preparation import DataPreparation
from settings import MODELS_FOLDER

model_name = "random_forest_100_bank_a.pkl"
scaler_name = "scaler.pkl"

with open(Path(MODELS_FOLDER, model_name), "rb") as model_file:
    model = pickle.load(model_file)

with open(Path(MODELS_FOLDER, scaler_name), "rb") as scaler_file:
    scaler = pickle.load(scaler_file)


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict_bank_decision")
def predict_bank_decision(data: BankDecisionInput):
    # Convert input data to a DataFrame
    input_data = pd.DataFrame([data.model_dump()])
    # input_data = input_data.drop('position', axis=1)
    dp = DataPreparation(df=input_data)
    dp.drop_columns(columns=['position'])
    dp.transform_to_categorical(columns=[
                'education',
                'employment_status',
                'gender',
                'family_status',
                'child_count',
                'loan_term',
                'goods_category',
                'value',
                'snils',
                'merch_code'
            ])
    dp.add_time_features()
    dp.ohe_categorical_columns()
    columns = ['month_profit', 'month_expense', 'loan_amount']
    dp.df[columns] = scaler.fit_transform(dp.df[columns])

    # Make predictions using the trained model
    prediction = model.predict(input_data)

    # Return the prediction as a response
    return {"prediction": prediction[0]}
