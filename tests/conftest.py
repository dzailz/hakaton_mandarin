from datetime import datetime
from random import choice, randint

import numpy as np
import pandas as pd
import pytest

from src.api.data_structures.bank_decision import BankDecisionInput
from src.data_preparation.data_preparation import DataPreparation

NUM_SAMPLES = 500


@pytest.fixture()
def normal_dist_income(num_samples: int = 500) -> np.ndarray:
    mean_income, std_dev_income = 250000, 100000
    income_samples = np.clip(np.random.normal(mean_income, std_dev_income, num_samples), 10000, 500000)
    return income_samples


@pytest.fixture()
def normal_dist_age(num_samples: int = 500) -> np.ndarray:
    mean_age, std_dev_age = 42, 10
    age_samples = np.clip(np.random.normal(mean_age, std_dev_age, num_samples), 19, 65)
    return age_samples


@pytest.fixture()
def normal_data(normal_dist_income, normal_dist_age) -> pd.DataFrame:
    # Create synthetic data for the DataFrame
    data = {
        "age": normal_dist_age,
        "income": normal_dist_income,
        "education": np.random.choice(["Bachelor", "Master", "PhD", "Bachelor"], NUM_SAMPLES),
        "bank_decision": np.random.choice(["declined", "approved"], NUM_SAMPLES),
    }

    # Convert the data dictionary to a Pandas DataFrame
    return pd.DataFrame(data)


@pytest.fixture()
def lognormal_data() -> pd.DataFrame:
    # Set the mean, standard deviation, and number of samples
    mean, std_dev, num_samples = 0, 1, 1000

    # Create synthetic data for the DataFrame
    data = {
        "age": np.random.choice(range(18, 65), num_samples),
        "income": np.random.lognormal(mean, std_dev, num_samples),
        "education": np.random.choice(["Bachelor", "Master", "PhD", "Bachelor"], num_samples),
        "bank_decision": np.random.choice(["declined", "approved"], num_samples),
    }

    # Convert the data dictionary to a Pandas DataFrame
    return pd.DataFrame(data)


@pytest.fixture()
def normal_data_preparation(normal_data) -> DataPreparation:
    return DataPreparation(df=normal_data, to_drop_columns=[], target_bank_col="Bank")


@pytest.fixture()
def lognormal_data_preparation(lognormal_data) -> DataPreparation:
    return DataPreparation(df=lognormal_data, to_drop_columns=[], target_bank_col="Bank")


@pytest.fixture()
def bank_names() -> list[str]:
    return ["bank_a", "bank_b", "bank_c", "bank_d", "bank_e"]


@pytest.fixture()
def input_data() -> BankDecisionInput:
    birth_date = str(datetime.date(datetime.strptime("1980-01-01", "%Y-%m-%d")))
    job_start_date = str(datetime.date(datetime.strptime("2000-01-01", "%Y-%m-%d")))
    month_profit = randint(10000, 10000000)
    month_expense = randint(10000, 10000000)
    gender = choice([0, 1])
    child_count = choice([0, 1, 2, 3, 4])
    snils = choice([0, 1])
    loan_amount = randint(10000, 10000000)
    loan_term = choice([6, 12, 18, 24])

    return BankDecisionInput(
        birth_date=birth_date,
        education="Высшее - специалист",
        employment_status="Работаю по найму полный рабочий день/служу",
        value="10 и более лет",
        job_start_date=job_start_date,
        position="Manager",
        month_profit=month_profit,
        month_expense=month_expense,
        gender=gender,
        family_status="Никогда в браке не состоял(а)",
        child_count=child_count,
        snils=snils,
        merch_code=61,
        loan_amount=loan_amount,
        loan_term=loan_term,
        goods_category="Mobile_devices",
    )
