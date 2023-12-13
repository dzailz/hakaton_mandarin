from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from random import choice, randint

from src.api.data_structures.bank_decision import BankDecisionInput
from src.data_preparation.data_preparation import DataPreparation

NUM_SAMPLES = 500


@pytest.fixture()
def normal_dist_income() -> np.ndarray:
    # Set the mean and standard deviation for income
    mean_income = 250000  # The mean income
    std_dev_income = 100000  # The standard deviation of income

    income_samples = np.random.normal(mean_income, std_dev_income, NUM_SAMPLES)

    # Ensure income values are within the desired range (10,000 to 500,000)
    income_samples = np.clip(income_samples, 10000, 500000)
    return income_samples


@pytest.fixture()
def normal_dist_age() -> np.ndarray:
    # Set the mean and standard deviation for age
    mean_age = 42  # The mean is set to the midpoint between 19 and 65
    std_dev_age = 10  # The standard deviation is set to 1/4 of the range (65 - 19) / 4

    age_samples = np.random.normal(mean_age, std_dev_age, NUM_SAMPLES)

    # Ensure values are within the desired range (19 to 65)
    age_samples = np.clip(age_samples, 19, 65)
    return age_samples


@pytest.fixture()
def normal_data(normal_dist_income, normal_dist_age) -> pd.DataFrame:
    # Create synthetic data for the DataFrame
    data = {
        'age': normal_dist_age,
        'income': normal_dist_income,
        'education': [choice(['Bachelor', 'Master', 'PhD', 'Bachelor']) for _ in range(NUM_SAMPLES)],
        'bank_decision': [choice(['declined', 'approved']) for _ in range(NUM_SAMPLES)]
    }

    # Convert the data dictionary to a Pandas DataFrame
    df = pd.DataFrame(data)

    return df


@pytest.fixture()
def lognormal_data() -> pd.DataFrame:
    # Set the mean and standard deviation
    mean = 0
    std_dev = 1

    # Generate random samples from a lognormal distribution
    num_samples = 1000
    income_samples = np.random.lognormal(mean, std_dev, num_samples)

    # Create synthetic data for the DataFrame
    data = {
        'age': [choice(range(18, 65)) for _ in range(num_samples)],
        'income': income_samples,
        'education': ['Bachelor', 'Master', 'PhD', 'Bachelor'] * (num_samples // 4),
        'bank_decision': [choice(['declined', 'approved']) for _ in range(num_samples)]
    }

    # Convert the data dictionary to a Pandas DataFrame
    df = pd.DataFrame(data)

    return df


@pytest.fixture()
def normal_data_preparation(normal_data) -> DataPreparation:
    return DataPreparation(df=normal_data, to_drop_columns=[], target_bank_col='Bank')


@pytest.fixture()
def lognormal_data_preparation(lognormal_data) -> DataPreparation:
    return DataPreparation(df=lognormal_data, to_drop_columns=[], target_bank_col='Bank')


@pytest.fixture()
def bank_names() -> list[str]:
    return ['bank_a', 'bank_b', 'bank_c', 'bank_d', 'bank_e']


@pytest.fixture()
def input_data() -> BankDecisionInput:
    return BankDecisionInput(
        birth_date=str(datetime.date(datetime.strptime('1980-01-01', '%Y-%m-%d'))),
        education="Высшее - специалист",
        employment_status="Работаю по найму полный рабочий день/служу",
        value="10 и более лет",
        job_start_date=str(datetime.date(datetime.strptime('2000-01-01', '%Y-%m-%d'))),
        position="Manager",
        month_profit=randint(10000, 10000000),
        month_expense=randint(10000, 10000000),
        gender=choice([0, 1]),
        family_status="Никогда в браке не состоял(а)",
        child_count=choice([0, 1, 2, 3, 4]),
        snils=choice([0, 1]),
        merch_code=61,
        loan_amount=randint(10000, 10000000),
        loan_term=choice([6, 12, 18, 24]),
        goods_category="Mobile_devices"
    )
