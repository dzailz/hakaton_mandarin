from random import choice, randint

import pytest
from fastapi.testclient import TestClient
from main import app, model, scaler
from src.api.data_structures.BankDecisionInput import BankDecisionInput
from datetime import datetime

client = TestClient(app)


class TestPredictHandler:
    def test_predict_bank_decision_with_valid_input(self):
        # Given
        input_data = BankDecisionInput(
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

        # When
        response = client.post("/predict_bank_decision", json=input_data.model_dump())

        # Then
        assert response.status_code == 200
        assert 'prediction' in response.json()
        for prediction in response.json()['prediction']:
            assert prediction in ['denied', 'success'], f"Expected int, got {type(response.json()['prediction'])}"
        for probability in response.json()['probability']:
            assert 0 <= probability <= 1, f"Expected int, got {type(response.json()['probability'])}"

    def test_predict_bank_decision_with_missing_input(self):
        # Given
        input_data = BankDecisionInput(
            month_profit=5000,
            month_expense=2000,
            loan_amount=10000,
            position='Manager',
            birth_date='1980-01-01',
            job_start_date='2000-01-01',
            education='Bachelor',
            employment_status='Employed',
            gender='Man',
            family_status='Married',
            child_count=2,
            loan_term=12,
            goods_category='Electronics',
            value=1,
            snils=1,
            merch_code=111,
            # Missing bank
        )

        # When
        response = client.post("/predict_bank_decision", json=input_data.model_dump())

        # Then
        assert response.status_code == 422

    def test_predict_bank_decision_with_invalid_input(self):
        # Given
        input_data = BankDecisionInput(
            month_profit='5000',  # Should be int
            month_expense=2000,
            loan_amount=10000,
            position='Manager',
            birth_date='1980-01-01',
            job_start_date='2000-01-01',
            education='Bachelor',
            employment_status='Employed',
            gender='Man',
            family_status='Married',
            child_count=2,
            loan_term=12,
            goods_category='Electronics',
            value=1,
            snils=1,
            merch_code=111,
            bank='Bank1'
        )

        # When
        response = client.post("/predict_bank_decision", json=input_data.model_dump())

        # Then
        assert response.status_code == 422
