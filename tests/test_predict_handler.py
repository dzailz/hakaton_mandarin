import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


class TestPredictHandler:
    def test_predict_bank_decision_with_valid_input(self, bank_names, input_data):
        # When
        response = client.post("/predict_bank_decision", json=input_data.model_dump())
        # Then
        assert response.status_code == 200
        resp = response.json()
        for bank in bank_names:
            assert resp[bank]["prediction"] in ["denied", "success"], f"Expected int, got {type(resp[bank]['prediction'])}"
            assert 0 <= resp[bank]["probability"]["denied"] <= 1, f"Expected int, got {type(resp[bank]['probability']['denied'])}"
            assert 0 <= resp[bank]["probability"]["success"] <= 1, f"Expected int, got {type(resp[bank]['probability']['success'])}"

    @pytest.mark.parametrize(("param", "value"), (("merch_code", 100000), ("goods_category", 100000), ("gender", 2), ("family_status", "test"), ("child_count", 5), ("snils", 2)))
    def test_cannot_predict_bank_decision_with_incorrect_fields(self, input_data, param, value):
        # Given
        setattr(input_data, param, value)
        # When
        response = client.post("/predict_bank_decision", json=input_data.model_dump())
        # Then
        assert response.status_code == 422
