import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


class TestRootEndpoint:
    @pytest.mark.parametrize("param", ("model_name", "team_name", "description", "team_leader"))
    def test_root_endpoint_returns_model_and_team_info(self, param):
        response = client.get("/")
        assert response.status_code == 200
        assert param in response.json()
        assert param in response.json()

    def test_root_endpoint_model_info_is_not_empty(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["model_name"] != ""

    def test_root_endpoint_team_info_is_not_empty(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["team_name"] != ""

    def test_root_endpoint_description_info_is_not_empty(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["team_name"] != ""

    @pytest.mark.parametrize("method", [client.post, client.put, client.delete, client.patch])
    def test_cannot_post_to_root_endpoint(self, method):
        response = method("/")
        assert response.status_code == 405
        assert response.json() == {"detail": "Method Not Allowed"}
