import app as app_module
from fastapi.testclient import TestClient


class FakeModel:
    def predict(self, texts):
        return ["compromised_card"]

    def decision_function(self, texts):
        return [[0.1, 1.2]]


def test_health_check():
    client = TestClient(app_module.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_triage_endpoint(monkeypatch):
    monkeypatch.setattr(app_module, "get_model", lambda: FakeModel())
    client = TestClient(app_module.app)

    response = client.post(
        "/triage",
        json={"text": "Someone used my card without permission and I need urgent help."},
    )

    assert response.status_code == 200
    assert response.json()["category"] == "compromised_card"
    assert response.json()["priority"] == "HIGH"
    assert response.json()["human_review_required"] is True


def test_triage_rejects_short_text():
    client = TestClient(app_module.app)

    response = client.post("/triage", json={"text": "bad"})

    assert response.status_code == 422