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


def test_triage_endpoint_with_rag_guidance(monkeypatch):
    monkeypatch.setattr(app_module, "get_model", lambda: FakeModel())
    client = TestClient(app_module.app)

    response = client.post(
        "/triage",
        json={"text": "Someone used my card without permission and I need urgent help."},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["category"] == "compromised_card"
    assert data["priority"] == "HIGH"
    assert data["recommended_policy"] == "Compromised card or unrecognised card payment"
    assert "freeze the card" in data["recommended_guidance"]
    assert data["retrieval_similarity"] > 0
    assert data["human_review_required"] is True


def test_triage_rejects_short_text():
    client = TestClient(app_module.app)

    response = client.post("/triage", json={"text": "bad"})

    assert response.status_code == 422