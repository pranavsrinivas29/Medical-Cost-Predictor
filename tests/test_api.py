from fastapi.testclient import TestClient

import app.main as main


client = TestClient(main.app)


def test_health_endpoint_returns_json():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "artifacts_loaded" in data


def test_predict_success_with_mock(monkeypatch):
    # Mock predict_one so we do not need model/preprocessor files
    def fake_predict_one(payload: dict) -> float:
        return 12345.67

    monkeypatch.setattr(main, "predict_one", fake_predict_one)

    payload = {
        "age": 19,
        "sex": "female",
        "bmi": 27.9,
        "children": 0,
        "smoker": "yes",
        "region": "southwest",
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data
    assert data["prediction"] == 12345.67


def test_predict_validation_error():
    # Missing required fields -> FastAPI/Pydantic returns 422
    payload = {
        "age": 19,
        "sex": "female",
        # bmi missing
        "children": 0,
        "smoker": "yes",
        "region": "southwest",
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 422
