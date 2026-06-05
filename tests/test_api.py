from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_prediction_validation_failure():
    # Pass an impossible age and invalid class to trigger a validation block
    invalid_payload = {
        "Pclass": 5, 
        "Age": -10.0,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 50.0,
        "IsMale": 1
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422