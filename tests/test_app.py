import os
import sys
from fastapi.testclient import TestClient

# Ensure root directory is on sys.path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app  # noqa: E402

client = TestClient(app)


def test_health_check_returns_200():
    """1. Health check endpoint must return HTTP 200 and model_loaded=True."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_ui_load_returns_html():
    """2. Root UI endpoint GET / must return 200 and HTML content."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Food Delivery ETA Prediction System" in response.text


def test_domain_logic_short_clear_vs_long_stormy():
    """3. Domain-logic test: short distance & clear weather yields lower ETA than long distance & stormy jam."""
    short_clear_payload = {
        "Delivery_person_Age": 25,
        "Delivery_person_Ratings": 4.9,
        "distance_km": 2.0,
        "Weather_conditions": "Sunny",
        "Road_traffic_density": "Low",
        "Vehicle_condition": 3,
        "Type_of_order": "Snack",
        "Type_of_vehicle": "motorcycle",
        "multiple_deliveries": 0,
        "Festival": "No",
        "City": "Urban",
    }
    long_stormy_payload = {
        "Delivery_person_Age": 36,
        "Delivery_person_Ratings": 4.0,
        "distance_km": 16.0,
        "Weather_conditions": "Stormy",
        "Road_traffic_density": "Jam",
        "Vehicle_condition": 1,
        "Type_of_order": "Buffet",
        "Type_of_vehicle": "scooter",
        "multiple_deliveries": 2,
        "Festival": "Yes",
        "City": "Metropolitian",
    }

    res_short = client.post("/predict", json=short_clear_payload)
    res_long = client.post("/predict", json=long_stormy_payload)

    assert res_short.status_code == 200
    assert res_long.status_code == 200

    eta_short = res_short.json()["predicted_eta_min"]
    eta_long = res_long.json()["predicted_eta_min"]

    assert eta_short < eta_long, (
        f"Expected short/clear ETA ({eta_short}) < long/stormy ETA ({eta_long})"
    )
    assert 2 <= len(res_short.json()["tips"]) <= 3


def test_boundary_validation_out_of_range_returns_422():
    """4. Out-of-range numeric inputs must be rejected with HTTP 422 Unprocessable Entity."""
    invalid_numeric_payload = {
        "Delivery_person_Age": 12,  # Below minimum 18
        "Delivery_person_Ratings": 6.5,  # Above maximum 5.0
        "distance_km": -3.0,  # Below minimum 0.5
        "Weather_conditions": "Sunny",
        "Road_traffic_density": "Low",
        "Vehicle_condition": 2,
        "Type_of_order": "Meal",
        "Type_of_vehicle": "motorcycle",
        "multiple_deliveries": 0,
        "Festival": "No",
        "City": "Urban",
    }
    response = client.post("/predict", json=invalid_numeric_payload)
    assert response.status_code == 422


def test_boundary_validation_invalid_category_returns_422():
    """5. Invalid categorical enum values or missing fields must return HTTP 422."""
    invalid_category_payload = {
        "Delivery_person_Age": 28,
        "Delivery_person_Ratings": 4.5,
        "distance_km": 5.0,
        "Weather_conditions": "Blizzard",  # Not in Data Contract
        "Road_traffic_density": "Extreme",  # Not in Data Contract
        "Vehicle_condition": 2,
        "Type_of_order": "Meal",
        "Type_of_vehicle": "motorcycle",
        "multiple_deliveries": 1,
        "Festival": "No",
        "City": "Urban",
    }
    response = client.post("/predict", json=invalid_category_payload)
    assert response.status_code == 422
