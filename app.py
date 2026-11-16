from datetime import datetime, timezone
import json
import os
from typing import List, Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = os.path.join("model", "model.joblib")
STATIC_DIR = "static"
PREDICTION_LOG_PATH = "predictions.log"

# Load serialized model pipeline ONCE globally at startup (not per request)
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model file not found at {MODEL_PATH}. Run `python train.py` first."
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Food Delivery ETA Prediction API",
    description=(
        "Production-grade MLOps service for predicting food delivery ETA (in minutes) "
        "with strict Pydantic data contract validation, rule-based operational tips, "
        "and request logging for data drift monitoring."
    ),
    version="1.0.0",
)

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class DeliveryETARequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "Delivery_person_Age": 28,
                "Delivery_person_Ratings": 4.7,
                "distance_km": 5.2,
                "Weather_conditions": "Sunny",
                "Road_traffic_density": "Medium",
                "Vehicle_condition": 2,
                "Type_of_order": "Meal",
                "Type_of_vehicle": "motorcycle",
                "multiple_deliveries": 1,
                "Festival": "No",
                "City": "Urban",
            }
        },
    )

    Delivery_person_Age: int = Field(
        ...,
        ge=18,
        le=65,
        description="Age of the delivery partner in years (18 to 65).",
    )
    Delivery_person_Ratings: float = Field(
        ...,
        ge=1.0,
        le=5.0,
        description="Historical customer rating of the delivery partner (1.0 to 5.0).",
    )
    distance_km: float = Field(
        ...,
        ge=0.5,
        le=50.0,
        description="Haversine distance between restaurant and customer in km (0.5 to 50.0).",
    )
    Weather_conditions: Literal[
        "Sunny", "Cloudy", "Windy", "Fog", "Stormy", "Sandstorms"
    ] = Field(..., description="Current ambient weather condition.")
    Road_traffic_density: Literal["Low", "Medium", "High", "Jam"] = Field(
        ..., description="Current road traffic congestion level."
    )
    Vehicle_condition: int = Field(
        ...,
        ge=0,
        le=3,
        description="Condition rating of the delivery vehicle (0=Poor to 3=Excellent).",
    )
    Type_of_order: Literal["Snack", "Meal", "Drinks", "Buffet"] = Field(
        ..., description="Category of food order being delivered."
    )
    Type_of_vehicle: Literal[
        "motorcycle", "scooter", "electric_scooter", "bicycle"
    ] = Field(..., description="Type of vehicle used by the courier.")
    multiple_deliveries: int = Field(
        ...,
        ge=0,
        le=3,
        description="Number of additional stacked deliveries on the same trip (0 to 3).",
    )
    Festival: Literal["No", "Yes"] = Field(
        ..., description="Whether the order is placed on a public festival day."
    )
    City: Literal["Metropolitian", "Urban", "Semi-Urban"] = Field(
        ..., description="Urban classification of the delivery zone."
    )


class DeliveryETAResponse(BaseModel):
    predicted_eta_min: float
    eta_window_min: str
    tips: List[str]
    status: str = "success"


def generate_rule_based_tips(payload: DeliveryETARequest, predicted_eta: float) -> List[str]:
    tips: List[str] = []

    if payload.Road_traffic_density in ("High", "Jam"):
        tips.append(
            "Heavy traffic alert: Route via secondary arterial roads and notify the customer of potential congestion delays."
        )
    elif payload.Road_traffic_density == "Low":
        tips.append(
            "Clear road conditions: Standard direct route recommended for optimal fuel and battery efficiency."
        )

    if payload.Weather_conditions in ("Stormy", "Sandstorms", "Fog"):
        tips.append(
            f"Adverse weather ({payload.Weather_conditions}): Use waterproof thermal sealing and allow a 5-minute safety buffer."
        )

    if payload.multiple_deliveries >= 2:
        tips.append(
            f"Multi-order batch ({payload.multiple_deliveries} extra stops): Sequence drop-offs by proximity and keep hot/cold items separated."
        )

    if payload.distance_km >= 10.0:
        tips.append(
            f"Long-distance trip ({payload.distance_km:.1f} km): Use an insulated high-retention bag to preserve food temperature."
        )

    if payload.Vehicle_condition == 0 or payload.Type_of_vehicle == "bicycle":
        tips.append(
            "Vehicle constraint detected: Prioritize flat urban corridors and avoid steep flyovers where possible."
        )

    if payload.Festival == "Yes":
        tips.append(
            "Festival surge active: Expect longer restaurant kitchen handover queues; dispatch courier slightly closer to prep completion."
        )

    if len(tips) < 2:
        tips.append(
            "Standard dispatch: Maintain live GPS tracking updates for smooth customer handover at the drop-off point."
        )
    if len(tips) < 2:
        tips.append(
            f"Estimated arrival around {round(predicted_eta)} minutes; verify order seal before leaving the restaurant."
        )

    return tips[:3]


def log_prediction_request(payload: DeliveryETARequest, predicted_eta: float) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload.model_dump(),
        "predicted_eta_min": round(predicted_eta, 2),
    }
    with open(PREDICTION_LOG_PATH, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record) + "\n")


@app.get("/", include_in_schema=False)
def serve_ui() -> FileResponse:
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="UI file static/index.html not found.")
    return FileResponse(index_path)


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "food-delivery-eta",
    }


@app.post("/predict", response_model=DeliveryETAResponse)
def predict_eta(request: DeliveryETARequest) -> DeliveryETAResponse:
    input_df = pd.DataFrame([request.model_dump()])
    raw_pred = float(model.predict(input_df)[0])
    predicted_eta = round(max(5.0, raw_pred), 2)

    low_bound = max(5, int(round(predicted_eta - 2)))
    high_bound = int(round(predicted_eta + 3))
    eta_window = f"{low_bound} - {high_bound} mins"

    tips = generate_rule_based_tips(request, predicted_eta)
    log_prediction_request(request, predicted_eta)

    return DeliveryETAResponse(
        predicted_eta_min=predicted_eta,
        eta_window_min=eta_window,
        tips=tips,
        status="success",
    )
