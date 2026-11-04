import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from forecasting.models import LSTMForecaster, RNNForecaster


MODEL_CLASSES = {"rnn": RNNForecaster, "lstm": LSTMForecaster}
MODEL_PATH = Path(os.getenv("MODEL_PATH", "forecasting/artifacts/best_model.pt"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() and os.getenv("DEVICE") == "cuda" else "cpu")


class ForecastRequest(BaseModel):
    values: list[float] = Field(..., min_length=1)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model = MODEL_CLASSES[checkpoint["model_name"]](
        hidden_size=checkpoint["hidden_size"]
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


app = FastAPI(title="Time-series Forecasting Inference Server")
model, checkpoint = load_model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": checkpoint["model_name"],
        "device": str(DEVICE),
        "sequence_length": checkpoint["sequence_length"],
    }


@app.post("/predict")
def predict(request: ForecastRequest):
    sequence_length = checkpoint["sequence_length"]
    if len(request.values) != sequence_length:
        raise HTTPException(
            status_code=422,
            detail=f"Exactly {sequence_length} values are required",
        )
    values = torch.tensor(request.values, dtype=torch.float32)
    normalized = (values - checkpoint["mean"]) / checkpoint["std"]
    with torch.no_grad():
        prediction = model(normalized.view(1, sequence_length, 1).to(DEVICE))
    value = prediction.item() * checkpoint["std"] + checkpoint["mean"]
    return {
        "prediction_celsius": value,
        "model": checkpoint["model_name"],
        "input_points": sequence_length,
    }


@app.get("/model")
def model_info():
    return {
        "model": checkpoint["model_name"],
        "run_id": checkpoint.get("run_id"),
        "val_loss": checkpoint.get("val_loss"),
    }
