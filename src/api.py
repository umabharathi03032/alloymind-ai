"""
api.py
------
Serves the trained alloy-adjustment model over HTTP using FastAPI.
Includes:
  - Input validation (rejects negative/absurd readings)
  - A rolling-average smoothing buffer to reduce sensor noise
  - A rule-based "decision engine" that vetoes unsafe recommendations
  - A /health endpoint and a /history endpoint for observability
  - Per-request latency logging, so real numbers back real claims
"""
import time
from collections import deque
from typing import Dict, List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

ELEMENTS = ["C", "Si", "Mn", "P", "S", "Cr", "Ni"]
import os
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "alloy_model.pkl")

# Safe post-adjustment bounds for each element (min%, max%) -- used by the
# decision engine to veto a recommendation that would push an element
# out of a safe range.
SAFE_RANGE = {
    "C": (0.05, 0.90),
    "Si": (0.20, 2.20),
    "Mn": (0.20, 1.80),
    "P": (0.00, 0.08),
    "S": (0.00, 0.06),
    "Cr": (0.10, 2.80),
    "Ni": (0.00, 1.30),
}

model = joblib.load(MODEL_PATH)
app = FastAPI(title="AlloyMindAI", version="0.1.0")

# In-memory rolling buffer per element for smoothing + a request log for /history
buffers: Dict[str, deque] = {el: deque(maxlen=20) for el in ELEMENTS}
request_log: List[dict] = []


class Reading(BaseModel):
    C: float
    Si: float
    Mn: float
    P: float
    S: float
    Cr: float
    Ni: float

    @field_validator("*")
    @classmethod
    def must_be_reasonable(cls, v: float):
        if v < 0:
            raise ValueError("Element percentage cannot be negative")
        if v > 10:
            raise ValueError("Element percentage looks unrealistic (>10%)")
        return v


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(reading: Reading):
    start = time.perf_counter()

    raw = reading.model_dump()
    for el in ELEMENTS:
        buffers[el].append(raw[el])
    smoothed = {el: float(np.mean(buffers[el])) for el in ELEMENTS}

    features = np.array([[smoothed[el] for el in ELEMENTS]])
    prediction = model.predict(features)[0]
    adjustment = dict(zip(ELEMENTS, prediction.tolist()))

    # Decision engine: veto any adjustment that would push an element
    # outside its safe post-adjustment range.
    vetoed = []
    for el, adj in adjustment.items():
        projected = smoothed[el] + adj
        low, high = SAFE_RANGE[el]
        if not (low <= projected <= high):
            vetoed.append(el)
            adjustment[el] = 0.0  # veto: no adjustment recommended

    latency_ms = (time.perf_counter() - start) * 1000

    result = {
        "smoothed_reading": smoothed,
        "recommended_adjustment": adjustment,
        "vetoed_elements": vetoed,
        "latency_ms": round(latency_ms, 3),
    }
    request_log.append(result)
    return result


@app.get("/history")
def history(limit: int = 10):
    return request_log[-limit:]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
