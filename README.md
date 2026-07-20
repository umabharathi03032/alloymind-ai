# AlloyMindAI (Simplified Working Version)

🔴 Live demo: https://alloymind-ai-nkec.onrender.com/docs (hosted on Render's free tier — the first request after a period of inactivity can take 30–60s to wake the service up; subsequent requests are fast)

An AI-assisted alloy composition adjustment pipeline: takes a composition
reading, cleans/smooths it, predicts the adjustment needed to hit a target
specification using a trained regression model, checks the recommendation
against safe composition bounds, and serves it over an API.

## Scope note

This is a **scoped-down, real, working version** of the AlloyMindAI concept —
built to be genuinely demoable rather than aspirational. It uses:
- **Synthetic data** (disclosed openly — no real spectrometer log was available;
  see `src/generate_data.py` for exactly how it's generated)
- **In-process request handling**, not a distributed Kafka streaming setup
- **Measured, real latency numbers** (see Results below) rather than assumed ones

## Architecture

```
generate_data.py  →  train_model.py  →  api.py  →  load_test.py
  (synthetic data)    (trains &          (FastAPI      (measures real
                        evaluates          serving       throughput &
                        2 models,          layer with    latency)
                        saves best)        smoothing +
                                           decision
                                           engine)
```

## Files

- `src/generate_data.py` — generates a synthetic dataset of alloy composition
  readings and the "correct" adjustment, based on a defined target spec with
  realistic noise.
- `src/train_model.py` — trains and compares Linear Regression vs. Random
  Forest, evaluates both with MAE/RMSE on a held-out test set, saves the
  better-performing model.
- `src/api.py` — FastAPI service exposing `/predict`, `/health`, `/history`.
  Includes input validation, rolling-average smoothing, and a rule-based
  decision engine that vetoes unsafe recommendations.
- `src/load_test.py` — fires real requests at the running API and reports
  measured average/median/95th-percentile latency and sustained throughput.

## How to run it

```bash
pip install -r requirements.txt

# 1. Generate the dataset
python3 src/generate_data.py

# 2. Train and evaluate the model
python3 src/train_model.py

# 3. Start the API
python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8000

# 4. In another terminal, test it
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"C":0.42,"Si":1.05,"Mn":0.75,"P":0.028,"S":0.018,"Cr":1.15,"Ni":0.48}'

# 5. Measure real latency/throughput
python3 src/load_test.py
```

## Results (actually measured on this run)

**Model comparison** (on a held-out 20% test set, 1000 rows):

| Model | MAE | RMSE |
|---|---|---|
| Linear Regression | 0.01076 | 0.01942 |
| Random Forest | 0.01139 | 0.02065 |

Linear Regression performed slightly better here — which makes sense, since
the synthetic data's true relationship (adjustment ≈ target − reading) is
fundamentally linear. This is a genuine, explainable finding from comparing
models, not an assumption.

**Load test** (300 sequential requests against the local API):

- Sustained throughput: **~330 requests/sec**
- Average latency: **3.0 ms**
- 95th percentile latency: **3.2 ms**
- Max latency: **10.8 ms**

All comfortably under the 500ms target — though note this is measured
locally (no real network hop, no real sensor hardware, no concurrent load
from other processes), so it should be described as such rather than as a
production benchmark.

