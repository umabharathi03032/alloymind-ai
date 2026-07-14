"""
generate_data.py
-----------------
Generates a synthetic but physically-plausible dataset simulating spectrometer
readings of alloy composition, along with the "correct" adjustment needed to
bring each element to its target specification.

This is SYNTHETIC data (disclosed honestly) since no real spectrometer log
was available. The target values are computed from a defined formula with
added noise, so the model has a genuine, learnable pattern -- not randomness.
"""
import os
import numpy as np
import pandas as pd

np.random.seed(42)

# Target specification (ideal %) for each element in this hypothetical alloy
TARGET_SPEC = {
    "C": 0.40,
    "Si": 1.00,
    "Mn": 0.80,
    "P": 0.03,
    "S": 0.02,
    "Cr": 1.20,
    "Ni": 0.50,
}

ELEMENTS = list(TARGET_SPEC.keys())
N_SAMPLES = 5000


def generate_reading():
    """Simulate one spectrometer reading: target +/- process variation."""
    reading = {}
    for el, target in TARGET_SPEC.items():
        # process variation: readings scatter around the target with some noise
        noise = np.random.normal(0, target * 0.25)
        reading[el] = max(0.001, target + noise)
    return reading


def compute_adjustment(reading):
    """
    The 'ground truth' adjustment a metallurgist would make: move the
    current reading toward the target spec, with a small amount of
    real-world imprecision (a human/process wouldn't correct perfectly).
    """
    adjustments = {}
    for el, target in TARGET_SPEC.items():
        ideal_adjustment = target - reading[el]
        imprecision = np.random.normal(0, abs(ideal_adjustment) * 0.1 + 0.002)
        adjustments[f"adj_{el}"] = ideal_adjustment + imprecision
    return adjustments


def build_dataset(n=N_SAMPLES):
    rows = []
    for _ in range(n):
        reading = generate_reading()
        adjustment = compute_adjustment(reading)
        row = {**reading, **adjustment}
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_dataset()
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "alloy_readings.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df.head())
    print("\nSummary stats:")
    print(df.describe())
