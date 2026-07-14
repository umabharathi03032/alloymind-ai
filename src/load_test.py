"""
load_test.py
------------
Fires repeated prediction requests at the running API and measures real
latency (average and 95th percentile), so any latency claim is backed by
an actual measurement rather than an assumption.
"""
import time
import random
import numpy as np
import requests

URL = "http://localhost:8000/predict"
N_REQUESTS = 300

ELEMENTS = ["C", "Si", "Mn", "P", "S", "Cr", "Ni"]
BASELINE = {"C": 0.40, "Si": 1.00, "Mn": 0.80, "P": 0.03, "S": 0.02, "Cr": 1.20, "Ni": 0.50}


def random_reading():
    return {el: max(0.001, v + random.gauss(0, v * 0.2)) for el, v in BASELINE.items()}


def main():
    latencies = []
    start_all = time.perf_counter()
    for _ in range(N_REQUESTS):
        payload = random_reading()
        t0 = time.perf_counter()
        resp = requests.post(URL, json=payload)
        t1 = time.perf_counter()
        resp.raise_for_status()
        latencies.append((t1 - t0) * 1000)  # ms
    total_time = time.perf_counter() - start_all

    latencies = np.array(latencies)
    print(f"Requests sent: {N_REQUESTS}")
    print(f"Total wall time: {total_time:.2f}s  (~{N_REQUESTS/total_time:.1f} req/sec sustained)")
    print(f"Average latency: {latencies.mean():.3f} ms")
    print(f"Median latency:  {np.median(latencies):.3f} ms")
    print(f"95th percentile: {np.percentile(latencies, 95):.3f} ms")
    print(f"Max latency:     {latencies.max():.3f} ms")


if __name__ == "__main__":
    main()
