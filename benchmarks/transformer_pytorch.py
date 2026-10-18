import json
import statistics
import time

import torch


torch.set_num_threads(1)
torch.manual_seed(23)
BATCH = 4
SEQUENCE = 256
DIMENSION = 512
HEADS = 8
WARMUP = 5
ITERATIONS = 20

model = torch.nn.MultiheadAttention(
    DIMENSION, HEADS, dropout=0.0, batch_first=True
).eval()
x = torch.randn(BATCH, SEQUENCE, DIMENSION)

with torch.inference_mode():
    for _ in range(WARMUP):
        model(x, x, x, need_weights=False)

    samples = []
    for _ in range(ITERATIONS):
        started = time.perf_counter_ns()
        model(x, x, x, need_weights=False)
        samples.append((time.perf_counter_ns() - started) / 1_000_000)

print(json.dumps({"runtime": "pytorch", "median_ms": statistics.median(samples)}))
