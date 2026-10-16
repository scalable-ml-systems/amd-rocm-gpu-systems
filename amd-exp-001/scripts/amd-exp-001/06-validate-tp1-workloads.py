
#!/usr/bin/env python3
"""Validate two TP1 workload shapes before profiling."""

import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3-30B-A3B"
OUTPUT = Path("/mnt/amd-scratch/amd-exp-001/tp1/workloads")
OUTPUT.mkdir(parents=True, exist_ok=True)

prefill_text = (
    "A GPU serves inference by executing matrix operations, "
    "reading model weights and managing its KV cache. "
    "Communication and scheduling can also limit throughput.\n"
) * 180

workloads = {
    "prefill": {
        "prompt": f"Summarize the following technical document:\n{prefill_text}",
        "min_tokens": 32,
        "max_tokens": 64,
    },
    "decode": {
        "prompt": (
            "Explain GPU memory bandwidth, KV-cache management, "
            "continuous batching and tensor-parallel communication "
            "in a detailed technical discussion."
        ),
        "min_tokens": 480,
        "max_tokens": 512,
    },
}

for name, config in workloads.items():
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": config["prompt"]}],
        "temperature": 0,
        "min_tokens": config["min_tokens"],
        "max_tokens": config["max_tokens"],
    }

    request = Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    with urlopen(request, timeout=300) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - start

    (OUTPUT / f"{name}-response.json").write_text(
        json.dumps(result, indent=2)
    )

    usage = result["usage"]
    print(
        f"{name}: "
        f"input={usage['prompt_tokens']} "
        f"output={usage['completion_tokens']} "
        f"finish={result['choices'][0]['finish_reason']} "
        f"wall_time={elapsed:.2f}s"
    )

