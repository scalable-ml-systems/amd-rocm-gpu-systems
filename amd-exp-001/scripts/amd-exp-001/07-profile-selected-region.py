
#!/usr/bin/env python3
"""Capture one prefill-heavy request, excluding initialization."""

import ctypes

# Fail before loading the model if ROCTx is unavailable.
ROCTX_LIB = (
    "/opt/python/lib/python3.14/site-packages/"
    "_rocm_sdk_devel/lib/librocprofiler-sdk-roctx.so"
)

roctx = ctypes.CDLL(ROCTX_LIB)
for name in ("roctxProfilerResume", "roctxProfilerPause"):
    function = getattr(roctx, name)
    function.argtypes = [ctypes.c_uint64]
    function.restype = ctypes.c_int

import torch
import sys
from vllm import LLM, SamplingParams

document = (
    "A GPU serves inference by executing matrix operations, "
    "reading model weights and managing its KV cache. "
    "Communication and scheduling can also limit throughput.\n"
) * 180

workload = sys.argv[1] if len(sys.argv) > 1 else "prefill"

if workload == "prefill":
    messages = [{
        "role": "user",
        "content": f"Summarize the following technical document:\n{document}",
    }]
    min_tokens, max_tokens = 32, 64

elif workload == "decode":
    messages = [{
        "role": "user",
        "content": (
            "Explain GPU memory bandwidth, KV-cache management, "
            "continuous batching and tensor-parallel communication "
            "in a detailed technical discussion."
        ),
    }]
    min_tokens, max_tokens = 480, 512

else:
    raise ValueError(f"Unknown workload: {workload}")


# After selecting the workload, before model initialization:

if workload == "decode":
    assert len(messages[0]["content"]) < 500, (
        "Decode workload unexpectedly contains a long prompt"
    )

print(
    f"Selected workload: {workload}, "
    f"max output tokens: {max_tokens}, "
    f"prompt: {messages[0]['content'][:100]!r}",
    flush=True,
)

llm = LLM(
    model="Qwen/Qwen3-30B-A3B",
    tensor_parallel_size=1,
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.85,
    enable_prefix_caching=False,
)

# Warm up the same prompt shape without collecting its kernels.
llm.chat(
    messages,
    sampling_params=SamplingParams(
        temperature=0,
        min_tokens=4,
        max_tokens=4,
    ),
    chat_template_kwargs={"enable_thinking": False},
    use_tqdm=False,
)
torch.cuda.synchronize()

print("START SELECTED PREFILL-HEAVY REGION", flush=True)

if roctx.roctxProfilerResume(0) != 0:
    raise RuntimeError("ROCTx profiler resume failed")

try:
    result = llm.chat(
        messages,
        sampling_params=SamplingParams(
            temperature=0,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        ),
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=False,
    )[0]

    torch.cuda.synchronize()
finally:
    if roctx.roctxProfilerPause(0) != 0:
        raise RuntimeError("ROCTx profiler pause failed")

print(
    "PROFILE COMPLETE: "
    f"input={len(result.prompt_token_ids)} "
    f"output={len(result.outputs[0].token_ids)}",
    flush=True,
)

