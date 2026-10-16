import ctypes
import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-30B-A3B"

# ---- 1. Load the real Qwen3 model through vLLM ----
llm = LLM(
    model=MODEL,
    dtype="bfloat16",
    max_model_len=8192,
)

tokenizer = llm.get_tokenizer()

messages = [{
    "role": "user",
    "content": (
        "Explain how a GPU executes parallel numerical work. "
        "Discuss threads, workgroups, memory access, and scheduling."
    ),
}]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

# ---- 2. Warm-up: excluded from profiler ----
warmup = SamplingParams(
    temperature=0,
    max_tokens=8,
    min_tokens=8,
    ignore_eos=True,
)
llm.generate([prompt], warmup)
torch.cuda.synchronize()

# ---- 3. Exact measured workload: 64 decode tokens ----
measured = SamplingParams(
    temperature=0,
    max_tokens=64,
    min_tokens=64,
    ignore_eos=True,
)

roctx = ctypes.CDLL("librocprofiler-sdk-roctx.so")
roctx.roctxProfilerResume(0)

outputs = llm.generate([prompt], measured)

# ---- 4. Wait until GPU work is finished, then stop profiler ----
torch.cuda.synchronize()
roctx.roctxProfilerPause(0)

print("GENERATED_TOKENS:", len(outputs[0].outputs[0].token_ids))
