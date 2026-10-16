from vllm import LLM, SamplingParams

model = "Qwen/Qwen3-30B-A3B"

document = (
    "A GPU serves inference by executing matrix operations, "
    "reading model weights and managing its KV cache. "
    "Communication and scheduling can also limit throughput.\n"
) * 180

llm = LLM(
    model=model,
    tensor_parallel_size=1,
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.85,
)

result = llm.chat(
    [{"role": "user",
      "content": f"Summarize the following technical document:\n{document}"}],
    sampling_params=SamplingParams(
        temperature=0,
        min_tokens=32,
        max_tokens=64,
    ),
    chat_template_kwargs={"enable_thinking": False},
    use_tqdm=False,
)[0]

print(
    f"PREFILL COMPLETE: "
    f"input={len(result.prompt_token_ids)} "
    f"output={len(result.outputs[0].token_ids)} "
    f"finish={result.outputs[0].finish_reason}",
    flush=True,
)
