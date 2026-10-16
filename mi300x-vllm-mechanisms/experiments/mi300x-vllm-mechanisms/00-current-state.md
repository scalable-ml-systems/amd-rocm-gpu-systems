# MI300X vLLM Runtime Mechanisms

## Completed foundation experiment

Model:
- Qwen/Qwen3-30B-A3B
- BF16
- vLLM 0.27.0
- AMD MI300X

Completed:
- real 64-token decode trace
- fused_moe_kernel identified as dominant kernel
- launch geometry captured
- hardware counters captured
- HBM arithmetic intensity calculated
- Roofline placement calculated
- two steady-state MoE GEMM stages identified

Primary evidence:
- benchmark-results/mix300-device/vllm-kernel-64tok/
- benchmark-results/mix300-device/vllm-kernel-64tok-counters/

Next phase:
1. Continuous batching
2. Chunked prefill
3. KV cache
4. PagedAttention
5. Prefix cache
6. Prefill/Flash attention
7. Speculative decoding

Rule:
Plan + code + dry-run locally before MI300X execution.
