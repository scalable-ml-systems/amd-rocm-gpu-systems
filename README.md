# AMD ROCm GPU SYSTEMS
---

Hands-on investigations of GPU hardware, inference runtime behavior, and kernel execution on AMD MI300X using ROCm.

| Project | Description |
|---|---|
| [AMD-EXP-001](./amd-exp-001/) | Establishes hardware and inference baselines through device inventory, empirical roofline measurements, and controlled workloads. |
| [MI300X vLLM Mechanisms](./mi300x-vllm-mechanisms/) | Investigates how batching, chunked prefill, KV cache, attention, prefix caching, and speculative decoding affect inference behavior. |
| [MI300X MoE Kernel Profiling](./mi300x-moe-profiling/) | Profiles Qwen3 MoE inference with rocprofv3, connecting kernel launch patterns and hardware counters to the model’s matrix arithmetic. |

Each project folder contains its scripts and supporting results.
