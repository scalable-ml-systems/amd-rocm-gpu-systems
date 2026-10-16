# MI300X MoE Kernel Profiling

How does one Qwen3 inference request become thousands of GPU launches, and how much matrix arithmetic do its MoE kernels execute?

This project follows a controlled request from kernel tracing to hardware counters and model mathematics. It uses **Qwen3-30B-A3B on one AMD MI300X**, with vLLM and rocprofv3. The committed captures support the measurements below; the analysis script makes the calculations repeatable without a GPU.

## Workload

| Item | Configuration |
|---|---|
| GPU | One AMD MI300X; gfx942; SPX/NPS1 |
| Software | ROCm 10.0 and vLLM 0.27.0, as recorded in experiment notes; full environment lock not captured |
| Model | `Qwen/Qwen3-30B-A3B`, BF16 |
| Input | One fixed chat-formatted prompt; thinking disabled |
| Batch size | One request |
| Output | Exactly 64 generated tokens |
| Execution | Offline `LLM.generate()`; temperature 0 |
| Warm-up | Eight generated tokens before measurement |
| Measurement boundary | ROCTx profiler resume/pause; GPU synchronization before and after the measured generation |

The 64 output tokens are generated sequentially. They are not a batch of 64 decode tokens. The workload script's comment “64 decode tokens” refers to output length; the launch pattern is consistent with the initial prediction followed by 63 decode steps.

## Results

| Observation | Result | Evidence |
|---|---|---|
| Total GPU dispatches | 54,194 | [Kernel trace](benchmark-results/mix300-device/vllm-kernel-64tok/5ab03fd80643/1_kernel_trace.csv) |
| `fused_moe_kernel` | 6,144 launches; 85.539 ms; 25.72% of summed kernel duration | [Kernel statistics](benchmark-results/mix300-device/vllm-kernel-64tok/5ab03fd80643/1_kernel_stats.csv) |
| Four MoE variants | 48, 48, 3,024, and 3,024 launches | [Analysis script](scripts/analyze_results.py) |
| Decode counter sampling | 1,008 samples per counter for each repeated variant | [Counter collection](benchmark-results/mix300-device/vllm-kernel-64tok-counters/1e0279077382/1_counter_collection.csv) |
| Executed versus useful matrix arithmetic | Approximately 16× for both inferred decode projections | [Calculation and assumptions](RUNBOOK.md#calculation-contract) |

The launch-count interpretation is **48 layers × 2 launches + 63 decode steps × 48 layers × 2 launches = 6,144**. It connects the repeated trace structure to model execution.

The arithmetic comparison asks whether tile padding or underfilled matrix work explains the gap. That mechanism remains a hypothesis. The 16× ratio does not establish a 16× latency penalty or possible speedup. Kernel IDs are local to these captures; gate/up and down assignments remain inferred until confirmed against the implementation.

## Reproduce the analysis

From this project folder, using Python 3.9 or newer:

```bash
python3 scripts/analyze_results.py
```

No GPU or third-party Python packages are required. The script reads the committed CSVs, prints Markdown tables and SHA256 fingerprints, and checks trace totals, statistics, variant counts, geometry, and sample counts. It does not change the raw results.

For GPU collection commands, environment requirements, and interpretation limits, see [RUNBOOK.md](RUNBOOK.md).

## Files

| File or directory | Role |
|---|---|
| [Workload](scripts/mix300-device/vllm-kernel-64tok.py) | Fixed prompt, model settings, warm-up, and profiling boundaries |
| [Counter configuration](scripts/mix300-device/fused-moe-counters.yaml) | Three counter groups, interval eight, MoE kernel filter |
| [Trace capture](benchmark-results/mix300-device/vllm-kernel-64tok/5ab03fd80643/) | Dispatch trace, kernel statistics, domain statistics, and agent information |
| [Counter capture](benchmark-results/mix300-device/vllm-kernel-64tok-counters/1e0279077382/) | Counter measurements and agent information |
| [Device inventory](docs/device-inventory.txt) | Hardware record |
| [Analysis](scripts/analyze_results.py) | Reproduces tables and calculations |

The extensionless `count` file is an existing excerpt of counter rows. The analysis uses the complete `1_counter_collection.csv`, not that excerpt.

## Article series

| Part | Question | Supporting material |
|---|---|---|
| 1. Reading the launch pattern | Why did the MoE family run 6,144 times? | Trace, statistics, analysis sections 1–2 |
| 2. Choosing and grouping counters | How are incompatible counter groups sampled? | Counter YAML and analysis section 3 |
| 3. From counters to FLOPs and bytes | What do the raw values mean? | Analysis section 4 and runbook conversion assumptions |
| 4. Comparing hardware work to MoE math | How much arithmetic did the useful projections require? | Model dimensions and executed-to-useful comparison |

Published article URLs will be added when available.

## References

- [AMD rocprofv3 tracing and counter collection](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
- [AMD MI300/MI200 counter definitions](https://rocm.docs.amd.com/en/docs-10.1.0/reference/gpu-arch/mi300-mi200-performance-counters.html)
- [Official Qwen3-30B-A3B model configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)

The reference documentation may describe a newer SDK than the capture environment. The runbook distinguishes saved measurements from environment details that still need to be recovered.
