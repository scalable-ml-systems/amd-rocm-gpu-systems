# Runbook: MI300X MoE Kernel Profiling

## 1. Analyze the saved captures first

From the repository root:

```bash
cd mi300x-moe-profiling
python3 scripts/analyze_results.py
```

Requirements: Python 3.9 or newer, the committed CSV files, and no additional packages. This works on a CPU-only computer. Paths resolve from the script location, so the script also works when invoked from another directory.

The report prints:

1. SHA256 fingerprints of its four input CSVs.
2. Top five kernel families by summed duration.
3. MoE variant counts, geometry, and mean trace durations.
4. Per-counter sample counts, averages, minima, and maxima.
5. Executed FLOPs, useful FLOPs, traffic estimates, arithmetic intensity, and estimated rates.
6. Capture verification results.

Expected results for the committed captures:

| Check | Expected |
|---|---:|
| All dispatches | 54,194 |
| Summed kernel duration | 332.566161 ms |
| MoE dispatches | 6,144 |
| MoE summed duration | 85.539433 ms |
| MoE share | 25.72% |
| IDs 9119 / 9120 | 48 launches each |
| IDs 9121 / 9122 | 3,024 launches each |
| Samples per counter, IDs 9119 / 9120 | 16 each |
| Samples per counter, IDs 9121 / 9122 | 1,008 each |

Exit status is 0 when capture checks pass, 1 when an expected result differs, and 2 for unreadable or invalid input. These checks establish agreement with the saved capture, not correctness of every interpretation.

Optional: save the printed report within the existing results directory:

```bash
python3 scripts/analyze_results.py > benchmark-results/mix300-device/analysis-report.md
```

The script itself only reads files and prints output. Shell redirection creates or replaces the named report.

## 2. GPU environment for a new capture

Use one MI300X with sufficient available memory for Qwen3-30B-A3B BF16, a working ROCm-compatible PyTorch/vLLM environment, rocprofv3, and `librocprofiler-sdk-roctx.so` available to the dynamic loader. The host needs model-download access or an existing model cache.

| Environment item | Recorded status |
|---|---|
| Hardware | See `docs/device-inventory.txt` and capture-specific `1_agent_info.csv` files |
| ROCm | 10.0, reported in experiment notes |
| vLLM | 0.27.0, reported in experiment notes |
| Exact rocprofiler SDK build | Not captured in this project's provenance |
| Exact PyTorch, Transformers, and dependency versions | Full package lock not captured |
| Container image digest / installation recipe | Not captured |
| Model and tokenizer revision | Not pinned in the workload script |
| Chat-formatted prompt token count | Not captured |
| Installed definitions of FETCH_SIZE / WRITE_SIZE | Not archived; needed to settle the memory unit conversion |

This is a procedure for repeating collection in a compatible environment, not a bit-for-bit environment reconstruction. Do not silently install newer packages and describe them as the original environment.

Before running, inspect your environment:

```bash
rocprofv3 --version
rocprofv3 --help
rocminfo
python3 -c 'import torch, vllm; print("torch:", torch.__version__); print("HIP:", torch.version.hip); print("vLLM:", vllm.__version__); print("GPU:", torch.cuda.get_device_name(0))'
python3 -c 'import ctypes; ctypes.CDLL("librocprofiler-sdk-roctx.so"); print("ROCTx library loaded")'
```

Confirm your profiler supports `--selected-regions`, YAML `pmc_groups`, and `pmc_group_interval`. Archive its version, package list, model revision, and metric definitions with a new capture. ROCm PyTorch uses the `torch.cuda` namespace for GPU operations; that name does not imply an NVIDIA GPU.

## 3. Collect the kernel trace

Run from `mi300x-moe-profiling/`. Choose a fresh output path for each new run; preserve the committed capture directories.

```bash
rocprofv3 \
  --kernel-trace \
  --stats \
  --output-format csv \
  --selected-regions \
  --output-directory benchmark-results/mix300-device/trace-rerun-01 \
  -- python3 scripts/mix300-device/vllm-kernel-64tok.py
```

The output-directory option is added here to make new outputs explicit. The historic shell invocation was not archived in this folder. Host/process subdirectories and CSV filename prefixes can differ by SDK version.

Expected application message: `GENERATED_TOKENS: 64`.

Expected profiler artifacts include a kernel trace, kernel statistics, domain statistics, and agent information. Verify that initialization and the warm-up are outside the selected measurement region. Changing the runtime or model revision can change dispatch counts, IDs, and kernel choices.

The script loads the model, formats a fixed prompt with thinking disabled, generates eight warm-up tokens, synchronizes, resumes profiling, generates exactly 64 output tokens, synchronizes again, and pauses profiling.

## 4. Collect counters in a separate run

```bash
rocprofv3 \
  --input scripts/mix300-device/fused-moe-counters.yaml \
  --output-format csv \
  --selected-regions \
  --output-directory benchmark-results/mix300-device/counters-rerun-01 \
  -- python3 scripts/mix300-device/vllm-kernel-64tok.py
```

Expected artifacts include counter collection and agent-information CSVs. The YAML selects `fused_moe_kernel` and rotates these groups:

```yaml
jobs:
  - pmc_groups:
      - ["SQ_INSTS_VALU_MFMA_MOPS_BF16"]
      - ["FETCH_SIZE"]
      - ["WRITE_SIZE"]
    pmc_group_interval: 8
    kernel_include_regex: "^fused_moe_kernel$"
```

The interval advances on per-device dispatch IDs; do not describe it as necessarily eight matching MoE launches. Counter groups measure different dispatches in the same application run. They are not three simultaneous measurements of one launch, nor three application replays.

The saved data has equal sample counts across groups. This does not establish representative sampling: periodic rotation can interact with periodic model execution. The report exposes minima and maxima, but does not test layer-by-layer sampling coverage.

## Calculation contract

### Timing and launch counts

- Dispatch duration = end timestamp − start timestamp, in nanoseconds.
- Mean microseconds = mean duration / 1,000.
- Family share = sum of family durations / sum of all kernel durations × 100.
- This sum is not end-to-end request latency; it does not account for overlap or gaps as a wall-clock timeline would.
- Trace grid size is the product of X, Y, and Z reported work-item dimensions. It is not a count of workgroups.
- The four IDs are capture-local. Cross-capture agreement of name and launch geometry is a consistency check, not proof of source-level operation identity.

### Executed matrix arithmetic

AMD defines `SQ_INSTS_VALU_MFMA_MOPS_BF16` in units of 512 floating-point operations:

```text
executed BF16 matrix FLOPs = mean counter value × 512
9121: 1,572,864 × 512 = 805,306,368 FLOPs
9122:   786,432 × 512 = 402,653,184 FLOPs
```

Do not multiply by two again. These counters measure matrix operation work, including work that may not contribute useful token results; they do not measure all inference arithmetic.

### Memory traffic and arithmetic intensity

```text
estimated bytes = (mean FETCH_SIZE + mean WRITE_SIZE) × bytes_per_unit
arithmetic intensity = executed BF16 matrix FLOPs / estimated bytes
```

The script defaults to **1,024 bytes per reported memory unit**, explicitly as a provisional convention. The installed metric definitions were not archived, so the CSV alone cannot settle the convention or all details of the traffic boundary. Do not publish these estimates as validated physical HBM traffic until those definitions are checked.

Compare the decimal convention without changing the source:

```bash
python3 scripts/analyze_results.py --memory-unit-bytes 1000
```

The difference affects bytes, arithmetic intensity, and bandwidth estimates; it does not affect the FLOP count or 16× arithmetic ratio. Read and write averages come from different sampled dispatches. Their sum is an estimate for the variant, not one launch's observed traffic.

### Model arithmetic and projection mapping

The official model configuration specifies 48 layers, hidden size 2,048, expert intermediate size 768, and eight selected experts out of 128. Gate and up together have width 2 × 768 = 1,536.

For one useful decode token across eight selected experts, counting a multiply-add as two FLOPs:

```text
gate/up = 8 × 1 × 2048 × 1536 × 2 = 50,331,648 FLOPs
down    = 8 × 1 ×  768 × 2048 × 2 = 25,165,824 FLOPs
```

Comparing ID 9121 to gate/up and ID 9122 to down yields 16× for each. Those assignments are inferred from execution order and relative work; the exported kernel name does not identify the projection. Confirm against the vLLM kernel implementation/configuration before presenting the assignments as proven. The model dimensions are an external interpretation input, not extracted from these CSVs.

### Duration-based rates

```text
estimated TFLOP/s = counter-derived FLOPs / mean trace nanoseconds / 1000
estimated GB/s   = counter-derived bytes / mean trace nanoseconds
```

These estimates combine separate runs. Counter collection may alter execution, and a ratio of averages is not an average of per-dispatch rates. Do not use these numbers alone to declare the kernel compute-bound or memory-bound.

## Scope and next experiment

The analysis script intentionally targets the committed captures and checks their expected counts. Use `--project-root` for another copy of this project with the same layout. It is not a generic classifier for new captures: new kernel IDs and shapes require inspection before changing the mapping or reference values.

The next experiment is to inspect tiling and vary useful tokens per expert while measuring the arithmetic ratio and latency. Increasing output length alone does not increase concurrent decode tokens per expert.

The offline analysis has been exercised against the committed files. The GPU collection commands have not been rerun as part of preparing this documentation.

## References

- [rocprofv3 tracing, selected regions, and counter multiplexing](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
- [AMD matrix operation counter definitions](https://rocm.docs.amd.com/en/docs-10.1.0/reference/gpu-arch/mi300-mi200-performance-counters.html)
- [Qwen3-30B-A3B configuration](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)
