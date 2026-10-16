
# AMD-EXP-001 — Experimental Evidence

**Study:** From Roofline to Serving SLO  
**Hardware:** AMD MI300X, SPX/NPS1  
**ROCm:** 10.0.0  
**Model:** Qwen3-30B-A3B, BF16

This directory contains the measurements and raw evidence from our
cross-layer investigation of MI300X inference performance.

## 1. Hardware roofline

`roofline/`

Empirical ceilings measured using ROCm Compute Profiler 3.8.0:

| Metric | Measured |
|---|---:|
| HBM bandwidth | 4.204 TB/s |
| BF16 MFMA | 1.029 PFLOP/s |
| FP8 MFMA | 1.867 PFLOP/s |

Source: `roofline/roofline.csv`

These are microbenchmark ceilings, not achieved vLLM performance.

## 2. TP1 workload validation

`tp1/`

Both workloads execute successfully on one MI300X.

| Workload | Input tokens | Output tokens |
|---|---:|---:|
| Prefill-heavy | 4,700 | 64 |
| Decode-heavy | 35 | 512 |

The Docker image digest and smoke-test response are retained here.

## 3. Kernel inventory

`tp1/selected-prefill/`

Selected-region trace of the 4,700-input-token request,
excluding model initialization and warmup.

Evidence includes:
- Kernel dispatch trace
- Kernel execution statistics
- GPU agent information
- Profiler execution log

**Initial observation:** `fused_moe_kernel` accounts for 29.53%
of recorded GPU kernel execution time.

This request also generates 64 output tokens. Its aggregate
statistics must not be interpreted as prefill-only measurements.

## 4. Next measurements

- Decode-heavy selected-region trace
- Prefill/decode kernel comparison
- Per-kernel roofline analysis
- Multi-GPU fabric and serving measurements

## Reproduction

Experiment scripts are maintained in:

`../../scripts/amd-exp-001/`

Raw evidence is copied from the AMD VM's scratch storage.
Large traces may be archived separately with checksums.

