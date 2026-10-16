
# AMD-EXP-001 — Run Ledger

## Frozen environment

- Hardware: AMD Instinct MI300X VF
- GPU count: 1
- Architecture: gfx942
- Compute units: 304
- Partition: SPX / NPS1
- HBM: ~192 GiB
- OS: Ubuntu 24.04.4
- Kernel: 6.8.0-138-generic
- ROCm: 10.0.0
- Profiler package: amdrocm-profiler10.0 (10.0.0-4)
- Profiler path: /opt/rocm/core-10.0/bin/rocprof-compute

---

## SETUP-001 — Profiler installation

Date: 2026-09-22
Purpose: Prepare the TP1 kernel investigation.

Preflight:
- Package candidate: 10.0.0-4
- Upgrades: 0
- New packages: 1
- Removals: 0

Result: PASS

ROCm Compute Profiler installed and located at:
  /opt/rocm/core-10.0/bin/rocprof-compute

Next: Validate profiler execution and establish the
empirical MI300X roofline.

---
### AMD001-S1-ROOFLINE-R1

Result: PASS
GPU: MI300X VF, SPX/NPS1
ROCm: 10.0.0
Profiler: 3.8.0

Empirical ceilings:
- HBM: 4,204.12 GB/s
- BF16 MFMA: 1,028,779.69 GFLOP/s
- FP8 MFMA: 1,867,019.53 GFLOP/s

Issue: Missing ROCm version metadata.
Resolution: ROCM_VER=10.0.0

Evidence: MI300X_A1/roofline.csv

---

### FAILURE 
E0923 01:12:46.171981     373 rocattach.cpp:358] [rocprofiler-sdk-rocattach] Cannot attach to process 185: 'rocp-bg-attach' thread not found. The target process does not appear to have attach support enabled. Start the target with ROCP_TOOL_ATTACH=1, or use a rocprofiler-register build configured with ROCPROFILER_REGISTER_BUILD_DEFAULT_ATTACHMENT=ON.
E0923 01:12:46.171984     373 rocattach.cpp:576] [rocprofiler-sdk-rocattach] rocattach_attach_tree failed for pid 185 with error code 1, continuing with remaining processes

Error : The ROCm profiler cannot attach to EngineCore because the required rocp-bg-attach thread is absent.
Even though ROCP_TOOL_ATTACH=1 was set when the container started, that setting hasn't enabled attachment in the process executing our model. This is a profiler-initialization problem, not a vLLM inference failure.

Solution : we'll use a dedicated profiling process and launch the workload under rocprofv3 from the beginning.
