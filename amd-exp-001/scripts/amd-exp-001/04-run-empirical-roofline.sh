#!/usr/bin/env bash
set -euo pipefail
export ROCM_VER=10.0.0

PROFILER=/opt/rocm/core-10.0/bin/rocprof-compute
SCRATCH=/mnt/amd-scratch
RUN_DIR="$SCRATCH/amd-exp-001/roofline"

# Do not accidentally write large results to the boot disk.
mountpoint -q "$SCRATCH" || {
    echo "ERROR: Mount the 5 TB scratch disk first."
    exit 1
}

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

echo "===== PROFILER VERSION ====="
"$PROFILER" --version | tee profiler-version.txt

echo "===== EMPIRICAL ROOFLINE ====="
"$PROFILER" profile \
    --name MI300X-SPX-NPS1-BASELINE \
    --bench-only \
    --device 0 \
    2>&1 | tee roofline-run.log

echo "===== EVIDENCE ====="
find "$RUN_DIR" -name roofline.csv -print
