#!/usr/bin/env bash
set -euo pipefail

echo "===== AMD-EXP-001 PREFLIGHT ====="
date -u

echo
echo "===== REQUIRED COMMANDS ====="

required_commands=(
  amd-smi
  rocminfo
  rocprofv3
  lspci
  numactl
  lsblk
)

failed=0

for cmd in "${required_commands[@]}"; do
    if command -v "$cmd" >/dev/null 2>&1; then
        printf "%-20s PASS  %s\n" "$cmd" "$(command -v "$cmd")"
    else
        printf "%-20s FAIL\n" "$cmd"
        failed=1
    fi
done

echo
echo "===== GPU VISIBILITY ====="

amd-smi static --asic --vram --driver

if [[ "$failed" -ne 0 ]]; then
    echo
    echo "PREFLIGHT FAILED"
    exit 1
fi

echo
echo "PREFLIGHT PASSED"
