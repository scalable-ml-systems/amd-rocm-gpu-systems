#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-./evidence-system}"

mkdir -p "$OUTPUT_DIR"

echo "Capturing system evidence to: $OUTPUT_DIR"

cat /etc/os-release \
  > "$OUTPUT_DIR/os-release.txt"

uname -a \
  > "$OUTPUT_DIR/kernel.txt"

rocminfo \
  > "$OUTPUT_DIR/rocminfo.txt"

amd-smi version \
  > "$OUTPUT_DIR/amd-smi-version.txt"

amd-smi static --asic --vram --driver \
  > "$OUTPUT_DIR/amd-smi-static.txt"

lscpu \
  > "$OUTPUT_DIR/lscpu.txt"

free -h \
  > "$OUTPUT_DIR/memory.txt"

lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINTS,MODEL \
  > "$OUTPUT_DIR/storage.txt"

df -hT \
  > "$OUTPUT_DIR/filesystems.txt"

echo "System capture complete."
