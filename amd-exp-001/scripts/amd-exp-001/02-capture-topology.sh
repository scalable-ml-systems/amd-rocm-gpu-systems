#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-./evidence-topology}"

mkdir -p "$OUTPUT_DIR"

echo "Capturing topology evidence to: $OUTPUT_DIR"

amd-smi partition \
  > "$OUTPUT_DIR/partition.txt"

amd-smi topology \
  > "$OUTPUT_DIR/amd-topology.txt"

numactl -H \
  > "$OUTPUT_DIR/numa.txt"

lspci -tv \
  > "$OUTPUT_DIR/pci-tree.txt"

lspci -nn \
  > "$OUTPUT_DIR/pci-devices.txt"

lspci -nn | egrep -i 'AMD|ATI|Ethernet|Network|Infiniband' \
  > "$OUTPUT_DIR/gpu-network-devices.txt" || true

ls -la /sys/class/infiniband \
  > "$OUTPUT_DIR/infiniband-devices.txt" 2>&1 || true

rdma link \
  > "$OUTPUT_DIR/rdma-links.txt" 2>&1 || true

echo "Topology capture complete."
