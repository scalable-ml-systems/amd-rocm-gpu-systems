
#!/usr/bin/env bash
set -euo pipefail

IMAGE="rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0"
MODEL="Qwen/Qwen3-30B-A3B"
SCRATCH="/mnt/amd-scratch"
NAME="amd001-tp1"

mountpoint -q "$SCRATCH" || {
    echo "ERROR: Scratch disk not mounted."
    exit 1
}

docker info >/dev/null 2>&1 || {
    echo "ERROR: Docker is unavailable."
    exit 1
}

if docker container inspect "$NAME" >/dev/null 2>&1; then
    echo "ERROR: Container already exists: $NAME"
    exit 1
fi

mkdir -p \
    "$SCRATCH/hf-cache" \
    "$SCRATCH/amd-exp-001/tp1"

echo "===== PULL PINNED IMAGE ====="
docker pull "$IMAGE"

echo "===== RECORD IMAGE DIGEST ====="
docker image inspect "$IMAGE" \
    --format '{{json .RepoDigests}}' \
    > "$SCRATCH/amd-exp-001/tp1/image-digest.json"

echo "===== START TP1 ====="
docker run -d \
    --name "$NAME" \
    --device /dev/kfd \
    --device /dev/dri \
    --network=host \
    --ipc=host \
    --group-add=video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v "$SCRATCH/hf-cache:/app/models" \
    -e HF_HOME=/app/models \
    -e ROCP_TOOL_ATTACH=1 \
    "$IMAGE" \
    vllm serve "$MODEL" \
        --host 127.0.0.1 \
        --port 8000 \
        --tensor-parallel-size 1 \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.85 \
        --default-chat-template-kwargs '{"enable_thinking":false}'

echo "===== CONTAINER ====="
docker ps --filter "name=$NAME"

echo
echo "Inspect startup with:"
echo "docker logs --tail 50 -f $NAME"
