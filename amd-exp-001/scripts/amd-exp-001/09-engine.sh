
#!/usr/bin/env bash
set -euo pipefail

ENGINE="${2:-}"
CACHE_MODE="${3:-off}"
CACHE_TOKENS="${4:-}"

REPO="/root/amd-rocm-platform-engineering"
MODEL_CACHE="/root/amd-model-cache"
MODEL_ID="Qwen/Qwen3-30B-A3B"
MODEL_DIR="$MODEL_CACHE/hub/models--Qwen--Qwen3-30B-A3B/snapshots"

VLLM_IMAGE="rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0"
SGLANG_IMAGE="rocm/sgl-dev:v0.5.15.post1-ubuntu24.04-py3.14-rocm10.0.0"

CONTAINER_PREFIX="amd001"
PORT=8000

usage() {
    echo "Usage:"
    echo "  $0 start vllm|sglang on|off"
    echo "  $0 stop"
    echo "  $0 status"
    exit 1
}

running_engine() {
    docker ps --format '{{.Names}}' |
        grep -E "^${CONTAINER_PREFIX}-(vllm|sglang)$" || true
}

case "${1:-}" in

    status)
        echo "===== INFERENCE ENGINES ====="
        docker ps -a \
            --filter "name=${CONTAINER_PREFIX}-" \
            --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
        exit 0
        ;;

    stop)
        for engine in vllm sglang; do
            name="${CONTAINER_PREFIX}-${engine}"

            if docker container inspect "$name" >/dev/null 2>&1; then
                docker logs "$name" > "/root/${name}-last.log" 2>&1
                docker stop "$name" || true
                docker rm "$name"
                echo "Stopped $name; saved its log."
            fi
        done
        exit 0
        ;;

    start)
        [[ "$ENGINE" == "vllm" || "$ENGINE" == "sglang" ]] || usage
        [[ "$CACHE_MODE" == "on" || "$CACHE_MODE" == "off" ]] || usage
        ;;

    *)
        usage
        ;;
esac

# Check hardware and software before starting a paid experiment.

[[ -e /dev/kfd ]] || {
    echo "ERROR: AMD GPU device is unavailable."
    exit 1
}

docker info >/dev/null

[[ -d "$MODEL_DIR" ]] || {
    echo "ERROR: Model cache is missing."
    exit 1
}

# Use the exact cached checkpoint. Never silently download another revision.

mapfile -t SNAPSHOTS < <(
    find "$MODEL_DIR" -mindepth 1 -maxdepth 1 -type d | sort
)

[[ "${#SNAPSHOTS[@]}" -eq 1 ]] || {
    echo "ERROR: Expected exactly one cached model revision."
    printf '%s\n' "${SNAPSHOTS[@]}"
    exit 1
}

MODEL_REVISION="$(basename "${SNAPSHOTS[0]}")"
CONTAINER_MODEL="/app/models/hub/models--Qwen--Qwen3-30B-A3B/snapshots/$MODEL_REVISION"

# Refuse to compete with another experiment on the GPU.

if [[ -n "$(running_engine)" ]]; then
    echo "ERROR: Another inference engine is already running."
    running_engine
    exit 1
fi

NAME="${CONTAINER_PREFIX}-${ENGINE}"

if docker container inspect "$NAME" >/dev/null 2>&1; then
    echo "ERROR: Container $NAME already exists."
    echo "Run '$0 stop' before starting another experiment."
    exit 1
fi

if ss -ltn '( sport = :8000 )' | grep -q LISTEN; then
    echo "ERROR: Port 8000 is already in use."
    exit 1
fi

# Record every startup configuration.

SESSION="$REPO/benchmark-results/amd-exp-001/serving/sessions/$(date -u +%Y%m%dT%H%M%SZ)-${ENGINE}-cache-${CACHE_MODE}"
mkdir -p "$SESSION"

COMMON_DOCKER=(
    --detach
    --name "$NAME"
    --device /dev/kfd
    --device /dev/dri
    --network host
    --ipc host
    --group-add video
    --env HF_HOME=/app/models
    --env HF_HUB_OFFLINE=1
    --volume "$MODEL_CACHE:/app/models:ro"
)

if [[ "$ENGINE" == "vllm" ]]; then
    IMAGE="$VLLM_IMAGE"

    CACHE_FLAGS=(--no-enable-prefix-caching)
    if [[ "$CACHE_MODE" == "on" ]]; then
        CACHE_FLAGS=(--enable-prefix-caching)
    fi

    KV_FLAGS=()

    if [[ -n "$CACHE_TOKENS" ]]; then
        [[ "$CACHE_TOKENS" =~ ^[1-9][0-9]*$ ]] || {
            echo "ERROR: Cache capacity must be a positive integer."
            exit 1
        }

        # Qwen3-30B-A3B: 98,304 bytes per token with BF16 KV cache.
        KV_BYTES=$((CACHE_TOKENS * 98304))

        KV_FLAGS=(
            --kv-cache-memory-bytes "$KV_BYTES"
            --kv-cache-dtype auto
        )
    fi

    COMMAND=(
        --entrypoint vllm
        "$IMAGE"
        serve "$CONTAINER_MODEL"
        --served-model-name "$MODEL_ID"
        --host 127.0.0.1
        --port "$PORT"
        --tensor-parallel-size 1
        --dtype bfloat16
        --max-model-len 8192
        --gpu-memory-utilization 0.80
        --default-chat-template-kwargs '{"enable_thinking":false}'
        "${CACHE_FLAGS[@]}"
	"${KV_FLAGS[@]}"
    )

else
    IMAGE="$SGLANG_IMAGE"

    CACHE_FLAGS=()
    if [[ "$CACHE_MODE" == "off" ]]; then
        CACHE_FLAGS=(--disable-radix-cache)
    fi
    
    COMMON_DOCKER+=(--env GPU_ARCHS=gfx942)

    CACHE_LIMIT_FLAGS=()

    if [[ -n "$CACHE_TOKENS" ]]; then
        if ! [[ "$CACHE_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
            echo "ERROR: Cache capacity must be a positive integer."
            exit 1
        fi

        CACHE_LIMIT_FLAGS=(
            --max-total-tokens "$CACHE_TOKENS"
        )
    fi

    COMMAND=(
        --entrypoint python3
        "$IMAGE"
        -m sglang.launch_server
        --model-path "$CONTAINER_MODEL"
        --host 127.0.0.1
        --port "$PORT"
        --tp 1
        --dtype bfloat16
        --context-length 8192
        --mem-fraction-static 0.80
        --enable-metrics
        "${CACHE_FLAGS[@]}"
	"${CACHE_LIMIT_FLAGS[@]}"
    )
fi

docker image inspect "$IMAGE" >/dev/null || {
    echo "ERROR: Docker image is missing: $IMAGE"
    exit 1
}

{
    echo "engine=$ENGINE"
    echo "cache_mode=$CACHE_MODE"
    echo "model=$MODEL_ID"
    echo "model_revision=$MODEL_REVISION"
    echo "image=$IMAGE"
    echo "image_id=$(docker image inspect "$IMAGE" --format '{{.Id}}')"
    echo "max_context_tokens=8192"
    echo "gpu_memory_setting=0.80"
    echo "started_utc=$(date -u +%FT%TZ)"
    printf 'command='
    printf '%q ' "${COMMAND[@]}"
    echo
} > "$SESSION/startup.txt"

docker run "${COMMON_DOCKER[@]}" "${COMMAND[@]}"

echo "Started $ENGINE (cache: $CACHE_MODE)"
echo "requested_cache_tokens=${CACHE_TOKENS:-automatic}"
echo "Session: $SESSION"
echo "API: http://127.0.0.1:$PORT"
echo "Run 10-smoke-test.py before sending workloads."
