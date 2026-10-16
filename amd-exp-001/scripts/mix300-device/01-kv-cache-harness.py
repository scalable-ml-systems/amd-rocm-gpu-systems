#!/usr/bin/env python3
"""MIX300 / Experiment 01: verify the installed vLLM reshape_and_cache GPU op.

The two-token baseline is deliberately a correctness experiment, not a
throughput benchmark. Requires an AMD GPU and the ROCm vLLM Python environment
for --run. --predict and --self-test use only Python's standard library.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path

NUM_TOKENS = 2
NUM_HEADS = 2
HEAD_SIZE = 128
BLOCK_SIZE = 16
NUM_CACHE_BLOCKS = 2
X = 8  # 16 bytes / 2 bytes per BF16 element, per inspected vLLM source
SENTINEL = -1000
CASES = {
    "baseline": [3, 19],
    "relocation": [19, 3],
    "invalid": [3, -1],
}


def destination(slot):
    """Convert a global slot number to (cache block, position)."""
    return None if slot < 0 else divmod(slot, BLOCK_SIZE)


def describe(mapping):
    for token, slot in enumerate(mapping):
        dest = destination(slot)
        print(f"  token {token}: slot={slot} -> " +
              ("skip" if dest is None else f"cache block {dest[0]}, position {dest[1]}"))


def self_test():
    assert destination(3) == (0, 3)
    assert destination(19) == (1, 3)
    assert destination(-1) is None
    assert NUM_HEADS * (HEAD_SIZE // X) == 32
    print("PASS: offline address and launch-configuration checks")


def gpu_run(case, json_out=None):
    try:
        import torch
        from vllm import _custom_ops as ops
    except ImportError as exc:
        raise SystemExit(f"Requires the ROCm vLLM container: {exc}")

    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible to PyTorch; check ROCm container/device mapping")

    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_version = "unavailable"

    if not hasattr(torch.ops._C_cache_ops, "reshape_and_cache"):
        raise SystemExit("Installed build lacks _C_cache_ops.reshape_and_cache; stop and inspect build")

    print(f"vLLM: {vllm_version}; PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"HIP runtime version: {getattr(torch.version, 'hip', None)}")
    print(f"Case: {case}; mapping: {CASES[case]}")
    describe(CASES[case])

    # All values are exactly representable in BF16, avoiding tolerance ambiguity.
    key_cpu = torch.empty((NUM_TOKENS, NUM_HEADS, HEAD_SIZE), dtype=torch.bfloat16)
    value_cpu = torch.empty_like(key_cpu)
    element = torch.arange(HEAD_SIZE, dtype=torch.float32)
    for token in range(NUM_TOKENS):
        for head in range(NUM_HEADS):
            base = token * 40 + head * 20
            key_cpu[token, head] = (element + base).to(torch.bfloat16)
            value_cpu[token, head] = (-(element + 1 + base)).to(torch.bfloat16)

    key_shape = (NUM_CACHE_BLOCKS, NUM_HEADS, HEAD_SIZE // X, BLOCK_SIZE, X)
    value_shape = (NUM_CACHE_BLOCKS, NUM_HEADS, HEAD_SIZE, BLOCK_SIZE)
    expected_key = torch.full(key_shape, SENTINEL, dtype=torch.bfloat16)
    expected_value = torch.full(value_shape, SENTINEL, dtype=torch.bfloat16)
    for token, slot in enumerate(CASES[case]):
        dest = destination(slot)
        if dest is None:
            continue
        block, offset = dest
        for head in range(NUM_HEADS):
            expected_key[block, head, :, offset, :] = key_cpu[token, head].reshape(HEAD_SIZE // X, X)
            expected_value[block, head, :, offset] = value_cpu[token, head]

    device = torch.device("cuda:0")  # ROCm PyTorch retains the 'cuda' API name.
    key = key_cpu.to(device)
    value = value_cpu.to(device)
    key_cache = torch.full(key_shape, SENTINEL, dtype=torch.bfloat16, device=device)
    value_cache = torch.full(value_shape, SENTINEL, dtype=torch.bfloat16, device=device)
    mapping = torch.tensor(CASES[case], dtype=torch.int64, device=device)
    k_scale = torch.ones(1, dtype=torch.float32, device=device)
    v_scale = torch.ones(1, dtype=torch.float32, device=device)

    # Calls the installed production extension; never substitute our HIP kernel.
    ops.reshape_and_cache(key, value, key_cache, value_cache,
                          mapping, "auto", k_scale, v_scale)
    torch.cuda.synchronize()

    observed_key = key_cache.cpu()
    observed_value = value_cache.cpu()
    key_ok = torch.equal(observed_key, expected_key)
    value_ok = torch.equal(observed_value, expected_value)
    print(f"key cache:   {'PASS' if key_ok else 'FAIL'}")
    print(f"value cache: {'PASS' if value_ok else 'FAIL'}")
    if not (key_ok and value_ok):
        for label, actual, expected in (("key", observed_key, expected_key),
                                        ("value", observed_value, expected_value)):
            if not torch.equal(actual, expected):
                mismatch = (actual != expected).nonzero()
                example = tuple(mismatch[0].tolist())
                print(f"first {label} mismatch at {example}: "
                      f"observed={actual[example].item()}, expected={expected[example].item()}")

    result = {"case": case, "slot_mapping": CASES[case],
              "vllm_version": vllm_version, "torch_version": str(torch.__version__),
              "gpu": torch.cuda.get_device_name(0), "hip_version": str(torch.version.hip),
              "key_cache_pass": key_ok, "value_cache_pass": value_ok}
    if json_out:
        path = Path(json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Evidence written: {path}")
    if not (key_ok and value_ok):
        raise SystemExit(1)
    print("PASS: actual vLLM GPU operation matches the independent CPU reference")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", action="store_true", help="Offline mental-model checkpoint")
    parser.add_argument("--self-test", action="store_true", help="Offline address-mapping checks")
    parser.add_argument("--run", action="store_true", help="Invoke the real vLLM GPU operation")
    parser.add_argument("--case", choices=CASES, default="baseline")
    parser.add_argument("--json-out", help="Optional output path for machine-readable evidence")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.predict:
        print(f"Inputs: {NUM_TOKENS} tokens, {NUM_HEADS} heads, "
              f"{HEAD_SIZE} elements/head, {BLOCK_SIZE} slots/block")
        print(f"Mapping: {CASES[args.case]}")
        print("Predict the cache block and position for each token, then check:")
        describe(CASES[args.case])
        print("Source-based expected launch: grid=2 blocks, block=32 threads")
        print("Think: Can two 32-thread blocks occupy a 304-CU MI300X?")
    elif args.run:
        gpu_run(args.case, args.json_out)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
