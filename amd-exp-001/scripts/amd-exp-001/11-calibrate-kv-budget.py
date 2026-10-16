#!/usr/bin/env python3
"""Offline token and KV-cache planning for AMD-EXP-001. Does not load model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen3-30B-A3B"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def common_prefix_length(sequences: list[list[int]]) -> int:
    if len(sequences) < 2:
        return 0
    length = 0
    for tokens in zip(*sequences):
        if len(set(tokens)) != 1:
            break
        length += 1
    return length


def round_up(value: int, multiple: int) -> int:
    return math.ceil(value / multiple) * multiple


def tokenizer_snapshot(path: str | None, revision: str | None) -> Path:
    if path:
        return Path(path).expanduser().resolve()
    from huggingface_hub import snapshot_download
    # Download tokenizer/config only. NEVER download the model weights to the Mac.
    location = snapshot_download(
        repo_id=MODEL,
        revision=revision,
        allow_patterns=[
            "config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "added_tokens.json", "vocab.json",
            "merges.txt", "tokenizer.model",
        ],
    )
    return Path(location)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--tokenizer-path", help="Existing local model snapshot; uses no network")
    parser.add_argument("--revision", help="Exact model commit; recommended for reproducibility")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--cache-block-size", type=int, default=16,
                        help="Planning granularity; actual engine allocation must be checked")
    parser.add_argument("--engine", choices=["vllm", "sglang"],
                        help="Use when recording an engine's reported capacity")
    parser.add_argument("--reported-capacity-tokens", type=int,
                        help="Usable KV-token capacity reported by a running engine")
    args = parser.parse_args()

    if min(args.max_model_len, args.concurrency, args.cache_block_size) < 1:
        parser.error("length, concurrency, and block size must be positive")
    if (args.engine is None) != (args.reported_capacity_tokens is None):
        parser.error("--engine and --reported-capacity-tokens must be supplied together")
    if args.reported_capacity_tokens is not None and args.reported_capacity_tokens < 1:
        parser.error("reported capacity must be positive")

    contract = args.repo / "benchmark-contracts/amd-exp-001/cache-pressure"
    doc_file = contract / "documents.jsonl"
    request_file = contract / "requests.jsonl"
    manifest = json.loads((contract / "manifest.json").read_text(encoding="utf-8"))

    # Reject changed input files before any expensive tokenization.
    for key, path in [("documents_sha256", doc_file), ("requests_sha256", request_file)]:
        actual = sha256(path.read_bytes())
        if actual != manifest[key]:
            raise ValueError(f"Frozen dataset changed: {path.name}; rerun script 08 deliberately")

    documents = read_jsonl(doc_file)
    requests = read_jsonl(request_file)
    if len({d["document_id"] for d in documents}) != len(documents):
        raise ValueError("Duplicate document IDs")
    if len({r["request_id"] for r in requests}) != len(requests):
        raise ValueError("Duplicate request IDs")
    document_by_id = {d["document_id"]: d for d in documents}
    for d in documents:
        if sha256(d["text"].encode("utf-8")) != d["sha256"]:
            raise ValueError(f"Document content changed: {d['document_id']}")

    model_dir = tokenizer_snapshot(args.tokenizer_path, args.revision)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True,
                                               trust_remote_code=False)

    # Two BF16 bytes for every Key and every Value, in each attention layer.
    bytes_per_token = (config["num_hidden_layers"] * config["num_key_value_heads"]
                       * config["head_dim"] * 2 * 2)
    max_output = manifest["generation"]["max_output_tokens"]
    document_rows = {}
    request_rows = []
    token_sequences = {}

    for d in documents:
        document_rows[d["document_id"]] = {
            "class": d["class"],
            "document_tokens": len(tokenizer(d["text"], add_special_tokens=False)["input_ids"]),
        }
        token_sequences[d["document_id"]] = []

    for r in requests:
        doc = document_by_id[r["document_id"]]
        if r["class"] != doc["class"]:
            raise ValueError(f"Request class mismatch: {r['request_id']}")
        if r["max_output_tokens"] != max_output:
            raise ValueError(f"Output length mismatch: {r['request_id']}")
        prompt = doc["text"] + "\n\nQUESTION:\n" + r["question"] + "\n\nANSWER:\n"
        if sha256(prompt.encode("utf-8")) != r["prompt_sha256"]:
            raise ValueError(f"Prompt changed: {r['request_id']}")
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        if not isinstance(token_ids, list):
            raise ValueError("Tokenizer did not return token IDs")
        token_sequences[r["document_id"]].append(token_ids)
        tokens = len(token_ids)
        request_rows.append({
            "request_id": r["request_id"], "document_id": r["document_id"],
            "class": r["class"], "prompt_tokens": tokens,
            "max_output_tokens": max_output,
            "total_tokens": tokens + max_output,
            "fits_context": tokens + max_output <= args.max_model_len,
        })

    for doc_id, sequences in token_sequences.items():
        common = common_prefix_length(sequences)
        document_rows[doc_id].update({
            "shared_prefix_tokens": common,
            "estimated_reusable_tokens": common // args.cache_block_size * args.cache_block_size,
        })

    hot_tokens = sum(d["estimated_reusable_tokens"] for d in document_rows.values()
                     if d["class"] == "hot")
    cold_tokens = sum(d["estimated_reusable_tokens"] for d in document_rows.values()
                      if d["class"] == "cold")
    largest_request = max(row["total_tokens"] for row in request_rows)
    all_fit = all(row["fits_context"] for row in request_rows)

    # Planning estimate only. Active requests need their own space in addition
    # to reusable prefixes. Allow a 10% margin before selecting a test budget.
    minimum_with_headroom = hot_tokens + args.concurrency * largest_request
    budget = (round_up(math.ceil(minimum_with_headroom * 1.10), args.cache_block_size)
              if all_fit else None)

    revision = model_dir.name if re.fullmatch(r"[0-9a-f]{40}", model_dir.name) else "verify-manually"
    report = {
        "experiment": "AMD-EXP-001", "purpose": "offline-cache-budget-planning",
        "tokenizer_snapshot": str(model_dir), "tokenizer_revision": revision,
        "contract_sha256": manifest["requests_sha256"],
        "assumptions": {
            "max_model_len": args.max_model_len, "max_output_tokens": max_output,
            "concurrency": args.concurrency, "cache_block_size_for_planning": args.cache_block_size,
            "kv_dtype": "BF16", "kv_bytes_per_token": bytes_per_token,
            "note": "Memory estimates omit allocator overhead and engine-specific reservations.",
        },
        "all_requests_fit": all_fit, "documents": document_rows, "requests": request_rows,
        "planning": {
            "hot_reusable_prefix_tokens": hot_tokens,
            "cold_reusable_prefix_tokens": cold_tokens,
            "largest_request_including_output": largest_request,
            "minimum_hot_plus_active_tokens": minimum_with_headroom,
            "candidate_cache_budget_tokens": budget,
            "candidate_cache_gib_bf16": round(budget * bytes_per_token / 2**30, 2) if budget else None,
            "hot_plus_cold_exceeds_budget": (hot_tokens + cold_tokens > budget)
                                             if budget else None,
        },
    }
    if args.engine:
        report["runtime_observation"] = {
            "engine": args.engine,
            "reported_capacity_tokens": args.reported_capacity_tokens,
            "reported_capacity_gib_bf16_estimate": round(
                args.reported_capacity_tokens * bytes_per_token / 2**30, 2),
            "meets_candidate_budget": (args.reported_capacity_tokens >= budget)
                                      if budget else None,
        }

    output_dir = args.repo / "benchmark-results/amd-exp-001/serving/calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (f"{args.engine}.json" if args.engine else "offline.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"TOKEN CHECK: {'PASS' if all_fit else 'FAIL'}")
    print(f"Documents: {len(documents)}; requests: {len(requests)}")
    print(f"Largest request: {largest_request:,} tokens including reserved output")
    print(f"Estimated BF16 KV memory: {bytes_per_token:,} bytes/token")
    print(f"Reusable hot/cold prefixes: {hot_tokens:,} / {cold_tokens:,} tokens")
    if budget:
        print(f"Candidate budget: {budget:,} tokens (~{report['planning']['candidate_cache_gib_bf16']} GiB)")
        if hot_tokens + cold_tokens <= budget:
            print("WARNING: all document prefixes may fit; adjust dataset or budget before eviction tests")
    else:
        invalid = [r for r in request_rows if not r["fits_context"]]
        print(f"BLOCKED: {len(invalid)} requests exceed {args.max_model_len:,} tokens including output")
        for row in invalid[:4]:
            print(f"  {row['request_id']}: {row['total_tokens']:,} tokens")
        print("Reduce document length in 08, deliberately re-freeze, and rerun 11. Do not start load tests.")
    print(f"Saved: {output}")
    return 0 if all_fit else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, KeyError, ImportError) as exc:
        print(f"CALIBRATION ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
