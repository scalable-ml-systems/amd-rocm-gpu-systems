#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = (
    ROOT
    / "benchmark-contracts"
    / "amd-exp-001"
    / "cache-pressure"
)

DOCUMENTS_FILE = OUT_DIR / "documents.jsonl"
REQUESTS_FILE = OUT_DIR / "requests.jsonl"
MANIFEST_FILE = OUT_DIR / "manifest.json"

HOT_DOCUMENTS = 8
COLD_DOCUMENTS = 8

# We deliberately freeze text length here, not token length.
# Exact token counts will be measured later using the model tokenizer.
TARGET_CHARACTERS = 36_000

MAX_OUTPUT_TOKENS = 128

QUESTIONS = [
    "Summarize the main system design decisions in this document.",
    "Identify the major performance bottlenecks described in this document.",
    "What reliability risks are discussed and how are they mitigated?",
    "Explain the memory and scheduling behavior described in this document.",
]


BASE_SECTIONS = [
    """
System Overview

The platform serves large language model inference requests on GPU
accelerators. Requests first enter an API layer and are placed into a
scheduler. The scheduler decides which requests can execute based on
available GPU memory, active sequences, and the amount of work required
to process new prompts.

Prompt processing and token generation place different demands on the
GPU. Long prompts require substantial initial processing before the
first output token can be returned. Once generation begins, the system
repeatedly executes the model to produce one or more additional tokens.
""",

    """
Memory Management

The inference engine stores intermediate attention information in a
key-value cache. Reusing previously processed prompt prefixes can avoid
performing the same calculations again. The effectiveness of this cache
depends on available GPU memory, request patterns, prefix reuse, and the
engine's eviction behavior.

When many distinct documents compete for the same cache capacity,
previously useful entries may be removed. A later request for an evicted
document must process that prefix again.
""",

    """
Scheduling

The inference scheduler must decide how to share GPU execution between
new requests and requests that are already generating tokens. Incoming
long prompts can require much more initial computation than small
requests.

A production system therefore has to balance throughput, time to first
token, generation speed, memory consumption, and fairness between
different users.
""",

    """
Reliability

Performance problems do not always appear as failures. A service may
remain technically available while request queues grow, first-token
latency increases, or generation becomes inconsistent.

Useful capacity is therefore defined by the amount of traffic that can
be served while still meeting response-time objectives rather than by
the absolute maximum number of requests the system can accept.
""",
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_document(group: str, index: int) -> str:
    """
    Produce deterministic but distinct documents.

    We intentionally vary identifying markers so hot/cold documents
    do not accidentally share their entire prefix.
    """

    header = f"""
DOCUMENT CLASS: {group.upper()}
DOCUMENT ID: {group}-{index:02d}

This is a deterministic inference systems document created for
AMD-EXP-001 cache-pressure experiments.

"""

    sections = []

    repeat = 0

    while len(header) + sum(len(x) for x in sections) < TARGET_CHARACTERS:
        section = BASE_SECTIONS[repeat % len(BASE_SECTIONS)]

        sections.append(
            f"\nSECTION ITERATION {repeat:04d} "
            f"FOR DOCUMENT {group}-{index:02d}\n"
            f"{section}"
        )

        repeat += 1

    document = header + "".join(sections)

    return document[:TARGET_CHARACTERS]


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = []

    for group, count in [
        ("hot", HOT_DOCUMENTS),
        ("cold", COLD_DOCUMENTS),
    ]:
        for i in range(1, count + 1):
            text = build_document(group, i)

            documents.append(
                {
                    "document_id": f"{group}-{i:02d}",
                    "class": group,
                    "text": text,
                    "characters": len(text),
                    "sha256": sha256_text(text),
                }
            )

    requests = []

    request_number = 1

    for doc in documents:
        for question_number, question in enumerate(QUESTIONS, start=1):

            prompt = (
                doc["text"]
                + "\n\nQUESTION:\n"
                + question
                + "\n\nANSWER:\n"
            )

            requests.append(
                {
                    "request_id": f"req-{request_number:04d}",
                    "document_id": doc["document_id"],
                    "class": doc["class"],
                    "question_id": question_number,
                    "question": question,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "prompt_sha256": sha256_text(prompt),
                }
            )

            request_number += 1

    write_jsonl(DOCUMENTS_FILE, documents)
    write_jsonl(REQUESTS_FILE, requests)

    manifest = {
        "experiment": "AMD-EXP-001",
        "workload": "cache-pressure",
        "schema_version": 1,

        "documents": {
            "hot": HOT_DOCUMENTS,
            "cold": COLD_DOCUMENTS,
            "target_characters_each": TARGET_CHARACTERS,
        },

        "questions_per_document": len(QUESTIONS),

        "generation": {
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "thinking": False,
        },

        "important_note": (
            "Character length is frozen here. Exact model token counts "
            "must be measured during cache-capacity calibration."
        ),

        "documents_sha256": sha256_text(
            DOCUMENTS_FILE.read_text(encoding="utf-8")
        ),

        "requests_sha256": sha256_text(
            REQUESTS_FILE.read_text(encoding="utf-8")
        ),
    }

    MANIFEST_FILE.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("WORKLOAD FROZEN")
    print(f"Documents: {len(documents)}")
    print(f"Requests:  {len(requests)}")
    print(f"Output:    {OUT_DIR}")
    print()
    print("Manifest:")
    print(MANIFEST_FILE.read_text())


if __name__ == "__main__":
    main()
