from __future__ import annotations

import argparse
import csv
import json
import urllib.request

from pathlib import Path

from common import (
    DEFAULT_MODEL_NAME,
    RequestSpec,
    print_dry_run,
    run_staggered_requests,
    write_csv,
    write_jsonl,
)


EXPERIMENT_NAME = "04-paged-attention"

RESULTS_DIRECTORY = Path(
    "results/04-paged-attention"
)

OUTPUT_TOKENS = 64

DEFAULT_BLOCK_SIZE = 16


def build_request() -> RequestSpec:
    """
    Build one request that generates enough tokens to cross
    several KV-cache block boundaries.

    We need only one request because this experiment is about:

        logical sequence position
            ->
        logical KV block
            ->
        physical KV block
            ->
        PagedAttention
    """

    return RequestSpec(
        request_id="R1-PAGED-KV",
        prompt=(
            "Explain how a GPU memory hierarchy works, including "
            "registers, local memory, caches, and HBM. Then explain "
            "how an inference system can store growing attention state "
            "without requiring one contiguous memory allocation."
        ),
        max_output_tokens=OUTPUT_TOKENS,
        start_delay_seconds=0.0,
        temperature=0.0,
        ignore_eos=True,
    )


def get_exact_prompt_token_count(
    prompt: str,
    vllm_url: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> int:
    """
    Ask the running Qwen3 server for the exact chat-template
    token count.

    We use the actual model tokenizer instead of estimating
    token count from words or characters.
    """

    tokenize_payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "add_generation_prompt": True,
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    }

    encoded_payload = json.dumps(
        tokenize_payload
    ).encode("utf-8")

    tokenize_request = urllib.request.Request(
        url=f"{vllm_url}/tokenize",
        data=encoded_payload,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(
        tokenize_request,
        timeout=30,
    ) as response:

        tokenize_response = json.loads(
            response.read().decode("utf-8")
        )

    return int(
        tokenize_response["count"]
    )


def calculate_logical_block_number(
    token_position: int,
    block_size: int,
) -> int:
    """
    Convert a zero-based sequence token position into
    its logical KV-cache block.

    Example with block_size=16:

        tokens 0-15   -> block 0
        tokens 16-31  -> block 1
        tokens 32-47  -> block 2
    """

    return token_position // block_size


def build_block_boundary_records(
    prompt_token_count: int,
    output_token_count: int,
    block_size: int,
) -> list[dict]:
    """
    Record each point during generation where the sequence
    enters a new logical KV block.

    Physical block IDs are NOT calculated here.

    They come from TraceScheduler using:

        kv_cache_manager.get_block_ids(request_id)

    during the real MI300X run.
    """

    block_boundary_records: list[dict] = []

    first_generated_token_position = (
        prompt_token_count
    )

    final_sequence_token_position = (
        prompt_token_count
        + output_token_count
        - 1
    )

    previous_logical_block = None

    for sequence_token_position in range(
        first_generated_token_position,
        final_sequence_token_position + 1,
    ):

        logical_block = (
            calculate_logical_block_number(
                token_position=sequence_token_position,
                block_size=block_size,
            )
        )

        if logical_block == previous_logical_block:
            continue

        generated_token_number = (
            sequence_token_position
            - prompt_token_count
            + 1
        )

        block_boundary_records.append(
            {
                "sequence_token_position":
                    sequence_token_position,

                "generated_token_number":
                    generated_token_number,

                "logical_block":
                    logical_block,

                "block_size":
                    block_size,
            }
        )

        previous_logical_block = logical_block

    return block_boundary_records


def write_dictionary_csv(
    records: list[dict],
    output_path: Path,
) -> None:

    if not records:
        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=records[0].keys(),
        )

        writer.writeheader()
        writer.writerows(records)


def print_logical_block_plan(
    prompt_token_count: int,
    block_size: int,
) -> None:

    total_sequence_tokens = (
        prompt_token_count
        + OUTPUT_TOKENS
    )

    prompt_last_token_position = (
        prompt_token_count - 1
    )

    final_sequence_token_position = (
        total_sequence_tokens - 1
    )

    prompt_ending_block = (
        calculate_logical_block_number(
            token_position=prompt_last_token_position,
            block_size=block_size,
        )
    )

    final_sequence_block = (
        calculate_logical_block_number(
            token_position=final_sequence_token_position,
            block_size=block_size,
        )
    )

    logical_blocks_used = (
        final_sequence_block + 1
    )

    print()
    print("Logical KV layout")
    print("-----------------")

    print(
        f"Prompt tokens          : {prompt_token_count}"
    )

    print(
        f"Generated tokens       : {OUTPUT_TOKENS}"
    )

    print(
        f"KV block size          : {block_size}"
    )

    print(
        f"Prompt ends in block   : {prompt_ending_block}"
    )

    print(
        f"Sequence ends in block : {final_sequence_block}"
    )

    print(
        f"Logical blocks used    : {logical_blocks_used}"
    )

    print()


def print_prediction(
    block_size: int,
) -> None:

    print()
    print("Prediction")
    print("----------")

    print(
        f"With a KV block size of {block_size}, every "
        f"{block_size} logical sequence positions require "
        "another KV-cache block."
    )

    print(
        "Logical blocks should look contiguous from the request's "
        "point of view, but the actual physical GPU block IDs "
        "do not need to be contiguous."
    )

    print(
        "PagedAttention should use the request's block table "
        "to locate those physical KV blocks during decode."
    )

    print()


def print_server_requirement(
    block_size: int,
) -> None:

    print("Required server configuration")
    print("-----------------------------")

    print(
        "Start Qwen3 with TraceScheduler and:"
    )

    print()

    print(
        f"  --block-size {block_size}"
    )

    print()

    print(
        "Enable full physical KV block tracing:"
    )

    print()

    print(
        "  TRACE_KV_BLOCK_IDS=1"
    )

    print()

    print(
        "The scheduler trace should therefore show:"
    )

    print(
        "  request_id"
    )

    print(
        "  num_computed_tokens_before_step"
    )

    print(
        "  physical_block_count"
    )

    print(
        "  physical_block_ids_by_group"
    )

    print(
        "  new_block_ids_this_step"
    )

    print()


def print_evidence_goal() -> None:

    print("Evidence goal")
    print("-------------")

    print(
        "For R1-PAGED-KV we want to correlate:"
    )

    print()

    print(
        "  sequence token position"
    )

    print(
        "        -> logical KV block"
    )

    print(
        "        -> physical GPU KV block ID"
    )

    print(
        "        -> paged_attention kernel"
    )

    print()


def parse_arguments() -> argparse.Namespace:

    argument_parser = argparse.ArgumentParser(
        description=(
            "EXP-04: Connect logical sequence positions to "
            "physical KV-cache blocks and the real "
            "PagedAttention decode path."
        )
    )

    argument_parser.add_argument(
        "--block-size",
        type=int,
        choices=[
            16,
            32,
        ],
        default=DEFAULT_BLOCK_SIZE,
        help=(
            "KV-cache block size. "
            "Use 16 for the primary experiment. "
            "32 is optional follow-up."
        ),
    )

    argument_parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    argument_parser.add_argument(
        "--vllm-url",
        default="http://127.0.0.1:8000",
    )

    return argument_parser.parse_args()


def main() -> None:

    arguments = parse_arguments()

    request_spec = build_request()

    experiment_mode = (
        f"block-{arguments.block_size}"
    )

    print()
    print(
        f"Experiment : {EXPERIMENT_NAME}"
    )

    print(
        f"Mode       : {experiment_mode}"
    )

    print_prediction(
        block_size=arguments.block_size,
    )

    print_server_requirement(
        block_size=arguments.block_size,
    )

    print_evidence_goal()

    if arguments.dry_run:

        print_dry_run(
            request_specs=[request_spec],
            experiment_name=EXPERIMENT_NAME,
            experiment_mode=experiment_mode,
        )

        print(
            "Exact Qwen token count and logical block boundaries "
            "will be calculated against the running server."
        )

        return

    prompt_token_count = (
        get_exact_prompt_token_count(
            prompt=request_spec.prompt,
            vllm_url=arguments.vllm_url,
        )
    )

    block_boundary_records = (
        build_block_boundary_records(
            prompt_token_count=prompt_token_count,
            output_token_count=OUTPUT_TOKENS,
            block_size=arguments.block_size,
        )
    )

    print_logical_block_plan(
        prompt_token_count=prompt_token_count,
        block_size=arguments.block_size,
    )

    request_results, event_records = (
        run_staggered_requests(
            request_specs=[request_spec],
            experiment_name=EXPERIMENT_NAME,
            experiment_mode=experiment_mode,
            vllm_url=arguments.vllm_url,
        )
    )

    mode_results_directory = (
        RESULTS_DIRECTORY
        / experiment_mode
    )

    write_csv(
        records=request_results,
        output_path=(
            mode_results_directory
            / "request-results.csv"
        ),
    )

    write_jsonl(
        records=event_records,
        output_path=(
            mode_results_directory
            / "client-events.jsonl"
        ),
    )

    write_dictionary_csv(
        records=block_boundary_records,
        output_path=(
            mode_results_directory
            / "logical-block-boundaries.csv"
        ),
    )

    print()
    print(
        "Client-side evidence saved under:"
    )

    print(
        mode_results_directory
    )

    print()

    print(
        "The server-side [BATCH_TRACE] log contains the "
        "physical KV block IDs."
    )

    print()

    print(
        "During the GPU execution phase, correlate those block "
        "transitions with one targeted PagedAttention kernel trace."
    )

    print()


if __name__ == "__main__":
    main()
