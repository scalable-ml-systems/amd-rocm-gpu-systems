from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request

from pathlib import Path

from common import (
    DEFAULT_MODEL_NAME,
    RequestSpec,
    make_approximate_token_prompt,
    print_dry_run,
    send_streaming_request,
    write_csv,
    write_jsonl,
)


EXPERIMENT_NAME = "06-prefill-attention"

RESULTS_DIRECTORY = Path(
    "results/06-prefill-attention"
)

TARGET_PROMPT_SIZES = [
    256,
    1024,
    4096,
]


def build_prompt(
    approximate_token_count: int,
) -> str:
    """
    Build a deterministic prompt intended to produce
    approximately the requested number of tokens.

    Exact Qwen token count is measured against the running
    server before inference.
    """

    technical_context = make_approximate_token_prompt(
        approximate_token_count=approximate_token_count,
        repeated_text=(
            "GPU attention memory query key value "
            "prefill inference compute bandwidth tile "
        ),
    )

    return (
        "Read the technical context below and answer with one word.\n\n"
        + technical_context
        + "\n\nQuestion: What hardware executes this workload?"
    )


def build_requests() -> list[RequestSpec]:
    """
    Generate only one output token.

    This keeps each request dominated by prompt prefill rather
    than autoregressive decode.

    These requests will be executed SEQUENTIALLY.
    """

    request_specs = []

    for approximate_token_count in TARGET_PROMPT_SIZES:

        request_specs.append(
            RequestSpec(
                request_id=(
                    f"PREFILL-{approximate_token_count}"
                ),
                prompt=build_prompt(
                    approximate_token_count
                ),
                max_output_tokens=1,
                start_delay_seconds=0.0,
                temperature=0.0,
                ignore_eos=True,
            )
        )

    return request_specs


def get_exact_prompt_token_count(
    prompt: str,
    vllm_url: str,
    model_name: str = DEFAULT_MODEL_NAME,
) -> int:
    """
    Use Qwen's actual tokenizer/chat template.

    We do not assume words or characters equal model tokens.
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


def run_one_prefill_request(
    request_spec: RequestSpec,
    experiment_mode: str,
    experiment_start_time: float,
    vllm_url: str,
) -> tuple:
    """
    Execute exactly one prefill workload.

    The next prompt size does not start until this request
    has completely finished.
    """

    exact_prompt_token_count = (
        get_exact_prompt_token_count(
            prompt=request_spec.prompt,
            vllm_url=vllm_url,
        )
    )

    print()
    print(
        f"Running {request_spec.request_id}"
    )

    print(
        f"Exact prompt tokens: "
        f"{exact_prompt_token_count}"
    )

    request_result, event_records = (
        send_streaming_request(
            request_spec=request_spec,
            experiment_start_time=experiment_start_time,
            vllm_url=vllm_url,
        )
    )

    for event_record in event_records:

        event_record.experiment_name = (
            EXPERIMENT_NAME
        )

        event_record.experiment_mode = (
            experiment_mode
        )

    summary_row = {
        "request_id":
            request_spec.request_id,

        "exact_prompt_tokens":
            exact_prompt_token_count,

        "time_to_first_token_seconds":
            request_result.time_to_first_token_seconds,

        "total_request_seconds":
            request_result.total_request_seconds,
    }

    return (
        request_result,
        event_records,
        summary_row,
    )


def run_prefill_experiment(
    request_specs: list[RequestSpec],
    experiment_mode: str,
    vllm_url: str,
) -> None:
    """
    Run:

        PREFILL-256
            finish
        PREFILL-1024
            finish
        PREFILL-4096
            finish

    No continuous batching is allowed to mix the prompt sizes.
    """

    experiment_start_time = (
        time.perf_counter()
    )

    request_results = []
    event_records = []
    prefill_summary_rows = []

    for request_spec in request_specs:

        (
            request_result,
            request_event_records,
            summary_row,
        ) = run_one_prefill_request(
            request_spec=request_spec,
            experiment_mode=experiment_mode,
            experiment_start_time=experiment_start_time,
            vllm_url=vllm_url,
        )

        request_results.append(
            request_result
        )

        event_records.extend(
            request_event_records
        )

        prefill_summary_rows.append(
            summary_row
        )

    save_results(
        experiment_mode=experiment_mode,
        request_results=request_results,
        event_records=event_records,
        prefill_summary_rows=prefill_summary_rows,
    )

    print_summary(
        prefill_summary_rows=(
            prefill_summary_rows
        )
    )


def save_results(
    experiment_mode: str,
    request_results,
    event_records,
    prefill_summary_rows: list[dict],
) -> None:

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

    output_path = (
        mode_results_directory
        / "prefill-summary.csv"
    )

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
            fieldnames=(
                prefill_summary_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            prefill_summary_rows
        )


def print_summary(
    prefill_summary_rows: list[dict],
) -> None:

    print()
    print("Prefill summary")
    print("---------------")

    print(
        f"{'Request':<16}"
        f"{'Prompt tokens':<16}"
        f"{'TTFT(s)':<14}"
        f"{'Total(s)':<14}"
    )

    print("-" * 60)

    for row in prefill_summary_rows:

        ttft = row[
            "time_to_first_token_seconds"
        ]

        if ttft is None:
            formatted_ttft = "N/A"
        else:
            formatted_ttft = (
                f"{ttft:.4f}"
            )

        print(
            f"{row['request_id']:<16}"
            f"{row['exact_prompt_tokens']:<16}"
            f"{formatted_ttft:<14}"
            f"{row['total_request_seconds']:<14.4f}"
        )

    print()


def print_prediction(
    experiment_mode: str,
) -> None:

    print()
    print("Prediction")
    print("----------")

    print(
        "As prompt length grows from roughly 256 to 4096 tokens, "
        "prefill attention must process substantially more "
        "query-key interactions."
    )

    print(
        "The optimized ROCm attention implementation should tile "
        "that work rather than materializing the complete N x N "
        "attention-score matrix in HBM."
    )

    if experiment_mode == "default":

        print(
            "We first observe the attention backend selected "
            "normally by this vLLM/ROCm image."
        )

    elif experiment_mode == "aiter":

        print(
            "This optional follow-up starts vLLM with AITER enabled "
            "and observes whether backend/kernel selection changes."
        )

    print()


def print_server_requirement(
    experiment_mode: str,
) -> None:

    print(
        "Required server configuration"
    )

    print(
        "-----------------------------"
    )

    if experiment_mode == "default":

        print(
            "Normal Qwen3 server."
        )

        print(
            "Do NOT force an attention backend."
        )

    elif experiment_mode == "aiter":

        print(
            "Optional comparison server with:"
        )

        print()

        print(
            "  VLLM_ROCM_USE_AITER=1"
        )

        print()

        print(
            "Do NOT force --attention-backend."
        )

    print()


def print_evidence_goal() -> None:

    print("Evidence goal")
    print("-------------")

    print(
        "For each isolated prefill size:"
    )

    print()

    print(
        "  exact prompt tokens"
    )

    print(
        "  selected attention backend"
    )

    print(
        "  actual prefill attention kernel"
    )

    print(
        "  prefill execution time"
    )

    print(
        "  memory traffic only if targeted profiling "
        "is needed to answer the mechanism question"
    )

    print()


def parse_arguments() -> argparse.Namespace:

    argument_parser = argparse.ArgumentParser(
        description=(
            "EXP-06: Observe how MI300X executes "
            "long-prompt prefill attention."
        )
    )

    argument_parser.add_argument(
        "--mode",
        choices=[
            "default",
            "aiter",
        ],
        default="default",
        help=(
            "default = observe normal backend selection; "
            "aiter = optional follow-up with AITER enabled"
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

    request_specs = build_requests()

    print()
    print(
        f"Experiment : {EXPERIMENT_NAME}"
    )

    print(
        f"Mode       : {arguments.mode}"
    )

    print_prediction(
        experiment_mode=arguments.mode,
    )

    print_server_requirement(
        experiment_mode=arguments.mode,
    )

    print_evidence_goal()

    if arguments.dry_run:

        print_dry_run(
            request_specs=request_specs,
            experiment_name=EXPERIMENT_NAME,
            experiment_mode=arguments.mode,
        )

        print()
        print(
            "Execution order:"
        )

        print(
            "PREFILL-256 -> finish"
        )

        print(
            "PREFILL-1024 -> finish"
        )

        print(
            "PREFILL-4096 -> finish"
        )

        print()

        return

    run_prefill_experiment(
        request_specs=request_specs,
        experiment_mode=arguments.mode,
        vllm_url=arguments.vllm_url,
    )


if __name__ == "__main__":
    main()
