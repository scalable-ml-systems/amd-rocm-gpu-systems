from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    RequestSpec,
    print_dry_run,
    run_staggered_requests,
    write_csv,
    write_jsonl,
)


EXPERIMENT_NAME = "01-continuous-batching"

RESULTS_DIRECTORY = Path(
    "results/01-continuous-batching"
)


def build_baseline_requests() -> list[RequestSpec]:
    """
    Baseline:
    Four requests arrive at different times.

    The goal is to observe how vLLM continuously rebuilds
    the active batch as requests arrive and finish.
    """

    return [
        RequestSpec(
            request_id="R1",
            prompt=(
                "Explain how GPU memory bandwidth affects "
                "large language model inference."
            ),
            max_output_tokens=128,
            start_delay_seconds=0.00,
            temperature=0.0,
            ignore_eos=True,
        ),
        RequestSpec(
            request_id="R2",
            prompt=(
                "Explain why continuous batching can improve "
                "GPU utilization during inference."
            ),
            max_output_tokens=96,
            start_delay_seconds=0.05,
            temperature=0.0,
            ignore_eos=True,
        ),
        RequestSpec(
            request_id="R3",
            prompt=(
                "Explain the role of the KV cache during "
                "autoregressive decoding."
            ),
            max_output_tokens=64,
            start_delay_seconds=0.10,
            temperature=0.0,
            ignore_eos=True,
        ),
        RequestSpec(
            request_id="R4",
            prompt=(
                "Explain what a GPU workgroup is."
            ),
            max_output_tokens=32,
            start_delay_seconds=0.15,
            temperature=0.0,
            ignore_eos=True,
        ),
    ]


def build_break_requests() -> list[RequestSpec]:
    """
    Break condition:
    Use the same workload as baseline.

    The difference is NOT in this client script.

    The vLLM server must be started with:

        --max-num-seqs 1

    That prevents multiple requests from being active
    together and gives us a comparison against continuous
    batching.
    """

    return build_baseline_requests()


def get_requests_for_mode(
    experiment_mode: str,
) -> list[RequestSpec]:

    if experiment_mode == "baseline":
        return build_baseline_requests()

    if experiment_mode == "break":
        return build_break_requests()

    raise ValueError(
        f"Unsupported experiment mode: {experiment_mode}"
    )


def print_experiment_prediction(
    experiment_mode: str,
) -> None:

    print()
    print("Prediction")
    print("----------")

    if experiment_mode == "baseline":
        print(
            "R1 should begin first. As R2, R3, and R4 arrive, "
            "the scheduler should add them to later iterations "
            "while earlier requests continue decoding."
        )

    elif experiment_mode == "break":
        print(
            "With the server configured as --max-num-seqs 1, "
            "only one request should receive model execution "
            "at a time. The multi-request active batch should disappear."
        )

    print()


def print_server_requirement(
    experiment_mode: str,
) -> None:

    print("Required server configuration")
    print("-----------------------------")

    if experiment_mode == "baseline":
        print(
            "Normal Qwen3 vLLM server with scheduler tracing enabled."
        )

    elif experiment_mode == "break":
        print(
            "Same server, but add: --max-num-seqs 1"
        )

    print()


def save_results(
    experiment_mode: str,
    request_results,
    event_records,
) -> None:

    mode_results_directory = (
        RESULTS_DIRECTORY / experiment_mode
    )

    write_csv(
        records=request_results,
        output_path=(
            mode_results_directory
            / "request-results.csv"
        ),
    )

    write_csv(
        records=event_records,
        output_path=(
            mode_results_directory
            / "client-events.csv"
        ),
    )

    write_jsonl(
        records=request_results,
        output_path=(
            mode_results_directory
            / "request-results.jsonl"
        ),
    )

    write_jsonl(
        records=event_records,
        output_path=(
            mode_results_directory
            / "client-events.jsonl"
        ),
    )


def print_request_summary(
    request_results,
) -> None:

    print()
    print("Client-side request summary")
    print("---------------------------")

    print(
        f"{'Request':<10}"
        f"{'Start(s)':<12}"
        f"{'TTFT(s)':<12}"
        f"{'Total(s)':<12}"
    )

    print("-" * 46)

    for request_result in request_results:

        if request_result.time_to_first_token_seconds is None:
            time_to_first_token = "N/A"
        else:
            time_to_first_token = (
                f"{request_result.time_to_first_token_seconds:.4f}"
            )

        print(
            f"{request_result.request_id:<10}"
            f"{request_result.request_started_seconds:<12.4f}"
            f"{time_to_first_token:<12}"
            f"{request_result.total_request_seconds:<12.4f}"
        )

    print()


def parse_arguments() -> argparse.Namespace:

    argument_parser = argparse.ArgumentParser(
        description=(
            "EXP-01: Observe how vLLM continuously rebuilds "
            "the active request batch."
        )
    )

    argument_parser.add_argument(
        "--mode",
        choices=[
            "baseline",
            "break",
        ],
        required=True,
        help=(
            "baseline = normal continuous batching; "
            "break = server constrained with --max-num-seqs 1"
        ),
    )

    argument_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the planned workload without contacting vLLM."
        ),
    )

    argument_parser.add_argument(
        "--vllm-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the vLLM server.",
    )

    return argument_parser.parse_args()


def main() -> None:

    arguments = parse_arguments()

    request_specs = get_requests_for_mode(
        experiment_mode=arguments.mode,
    )

    print()
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"Mode       : {arguments.mode}")

    print_experiment_prediction(
        experiment_mode=arguments.mode,
    )

    print_server_requirement(
        experiment_mode=arguments.mode,
    )

    if arguments.dry_run:
        print_dry_run(
            request_specs=request_specs,
            experiment_name=EXPERIMENT_NAME,
            experiment_mode=arguments.mode,
        )
        return

    request_results, event_records = run_staggered_requests(
        request_specs=request_specs,
        experiment_name=EXPERIMENT_NAME,
        experiment_mode=arguments.mode,
        vllm_url=arguments.vllm_url,
    )

    save_results(
        experiment_mode=arguments.mode,
        request_results=request_results,
        event_records=event_records,
    )

    print_request_summary(
        request_results=request_results,
    )

    print(
        "Client evidence saved under:"
    )

    print(
        RESULTS_DIRECTORY
        / arguments.mode
    )

    print()
    print(
        "Important: the mechanism evidence comes from the "
        "server-side [BATCH_TRACE] scheduler log. "
        "Client timings only show when requests arrived, "
        "received their first token, and finished."
    )
    print()


if __name__ == "__main__":
    main()
