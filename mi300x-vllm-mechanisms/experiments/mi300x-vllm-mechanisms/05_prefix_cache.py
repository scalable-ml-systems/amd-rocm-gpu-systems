from __future__ import annotations

import argparse
import csv
import time
import urllib.request

from pathlib import Path

from common import (
    RequestSpec,
    make_approximate_token_prompt,
    print_dry_run,
    send_streaming_request,
    write_csv,
    write_jsonl,
)


EXPERIMENT_NAME = "05-prefix-cache"

RESULTS_DIRECTORY = Path(
    "results/05-prefix-cache"
)


PREFIX_CACHE_METRIC_NAMES = {
    "prefix_queries": {
        "vllm:prefix_cache_queries",
        "vllm:prefix_cache_queries_total",
    },
    "prefix_hits": {
        "vllm:prefix_cache_hits",
        "vllm:prefix_cache_hits_total",
    },
}


def build_shared_prefix() -> str:
    """
    Build one deterministic long prefix.

    R1 and R2 use this exact prefix.

    The prefix is deliberately long so that it spans many
    full KV-cache blocks and therefore gives prefix caching
    meaningful work to reuse.
    """

    repeated_context = make_approximate_token_prompt(
        approximate_token_count=2048,
        repeated_text=(
            "GPU inference memory scheduler attention "
            "model request token cache execution "
        ),
    )

    return (
        "SYSTEM NOTES\n\n"
        "GPU infrastructure observations follow.\n\n"
        + repeated_context
        + "\n\nEND SYSTEM NOTES\n\n"
    )


def build_mutated_prefix(
    shared_prefix: str,
) -> str:
    """
    Change one word near the beginning of the prefix.

    Prefix-cache block hashes depend on the preceding prefix.

    Therefore an early mutation should cause downstream block
    hashes to differ from those produced by R1.
    """

    mutated_prefix = shared_prefix.replace(
        "GPU infrastructure",
        "CPU infrastructure",
        1,
    )

    if mutated_prefix == shared_prefix:
        raise RuntimeError(
            "Prefix mutation failed. "
            "The expected text was not found."
        )

    return mutated_prefix


def build_requests() -> list[RequestSpec]:
    """
    Three sequential cases:

    R1-COLD
        Builds the initial prefix KV state.

    R2-REUSE
        Uses the identical prefix.

    R3-MUTATED
        Changes one early word in that prefix.
    """

    shared_prefix = build_shared_prefix()

    mutated_prefix = build_mutated_prefix(
        shared_prefix=shared_prefix,
    )

    return [
        RequestSpec(
            request_id="R1-COLD",
            prompt=(
                shared_prefix
                + "Question: summarize the role of memory bandwidth."
            ),
            max_output_tokens=32,
            temperature=0.0,
            ignore_eos=True,
        ),

        RequestSpec(
            request_id="R2-REUSE",
            prompt=(
                shared_prefix
                + "Question: summarize the role of request scheduling."
            ),
            max_output_tokens=32,
            temperature=0.0,
            ignore_eos=True,
        ),

        RequestSpec(
            request_id="R3-MUTATED",
            prompt=(
                mutated_prefix
                + "Question: summarize the role of attention."
            ),
            max_output_tokens=32,
            temperature=0.0,
            ignore_eos=True,
        ),
    ]


def fetch_prefix_cache_metrics(
    metrics_url: str,
) -> dict[str, float]:
    """
    Read only prefix-cache counters from vLLM /metrics.

    Prometheus counters may appear using their base name
    or with the standard "_total" suffix.
    """

    observed_metrics = {
        "prefix_queries": 0.0,
        "prefix_hits": 0.0,
    }

    with urllib.request.urlopen(
        metrics_url,
        timeout=5,
    ) as response:

        metrics_text = response.read().decode(
            "utf-8"
        )

    matched_total_metric = {
        "prefix_queries": False,
        "prefix_hits": False,
    }

    for line in metrics_text.splitlines():

        line = line.strip()

        if not line or line.startswith("#"):
            continue

        metric_fields = line.split()

        if len(metric_fields) < 2:
            continue

        metric_name = (
            metric_fields[0]
            .split("{", 1)[0]
        )

        try:
            metric_value = float(
                metric_fields[1]
            )
        except ValueError:
            continue

        for (
            logical_metric_name,
            accepted_metric_names,
        ) in PREFIX_CACHE_METRIC_NAMES.items():

            if metric_name not in accepted_metric_names:
                continue

            is_total_metric = (
                metric_name.endswith("_total")
            )

            if is_total_metric:
                observed_metrics[
                    logical_metric_name
                ] = metric_value

                matched_total_metric[
                    logical_metric_name
                ] = True

            elif not matched_total_metric[
                logical_metric_name
            ]:
                observed_metrics[
                    logical_metric_name
                ] = metric_value

    return observed_metrics


def calculate_metric_delta(
    metrics_before: dict[str, float],
    metrics_after: dict[str, float],
) -> dict[str, float]:
    """
    Convert cumulative server counters into evidence for
    one specific request.
    """

    return {
        metric_name: max(
            metrics_after.get(
                metric_name,
                0.0,
            )
            - metrics_before.get(
                metric_name,
                0.0,
            ),
            0.0,
        )
        for metric_name in metrics_before
    }


def run_one_request(
    request_spec: RequestSpec,
    experiment_start_time: float,
    vllm_url: str,
):
    """
    Run exactly one request.

    Prefix-cache cases are deliberately sequential so the
    cache state established by R1 exists before R2 runs.
    """

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
            "prefix-reuse"
        )

    return (
        request_result,
        event_records,
    )


def build_cache_evidence(
    request_id: str,
    metric_delta: dict[str, float],
    time_to_first_token_seconds: float | None,
    total_request_seconds: float,
) -> dict:

    queried_tokens = metric_delta[
        "prefix_queries"
    ]

    hit_tokens = metric_delta[
        "prefix_hits"
    ]

    if queried_tokens > 0:

        hit_rate_percent = (
            hit_tokens
            / queried_tokens
            * 100.0
        )

    else:
        hit_rate_percent = 0.0

    return {
        "request_id":
            request_id,

        "prefix_query_tokens":
            queried_tokens,

        "prefix_hit_tokens":
            hit_tokens,

        "prefix_hit_rate_percent":
            hit_rate_percent,

        "time_to_first_token_seconds":
            time_to_first_token_seconds,

        "total_request_seconds":
            total_request_seconds,
    }


def run_prefix_cache_experiment(
    vllm_url: str,
) -> None:

    request_specs = build_requests()

    metrics_url = (
        f"{vllm_url}/metrics"
    )

    experiment_start_time = (
        time.perf_counter()
    )

    all_request_results = []
    all_event_records = []
    cache_evidence_rows = []

    #
    # IMPORTANT:
    #
    # Sequential execution is intentional.
    #
    # R1 must complete and leave reusable KV blocks behind
    # before R2 tests reuse.
    #
    for request_spec in request_specs:

        print(
            f"Running {request_spec.request_id}..."
        )

        metrics_before = (
            fetch_prefix_cache_metrics(
                metrics_url=metrics_url,
            )
        )

        request_result, event_records = (
            run_one_request(
                request_spec=request_spec,
                experiment_start_time=experiment_start_time,
                vllm_url=vllm_url,
            )
        )

        metrics_after = (
            fetch_prefix_cache_metrics(
                metrics_url=metrics_url,
            )
        )

        metric_delta = (
            calculate_metric_delta(
                metrics_before=metrics_before,
                metrics_after=metrics_after,
            )
        )

        cache_evidence = (
            build_cache_evidence(
                request_id=(
                    request_spec.request_id
                ),
                metric_delta=metric_delta,
                time_to_first_token_seconds=(
                    request_result
                    .time_to_first_token_seconds
                ),
                total_request_seconds=(
                    request_result
                    .total_request_seconds
                ),
            )
        )

        cache_evidence_rows.append(
            cache_evidence
        )

        all_request_results.append(
            request_result
        )

        all_event_records.extend(
            event_records
        )

    save_results(
        request_results=all_request_results,
        event_records=all_event_records,
        cache_evidence_rows=(
            cache_evidence_rows
        ),
    )

    print_summary(
        cache_evidence_rows=(
            cache_evidence_rows
        ),
    )


def save_results(
    request_results,
    event_records,
    cache_evidence_rows: list[dict],
) -> None:

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        records=request_results,
        output_path=(
            RESULTS_DIRECTORY
            / "request-results.csv"
        ),
    )

    write_jsonl(
        records=event_records,
        output_path=(
            RESULTS_DIRECTORY
            / "client-events.jsonl"
        ),
    )

    output_path = (
        RESULTS_DIRECTORY
        / "prefix-cache-evidence.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                cache_evidence_rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            cache_evidence_rows
        )


def print_summary(
    cache_evidence_rows: list[dict],
) -> None:

    print()
    print("Prefix-cache evidence")
    print("---------------------")

    print(
        f"{'Request':<16}"
        f"{'Queried':<12}"
        f"{'Hits':<12}"
        f"{'Hit rate':<12}"
        f"{'TTFT(s)':<12}"
    )

    print("-" * 64)

    for evidence in cache_evidence_rows:

        time_to_first_token = evidence[
            "time_to_first_token_seconds"
        ]

        if time_to_first_token is None:

            formatted_ttft = "N/A"

        else:

            formatted_ttft = (
                f"{time_to_first_token:.4f}"
            )

        print(
            f"{evidence['request_id']:<16}"
            f"{evidence['prefix_query_tokens']:<12.0f}"
            f"{evidence['prefix_hit_tokens']:<12.0f}"
            f"{evidence['prefix_hit_rate_percent']:<11.1f}%"
            f"{formatted_ttft:<12}"
        )

    print()

    print("Expected pattern")
    print("----------------")

    print(
        "R1-COLD    -> little/no cached prefix"
    )

    print(
        "R2-REUSE   -> high prefix reuse"
    )

    print(
        "R3-MUTATED -> reuse should fall sharply"
    )

    print()


def print_prediction() -> None:

    print()
    print("Prediction")
    print("----------")

    print(
        "R1-COLD has no matching prefix cached yet, "
        "so it should perform the original prefill work."
    )

    print(
        "R2-REUSE uses the exact same long prefix. "
        "Most full prefix blocks should therefore be reusable."
    )

    print(
        "R3-MUTATED changes one word near the beginning "
        "of the prefix. Because downstream block hashes depend "
        "on their preceding prefix, reuse should fall sharply "
        "after that mutation."
    )

    print()


def print_server_requirement() -> None:

    print(
        "Required server configuration"
    )

    print(
        "-----------------------------"
    )

    print(
        "Start Qwen3 with automatic prefix caching enabled:"
    )

    print()

    print(
        "  --enable-prefix-caching"
    )

    print()


def parse_arguments() -> argparse.Namespace:

    argument_parser = argparse.ArgumentParser(
        description=(
            "EXP-05: Observe automatic prefix-cache reuse "
            "and deliberately break the shared prefix."
        )
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

    print_prediction()

    print_server_requirement()

    if arguments.dry_run:

        print_dry_run(
            request_specs=request_specs,
            experiment_name=EXPERIMENT_NAME,
            experiment_mode="prefix-reuse",
        )

        print()
        print(
            "Execution order on MI300X:"
        )

        print(
            "R1-COLD -> R2-REUSE -> R3-MUTATED"
        )

        print()

        return

    run_prefix_cache_experiment(
        vllm_url=arguments.vllm_url,
    )


if __name__ == "__main__":
    main()
