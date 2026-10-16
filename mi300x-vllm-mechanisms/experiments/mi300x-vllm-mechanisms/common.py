from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.request

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_VLLM_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-30B-A3B"


@dataclass
class RequestSpec:
    """Defines one inference request used by an experiment."""

    request_id: str
    prompt: str
    max_output_tokens: int
    start_delay_seconds: float = 0.0
    temperature: float = 0.0
    ignore_eos: bool = False


@dataclass
class RequestResult:
    """Client-side timing and output from one completed request."""

    request_id: str
    request_started_seconds: float
    first_token_seconds: float | None
    request_finished_seconds: float
    time_to_first_token_seconds: float | None
    total_request_seconds: float
    output_text: str


@dataclass
class EventRecord:
    """One timestamped client-side event."""

    experiment_name: str
    experiment_mode: str
    request_id: str
    event_name: str
    elapsed_seconds: float
    value: str = ""


def build_chat_request_payload(
    request_spec: RequestSpec,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict:
    """
    Convert a RequestSpec into the OpenAI-compatible payload expected by vLLM.
    """

    return {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": request_spec.prompt,
            }
        ],
        "temperature": request_spec.temperature,
        "max_completion_tokens": request_spec.max_output_tokens,
        "ignore_eos": request_spec.ignore_eos,
        "stream": True,
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    }


def send_streaming_request(
    request_spec: RequestSpec,
    experiment_start_time: float,
    vllm_url: str = DEFAULT_VLLM_URL,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[RequestResult, list[EventRecord]]:
    """
    Send one streaming request to vLLM.

    Records:
    - when the request starts,
    - when the first generated text arrives,
    - when the request finishes.
    """

    request_start_time = time.perf_counter()
    request_started_seconds = request_start_time - experiment_start_time

    events = [
        EventRecord(
            experiment_name="",
            experiment_mode="",
            request_id=request_spec.request_id,
            event_name="request_started",
            elapsed_seconds=request_started_seconds,
        )
    ]

    request_payload = build_chat_request_payload(
        request_spec=request_spec,
        model_name=model_name,
    )

    encoded_payload = json.dumps(request_payload).encode("utf-8")

    http_request = urllib.request.Request(
        url=f"{vllm_url}/v1/chat/completions",
        data=encoded_payload,
        headers={
            "Content-Type": "application/json",
            "x-request-id": request_spec.request_id,
        },
        method="POST",
    )

    first_token_time: float | None = None
    generated_text_parts: list[str] = []

    try:
        with urllib.request.urlopen(http_request, timeout=600) as response:
            for raw_line in response:
                decoded_line = raw_line.decode("utf-8").strip()

                if not decoded_line:
                    continue

                if not decoded_line.startswith("data:"):
                    continue

                event_payload = decoded_line.removeprefix("data:").strip()

                if event_payload == "[DONE]":
                    break

                response_chunk = json.loads(event_payload)

                choices = response_chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                generated_text = delta.get("content")

                if not generated_text:
                    continue

                if first_token_time is None:
                    first_token_time = time.perf_counter()

                    events.append(
                        EventRecord(
                            experiment_name="",
                            experiment_mode="",
                            request_id=request_spec.request_id,
                            event_name="first_token_received",
                            elapsed_seconds=(
                                first_token_time - experiment_start_time
                            ),
                        )
                    )

                generated_text_parts.append(generated_text)

    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Request {request_spec.request_id} failed: {error}"
        ) from error

    request_finish_time = time.perf_counter()

    request_finished_seconds = (
        request_finish_time - experiment_start_time
    )

    total_request_seconds = (
        request_finish_time - request_start_time
    )

    if first_token_time is None:
        time_to_first_token_seconds = None
        first_token_seconds = None
    else:
        time_to_first_token_seconds = (
            first_token_time - request_start_time
        )
        first_token_seconds = (
            first_token_time - experiment_start_time
        )

    events.append(
        EventRecord(
            experiment_name="",
            experiment_mode="",
            request_id=request_spec.request_id,
            event_name="request_finished",
            elapsed_seconds=request_finished_seconds,
        )
    )

    request_result = RequestResult(
        request_id=request_spec.request_id,
        request_started_seconds=request_started_seconds,
        first_token_seconds=first_token_seconds,
        request_finished_seconds=request_finished_seconds,
        time_to_first_token_seconds=time_to_first_token_seconds,
        total_request_seconds=total_request_seconds,
        output_text="".join(generated_text_parts),
    )

    return request_result, events


def run_staggered_requests(
    request_specs: list[RequestSpec],
    experiment_name: str,
    experiment_mode: str,
    vllm_url: str = DEFAULT_VLLM_URL,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[list[RequestResult], list[EventRecord]]:
    """
    Run multiple requests with controlled arrival delays.

    All start_delay_seconds values are measured from the same experiment
    start time.
    """

    experiment_start_time = time.perf_counter()

    request_results: list[RequestResult] = []
    event_records: list[EventRecord] = []

    def run_one_request(
        request_spec: RequestSpec,
    ) -> tuple[RequestResult, list[EventRecord]]:

        scheduled_start_time = (
            experiment_start_time
            + request_spec.start_delay_seconds
        )

        remaining_delay_seconds = (
            scheduled_start_time - time.perf_counter()
        )

        if remaining_delay_seconds > 0:
            time.sleep(remaining_delay_seconds)

        result, events = send_streaming_request(
            request_spec=request_spec,
            experiment_start_time=experiment_start_time,
            vllm_url=vllm_url,
            model_name=model_name,
        )

        for event in events:
            event.experiment_name = experiment_name
            event.experiment_mode = experiment_mode

        return result, events

    with ThreadPoolExecutor(
        max_workers=len(request_specs)
    ) as thread_pool:

        future_to_request_id = {
            thread_pool.submit(
                run_one_request,
                request_spec,
            ): request_spec.request_id
            for request_spec in request_specs
        }

        for future in as_completed(future_to_request_id):
            request_result, request_events = future.result()

            request_results.append(request_result)
            event_records.extend(request_events)

    request_results.sort(
        key=lambda result: result.request_started_seconds
    )

    event_records.sort(
        key=lambda event: event.elapsed_seconds
    )

    return request_results, event_records


def make_approximate_token_prompt(
    approximate_token_count: int,
    repeated_text: str = "GPU memory scheduling attention inference ",
) -> str:
    """
    Build a deterministic long prompt for experiments.

    This is intentionally approximate because exact token count depends
    on the model tokenizer. Exact token counts should be measured later
    against the Qwen tokenizer when required by an experiment.
    """

    if approximate_token_count <= 0:
        raise ValueError(
            "approximate_token_count must be greater than zero"
        )

    words = repeated_text.split()

    repeated_words = [
        words[index % len(words)]
        for index in range(approximate_token_count)
    ]

    return " ".join(repeated_words)


def write_jsonl(
    records: Iterable,
    output_path: str | Path,
) -> None:
    """Write dataclass records as newline-delimited JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(
                json.dumps(asdict(record))
                + "\n"
            )


def write_csv(
    records: Iterable,
    output_path: str | Path,
) -> None:
    """Write dataclass records to CSV."""

    records = list(records)

    if not records:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_record = asdict(records[0])

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:

        writer = csv.DictWriter(
            output_file,
            fieldnames=first_record.keys(),
        )

        writer.writeheader()

        for record in records:
            writer.writerow(asdict(record))


def print_dry_run(
    request_specs: list[RequestSpec],
    experiment_name: str,
    experiment_mode: str,
) -> None:
    """
    Print the intended workload without contacting vLLM.
    """

    print()
    print(f"Experiment : {experiment_name}")
    print(f"Mode       : {experiment_mode}")
    print()

    print(
        f"{'Request':<10}"
        f"{'Delay(s)':<12}"
        f"{'Output tokens':<16}"
        f"{'Prompt chars':<15}"
    )

    print("-" * 53)

    for request_spec in request_specs:
        print(
            f"{request_spec.request_id:<10}"
            f"{request_spec.start_delay_seconds:<12.3f}"
            f"{request_spec.max_output_tokens:<16}"
            f"{len(request_spec.prompt):<15}"
        )

    print()
    print("No request was sent to vLLM.")
