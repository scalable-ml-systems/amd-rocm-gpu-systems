
#!/usr/bin/env python3
"""Verify one vLLM or SGLang server before running experiments."""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RESULTS = (
    REPO / "benchmark-results" / "amd-exp-001"
    / "serving" / "smoke"
)

EXPECTED_MODEL = "Qwen3-30B-A3B"

# Identical request for both engines.
SMOKE_REQUEST = {
    "messages": [
        {
            "role": "user",
            "content": "Reply with exactly READY. No explanation.",
        }
    ],
    "temperature": 0,
    "max_tokens": 24,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
}


def request_json(url, payload=None, timeout=120):
    """Send an HTTP request and return the JSON response."""

    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )

    try:
        with urllib.request.urlopen(
            request, timeout=timeout
        ) as response:
            return json.load(response)

    except urllib.error.HTTPError as exc:
        body = exc.read(500).decode(
            "utf-8", errors="replace"
        )
        raise RuntimeError(
            f"HTTP {exc.code} from {url}: {body}"
        ) from exc


def wait_for_model(base_url, wait_seconds):
    """Wait until the server advertises its loaded model."""

    deadline = time.monotonic() + wait_seconds
    last_error = "Server has not responded."

    print("Checking server readiness...", flush=True)

    while time.monotonic() < deadline:
        try:
            result = request_json(
                f"{base_url}/v1/models",
                timeout=10,
            )

            models = result.get("data", [])
            model_ids = [
                item["id"]
                for item in models
                if isinstance(item.get("id"), str)
            ]

            if model_ids:
                return model_ids

            last_error = "No models are advertised yet."

        except (
            urllib.error.URLError,
            RuntimeError,
            ValueError,
        ) as exc:
            last_error = str(exc)

            # An incorrect URL or failed authentication
            # will not be fixed by waiting.
            if "HTTP 401" in last_error:
                raise RuntimeError(last_error)
            if "HTTP 403" in last_error:
                raise RuntimeError(last_error)
            if "HTTP 404" in last_error:
                raise RuntimeError(last_error)

        time.sleep(5)

    raise RuntimeError(
        f"Server did not become ready: {last_error}"
    )


def validate_response(response):
    """Check that inference actually worked."""

    choices = response.get("choices", [])

    if not choices:
        raise RuntimeError(
            "The server returned no completion."
        )

    message = choices[0].get("message", {})
    content = message.get("content")

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            "The model returned an empty answer."
        )

    if "<think" in content.lower():
        raise RuntimeError(
            "Thinking content appeared in the answer."
        )

    reasoning = message.get("reasoning_content")
    if reasoning not in (None, "", []):
        raise RuntimeError(
            "Thinking mode appears to be enabled."
        )

    if not re.fullmatch(
        r"READY[.!]?",
        content.strip(),
        flags=re.IGNORECASE,
    ):
        raise RuntimeError(
            f"Unexpected model answer: {content[:200]!r}"
        )

    usage = response.get("usage") or {}
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")

    if not isinstance(input_tokens, int) or input_tokens <= 0:
        raise RuntimeError(
            "Valid input-token count was not returned."
        )

    if not isinstance(output_tokens, int) or not (
        0 < output_tokens <= SMOKE_REQUEST["max_tokens"]
    ):
        raise RuntimeError(
            "Invalid or missing output-token count."
        )

    return {
        "answer": content.strip(),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "finish_reason": choices[0].get("finish_reason"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        choices=["vllm", "sglang"],
        required=True,
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=480,
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    output_dir = RESULTS / args.engine
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{timestamp}.json"

    report = {
        "engine": args.engine,
        "base_url": base_url,
        "timestamp_utc": timestamp,
        "expected_model": EXPECTED_MODEL,
        "status": "FAIL",
    }

    try:
        # 1. Check readiness and model identity.
        model_ids = wait_for_model(
            base_url, args.wait_seconds
        )

        matching = [
            model_id
            for model_id in model_ids
            if EXPECTED_MODEL in model_id
        ]

        if len(matching) != 1:
            raise RuntimeError(
                f"Expected one {EXPECTED_MODEL} model; "
                f"server advertised: {model_ids}"
            )

        model_id = matching[0]
        report["advertised_model"] = model_id

        print(f"Model ready: {model_id}", flush=True)

        # 2. Send one short inference request.
        payload = dict(SMOKE_REQUEST)
        payload["model"] = model_id

        started = time.monotonic()

        response = request_json(
            f"{base_url}/v1/chat/completions",
            payload=payload,
        )

        elapsed = time.monotonic() - started

        # 3. Check answer, thinking mode and token counts.
        result = validate_response(response)

        report.update({
            "status": "PASS",
            "request": payload,
            "result": result,
            "elapsed_seconds": round(elapsed, 3),
            "raw_response": response,
        })

    except Exception as exc:
        report["error"] = str(exc)

    finally:
        output_file.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        print(f"\nSMOKE TEST: {report['status']}")
        print(f"Evidence: {output_file}")

    if report["status"] != "PASS":
        print(f"Reason: {report['error']}", file=sys.stderr)
        return 1

    result = report["result"]
    print(f"Answer: {result['answer']}")
    print(f"Input tokens: {result['input_tokens']}")
    print(f"Output tokens: {result['output_tokens']}")
    print("Thinking mode: no thinking content detected")

    return 0


if __name__ == "__main__":
    sys.exit(main())
