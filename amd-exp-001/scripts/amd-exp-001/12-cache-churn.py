#!/usr/bin/env python3
"""AMD-EXP-001: sequential hot/cold prefix-cache competition trial.

Default mode validates the frozen dataset and prints a request plan without
contacting an inference server. --run sends one request at a time to a freshly
started vLLM or SGLang instance whose prefix cache is explicitly enabled.

Requires the frozen 08 dataset, 11 offline calibration, and matching runtime
capacity records for *both* engines. Fresh engines need prefix caching enabled.
The trial deliberately avoids concurrent requests so queuing cannot be
mistaken for cache eviction. Do not claim eviction from latency alone.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONTRACT_REL = Path("benchmark-contracts/amd-exp-001/cache-pressure")
CALIB_REL = Path("benchmark-results/amd-exp-001/serving/calibration")
RESULT_REL = Path("benchmark-results/amd-exp-001/serving/cache-churn")
MODEL = "Qwen3-30B-A3B"
FIELDS = [
    "sequence", "phase", "request_id", "document_id", "class", "question_id",
    "prompt_sha256", "expected_prompt_tokens", "actual_prompt_tokens",
    "cached_prompt_tokens", "output_tokens", "ttft_ms", "total_ms", "finish_reason",
    "response_sha256", "status",
]
PHASES = ["warm_hot", "baseline_hot", "pressure", "probe_hot"]
METRIC_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([^\s]+)")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def prepare(repo: Path, engine: str, require_runtime: bool, scenario: str = "churn",
            probe: str = "hot-01") -> tuple[list[dict], dict]:
    root = repo / CONTRACT_REL
    manifest = load_json(root / "manifest.json")
    docs_path, reqs_path = root / "documents.jsonl", root / "requests.jsonl"
    require(digest(docs_path.read_bytes()) == manifest["documents_sha256"],
            "documents.jsonl does not match frozen manifest. Re-freeze deliberately with 08.")
    require(digest(reqs_path.read_bytes()) == manifest["requests_sha256"],
            "requests.jsonl does not match frozen manifest. Re-freeze deliberately with 08.")
    docs, requests = load_jsonl(docs_path), load_jsonl(reqs_path)
    by_doc = {doc["document_id"]: doc for doc in docs}
    require(len(by_doc) == len(docs), "Duplicate document identifiers")
    expected = set()
    for doc in docs:
        require(doc["class"] in ("hot", "cold"), "Unexpected document class")
        require(digest(doc["text"].encode()) == doc["sha256"],
                f"Document content hash mismatch: {doc['document_id']}")
    by_pair: dict[tuple[str, int], dict] = {}
    for req in requests:
        doc = by_doc.get(req["document_id"])
        require(doc is not None, f"Unknown document for {req['request_id']}")
        require(doc["class"] == req["class"], "Document class mismatch")
        pair = (req["document_id"], req["question_id"])
        require(pair not in by_pair, f"Duplicate document/question pair: {pair}")
        prompt = doc["text"] + "\n\nQUESTION:\n" + req["question"] + "\n\nANSWER:\n"
        require(digest(prompt.encode()) == req["prompt_sha256"],
                f"Prompt checksum mismatch: {req['request_id']}")
        require(req["max_output_tokens"] == manifest["generation"]["max_output_tokens"],
                "Inconsistent generation limits")
        by_pair[pair] = {**req, "prompt": prompt}
        expected.add(req["request_id"])
    require(len(expected) == len(requests), "Duplicate request identifiers")

    hot = sorted((d["document_id"] for d in docs if d["class"] == "hot"))
    cold = sorted((d["document_id"] for d in docs if d["class"] == "cold"))
    require(len(hot) == 8 and len(cold) == 8,
            "This trial is designed for exactly 8 hot and 8 cold documents")
    require(probe in hot, f"Probe must be one of: {', '.join(hot)}")
    # Control and churn use the same number of requests. Only document identity differs.
    # One probe per fresh-engine trial avoids the probe itself evicting later targets.
    phase_pairs = {
        "warm_hot": [(d, 1) for d in hot],
        "baseline_hot": [(d, 2) for d in hot],
        "pressure": [(d, 1) for d in cold] if scenario == "churn"
                    else [(d, 3) for d in hot],
        "probe_hot": [(probe, 4)],
    }
    plan = []
    for phase, pairs in phase_pairs.items():
        for pair in pairs:
            require(pair in by_pair, f"Missing required request: {pair}")
            plan.append({**by_pair[pair], "phase": phase, "sequence": len(plan) + 1})

    calib = repo / CALIB_REL
    offline = load_json(calib / "offline.json")
    require(offline["all_requests_fit"] is True, "Offline token check failed; rerun 11")
    require(offline["contract_sha256"] == manifest["requests_sha256"],
            "Offline token calibration belongs to another dataset; rerun 11")
    expected_tokens = {r["request_id"]: r["prompt_tokens"] for r in offline["requests"]}
    for r in plan:
        require(r["request_id"] in expected_tokens,
                f"Missing calibrated request {r['request_id']}")
        r["expected_prompt_tokens"] = expected_tokens[r["request_id"]]
        require(r["expected_prompt_tokens"] + r["max_output_tokens"] <=
                offline["assumptions"]["max_model_len"], "Request exceeds model context")

    planning = offline["planning"]
    info = {
        "contract_sha256": manifest["requests_sha256"],
        "tokenizer_revision": offline.get("tokenizer_revision"),
        "max_model_len": offline["assumptions"]["max_model_len"],
        "hot_reusable_prefix_tokens": planning["hot_reusable_prefix_tokens"],
        "cold_reusable_prefix_tokens": planning["cold_reusable_prefix_tokens"],
        "candidate_cache_budget_tokens": planning["candidate_cache_budget_tokens"],
        "runtime_capacity_tokens": None,
        "capacity_gate": "not checked (offline plan)",
        "probe_document": probe,
        "scenario": scenario,
    }
    if require_runtime:
        engine_calib = load_json(calib / f"{engine}.json")
        observation = engine_calib.get("runtime_observation", {})
        require(engine_calib.get("contract_sha256") == info["contract_sha256"],
                f"{engine} runtime calibration belongs to another dataset")
        require(engine_calib.get("all_requests_fit") is True and
                observation.get("engine") == engine,
                f"Invalid {engine} runtime calibration")
        capacity = observation.get("reported_capacity_tokens")
        require(isinstance(capacity, int) and capacity > 0,
                f"No measured {engine} cache capacity; complete 11 runtime calibration")
        minimum = planning["candidate_cache_budget_tokens"]
        combined = (planning["hot_reusable_prefix_tokens"] +
                    planning["cold_reusable_prefix_tokens"])
        require(capacity >= minimum,
                f"Cache capacity {capacity:,} is below safe planning budget {minimum:,}; "
                "adjust memory budget and recalibrate")
        require(capacity < combined,
                f"All hot + cold prefixes ({combined:,} tokens) may fit into the "
                f"reported cache ({capacity:,}); constrain the engine's KV budget "
                "and rerun 11 runtime calibration")
        # Paired-engine comparison requires comparable *measured* capacity,
        # not merely matching 0.80 memory-utilization settings.
        other = "sglang" if engine == "vllm" else "vllm"
        other_path = calib / f"{other}.json"
        require(other_path.exists(), f"Run 11 runtime calibration for {other} before "
                "this paired-engine experiment")
        other_report = load_json(other_path)
        other_observation = other_report.get("runtime_observation", {})
        require(other_report.get("contract_sha256") == info["contract_sha256"] and
                other_observation.get("engine") == other,
                f"{other} runtime calibration belongs to a different dataset")
        other_capacity = other_observation.get("reported_capacity_tokens")
        require(isinstance(other_capacity, int) and other_capacity > 0,
                f"{other} runtime cache capacity is missing")
        difference = abs(capacity - other_capacity) / max(capacity, other_capacity)
        require(difference <= 0.05,
                f"Engine cache budgets differ by {difference:.1%} "
                f"({engine}={capacity:,}, {other}={other_capacity:,} tokens). "
                "Reconfigure and recalibrate before a cross-engine comparison")
        info["runtime_capacity_tokens"] = capacity
        info["other_engine_capacity_tokens"] = other_capacity
        info["capacity_mismatch_fraction"] = round(difference, 6)
        info["capacity_gate"] = "passed; occupancy/eviction still requires observation"
    return plan, info


def get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def fetch_metrics(base_url: str) -> str:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=15) as response:
        return response.read().decode("utf-8")


def parse_metrics(body: str) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for line in body.splitlines():
        match = METRIC_RE.match(line)
        if not match:
            continue
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        result.setdefault(match.group(1), []).append(value)
    return result


def metric_sum(metrics: dict, *names: str) -> float | None:
    for name in names:
        if name in metrics:
            return sum(metrics[name])
    return None


def phase_metrics(engine: str, before: str, after: str) -> dict:
    start, end = parse_metrics(before), parse_metrics(after)
    if engine == "vllm":
        hit_names = ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits")
        query_names = ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries")
        hits_start, hits_end = metric_sum(start, *hit_names), metric_sum(end, *hit_names)
        queries_start, queries_end = metric_sum(start, *query_names), metric_sum(end, *query_names)
        hits = hits_end - hits_start if None not in (hits_start, hits_end) else None
        queries = (queries_end - queries_start
                   if None not in (queries_start, queries_end) else None)
        if ((hits is not None and hits < 0) or
                (queries is not None and queries < 0)):
            raise RuntimeError("Engine counters reset during trial; restart the trial")
        return {
            "prefix_hit_tokens_delta": hits,
            "prefix_query_tokens_delta": queries,
            "prefix_hit_fraction": hits / queries if hits is not None and queries and queries > 0 else None,
            "kv_cache_usage_after": metric_sum(end, "vllm:kv_cache_usage_perc"),
            "method": "phase counter differences, if exposed",
        }
    return {
        "reported_cache_hit_rate_after": metric_sum(end, "sglang:cache_hit_rate"),
        "token_usage_after": metric_sum(end, "sglang:token_usage"),
        "method": ("SGLang gauges are point-in-time diagnostics, NOT a "
                   "phase hit ratio. Prefer per-request cached tokens when available."),
    }


def streamed_request(base_url: str, model: str, item: dict, timeout: float) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": item["prompt"]}],
        "temperature": 0,
        "max_tokens": item["max_output_tokens"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    start = time.monotonic()
    first_content = None
    fragments = []
    usage: dict = {}
    finish_reason = None
    saw_done = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                break
            if not data:
                continue
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(f"Engine reported streaming error: {chunk['error']}")
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):
                    raise RuntimeError("Thinking content appeared; verify thinking-disabled configuration")
                content = delta.get("content")
                if content:
                    if first_content is None:
                        first_content = time.monotonic()
                    fragments.append(content)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
    finished = time.monotonic()
    if not saw_done or first_content is None or not fragments:
        raise RuntimeError("Missing streamed content or [DONE] marker")
    if finish_reason not in ("stop", "length"):
        raise RuntimeError(f"Unexpected finish reason: {finish_reason}")
    actual_prompt = usage.get("prompt_tokens")
    if not isinstance(actual_prompt, int):
        raise RuntimeError("Server did not return prompt token count in final usage frame")
    if actual_prompt != item["expected_prompt_tokens"]:
        raise RuntimeError(
            f"Tokenization mismatch for {item['request_id']}: "
            f"expected {item['expected_prompt_tokens']}, got {actual_prompt}; "
            "check model revision, chat template and thinking mode")
    generated = usage.get("completion_tokens")
    if not isinstance(generated, int) or generated <= 0:
        raise RuntimeError("Missing/invalid completion token count")
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    text = "".join(fragments)
    return {
        "actual_prompt_tokens": actual_prompt,
        "cached_prompt_tokens": cached if isinstance(cached, int) else None,
        "output_tokens": generated,
        "ttft_ms": round((first_content - start) * 1000, 3),
        "total_ms": round((finished - start) * 1000, 3),
        "finish_reason": finish_reason,
        "response_sha256": digest(text.encode()),
        "status": "PASS",
    }


def summarize(rows: list[dict], measurements: dict[str, dict]) -> dict:
    phases = {}
    for phase in [phase for phase in PHASES if any(r["phase"] == phase for r in rows)]:
        selected = [row for row in rows if row["phase"] == phase and row["status"] == "PASS"]
        ttfts = [float(r["ttft_ms"]) for r in selected]
        cached = [r["cached_prompt_tokens"] for r in selected
                  if r["cached_prompt_tokens"] is not None]
        phases[phase] = {
            "completed_requests": len(selected),
            "median_ttft_ms": round(statistics.median(ttfts), 3) if ttfts else None,
            "mean_ttft_ms": round(statistics.mean(ttfts), 3) if ttfts else None,
            "mean_cached_prompt_tokens_when_reported": round(statistics.mean(cached), 2) if cached else None,
            "requests_with_cached_token_report": len(cached),
            "server_metrics": measurements.get(phase),
        }
    baseline = {r["document_id"]: r for r in rows if r["phase"] == "baseline_hot"}
    probes = [r for r in rows if r["phase"] == "probe_hot"]
    require(len(probes) == 1, "Exactly one probe is required per independent trial")
    probe = probes[0]
    before = baseline[probe["document_id"]]
    comparison = {
        "document_id": probe["document_id"],
        "baseline_ttft_ms": before["ttft_ms"],
        "probe_ttft_ms": probe["ttft_ms"],
        "probe_to_baseline_ttft_ratio": round(
            probe["ttft_ms"] / before["ttft_ms"], 3) if before["ttft_ms"] else None,
        "baseline_cached_tokens": before["cached_prompt_tokens"],
        "probe_cached_tokens": probe["cached_prompt_tokens"],
    }
    has_request_cache_data = (comparison["baseline_cached_tokens"] is not None and
                              comparison["probe_cached_tokens"] is not None)
    has_vllm_counters = (measurements.get("baseline_hot", {}).get(
        "prefix_hit_tokens_delta") is not None and
        measurements.get("probe_hot", {}).get("prefix_hit_tokens_delta") is not None)
    return {
        "phases": phases,
        "probe_comparison": comparison,
        "interpretation": (
            "One fresh-engine trial is not proof of eviction or an eviction policy. "
            "Compare control and churn across independent repetitions. "
            "Check baseline cache reuse and cache telemetry before attributing "
            "any latency change to eviction."),
        "cache_telemetry_present": has_request_cache_data or has_vllm_counters,
    }



def execute(args: argparse.Namespace, plan: list[dict], info: dict) -> int:
    require(args.confirm_cache_on and args.confirm_fresh_engine,
            "--run requires --confirm-cache-on and --confirm-fresh-engine. "
            "Restart between trials, verify caching is ON and redo runtime calibration if capacity changes.")
    base_url = args.base_url.rstrip("/")
    parsed = urllib.parse.urlparse(base_url)
    require(parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1"),
            "Only local HTTP servers are supported; run the client on the AMD VM")
    models = get_json(f"{base_url}/v1/models", timeout=15)
    ids = [v.get("id", "") for v in models.get("data", [])]
    matches = [name for name in ids if MODEL in name]
    require(len(matches) == 1, f"Expected exactly one {MODEL} model; got {ids}")
    model_id = matches[0]
    # Fail before sending any workload if metric collection is unavailable.
    initial_metrics = fetch_metrics(base_url)
    parsed_metrics = parse_metrics(initial_metrics)
    prefix = "vllm:" if args.engine == "vllm" else "sglang:"
    require(any(name.startswith(prefix) for name in parsed_metrics),
            f"/{'metrics'} does not expose {args.engine} measurements; check startup flags")
    when = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.repo / RESULT_REL / args.engine / f"trial-{args.trial:02d}-{args.scenario}-{when}"
    session_root = args.repo / "benchmark-results/amd-exp-001/serving/sessions"
    startup_records = sorted(session_root.glob(f"*-{args.engine}-cache-on/startup.txt"),
                             key=lambda path: path.stat().st_mtime, reverse=True)
    require(startup_records, "Missing cache-ON startup record from 09-engine.sh")
    startup_text = startup_records[0].read_text(encoding="utf-8")
    startup = dict(line.split("=", 1) for line in startup_text.splitlines()
                   if "=" in line and not line.startswith("command="))
    require(startup.get("engine") == args.engine and startup.get("cache_mode") == "on",
            "Startup record is not for the selected engine with caching ON")
    require(startup.get("max_context_tokens") == str(info["max_model_len"]),
            "Startup context length differs from the frozen workload")
    expected_revision = info.get("tokenizer_revision")
    if expected_revision and re.fullmatch(r"[a-f0-9]{40}", expected_revision):
        require(startup.get("model_revision") == expected_revision,
                "Model revision differs from offline tokenizer revision; "
                "resolve before testing")
    # The operator must still verify the live container matches this record.
    output.mkdir(parents=True, exist_ok=False)
    meta = {
        "experiment": "AMD-EXP-001/cache-churn", "engine": args.engine,
        "trial": args.trial, "scenario": args.scenario, "probe": args.probe,
        "started_utc": when, "model": model_id,
        "cache_enabled": "operator-confirmed", "fresh_engine": "operator-confirmed",
        "mode": "sequential/no-concurrency", "temperature": 0,
        "startup_record_path": str(startup_records[0]) if startup_records else None,
        "startup_record_sha256": digest(startup_text.encode()) if startup_text else None,
        "startup_record": startup_text,
        "run_config": info,
        "warning": "Actual eviction must be supported by measured cache behavior, not latency alone.",
    }
    (output / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    (output / "plan.json").write_text(json.dumps([{k: v for k, v in item.items() if k != "prompt"}
                                                 for item in plan], indent=2) + "\n")
    rows = []
    measurements = {}
    (output / "metrics-initial.prom").write_text(initial_metrics, encoding="utf-8")
    try:
        with (output / "requests.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for phase in [phase for phase in PHASES if any(item["phase"] == phase for item in plan)]:
                before = fetch_metrics(base_url)
                (output / f"metrics-{phase}-before.prom").write_text(before, encoding="utf-8")
                print(f"\n{phase}: ", end="", flush=True)
                for item in (r for r in plan if r["phase"] == phase):
                    result = {name: item.get(name) for name in FIELDS}
                    result["status"] = "ERROR"
                    try:
                        result.update(streamed_request(base_url, model_id, item, args.timeout))
                    except Exception as exc:
                        result["status"] = f"ERROR: {exc}"
                        rows.append(result)
                        writer.writerow(result)
                        handle.flush()
                        raise
                    rows.append(result)
                    writer.writerow(result)
                    handle.flush()
                    print(".", end="", flush=True)
                after = fetch_metrics(base_url)
                (output / f"metrics-{phase}-after.prom").write_text(after, encoding="utf-8")
                measurements[phase] = phase_metrics(args.engine, before, after)
                print(" done", flush=True)
        analysis = summarize(rows, measurements)
        (output / "summary.json").write_text(json.dumps(analysis, indent=2) + "\n")
        meta["status"] = "COMPLETE"
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        meta["cache_telemetry_present"] = analysis["cache_telemetry_present"]
        print(f"\nTRIAL COMPLETE: {args.engine}; evidence: {output}")
        for phase in analysis["phases"]:
            print(f"  {phase}: median TTFT = {analysis['phases'][phase]['median_ttft_ms']} ms")
        if not analysis["cache_telemetry_present"]:
            print("WARNING: Per-request cache reuse not verified; retain raw /metrics for investigation")
        baseline = analysis["phases"].get("baseline_hot", {})
        baseline_counter = (baseline.get("server_metrics") or {}).get("prefix_hit_tokens_delta")
        baseline_rows = [r for r in rows if r["phase"] == "baseline_hot"]
        if not ((baseline_counter is not None and baseline_counter > 0)
                or any((r["cached_prompt_tokens"] or 0) > 0 for r in baseline_rows)):
            print("WARNING: No baseline cache reuse verified; investigate startup/telemetry")
        return 0
    except Exception as exc:
        meta["status"] = "INCOMPLETE"
        meta["failure"] = str(exc)
        print(f"\nTRIAL INCOMPLETE: {exc}\nPartial evidence: {output}", file=sys.stderr)
        return 2
    finally:
        (output / "run.json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=("vllm", "sglang"))
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--scenario", choices=("churn", "control"), default="churn",
                        help="churn uses new documents; control uses familiar documents")
    parser.add_argument("--probe", default="hot-01",
                        help="One hot document to probe; choose a different one in independent trials")
    parser.add_argument("--trial", type=int, default=1,
                        help="Independent repeat; start with a newly restarted engine")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--run", action="store_true", help="Send requests; otherwise print the offline plan")
    parser.add_argument("--confirm-cache-on", action="store_true",
                        help="Confirm startup has enabled prefix caching")
    parser.add_argument("--confirm-fresh-engine", action="store_true",
                        help="Confirm the engine was restarted for this independent trial")
    args = parser.parse_args()
    if args.trial < 1 or args.timeout <= 0:
        parser.error("--trial and --timeout must be positive")
    try:
        plan, info = prepare(args.repo, args.engine, args.run, args.scenario, args.probe)
        print(f"Frozen dataset: {info['contract_sha256'][:12]}... | "
              f"{len(plan)} requests | engine={args.engine} | scenario={args.scenario} "
              f"| probe={args.probe} | trial={args.trial}")
        print(f"Hot/cold reusable prefixes: {info['hot_reusable_prefix_tokens']:,} / "
              f"{info['cold_reusable_prefix_tokens']:,} tokens")
        print(f"Candidate budget: {info['candidate_cache_budget_tokens']:,} tokens")
        print(f"Runtime capacity: {info['runtime_capacity_tokens'] or 'not checked'}")
        for phase in [phase for phase in PHASES if any(p["phase"] == phase for p in plan)]:
            group = [p for p in plan if p["phase"] == phase]
            print(f"  {phase}: {len(group)} requests; "
                  f"{group[0]['document_id']} ... {group[-1]['document_id']}")
        if not args.run:
            print("DRY RUN COMPLETE — no server contacted; use --run only after 09, 10 and runtime 11 pass")
            return 0
        return execute(args, plan, info)
    except (ValueError, KeyError, FileNotFoundError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
