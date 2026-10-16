from __future__ import annotations

import json
import os

from vllm.v1.core.sched.scheduler import Scheduler


class TraceScheduler(Scheduler):
    """
    Observe vLLM's real scheduling decisions without changing them.

    Every call to schedule() represents one scheduler iteration.

    We call the normal vLLM scheduler first, then print a compact
    JSON record describing what vLLM decided to execute.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._trace_step = 0

        self._trace_kv_block_ids = (
        os.getenv("TRACE_KV_BLOCK_IDS", "0") == "1"
    )


    def schedule(self, *args, **kwargs):
        scheduler_output = super().schedule(*args, **kwargs)

        self._trace_step += 1

        scheduled_tokens_by_request = dict(
            scheduler_output.num_scheduled_tokens
        )

        new_request_ids = [
            request.req_id
            for request in scheduler_output.scheduled_new_reqs
        ]

        cached_request_ids = list(
            scheduler_output.scheduled_cached_reqs.req_ids
        )

        resumed_request_ids = sorted(
            scheduler_output.scheduled_cached_reqs.resumed_req_ids
        )

        finished_request_ids = sorted(
            scheduler_output.finished_req_ids
        )

        request_states = self._build_request_states(
            scheduler_output=scheduler_output,
        )

        trace_record = {
            "trace_type": "scheduler_iteration",
            "step": self._trace_step,

            "total_scheduled_tokens":
                scheduler_output.total_num_scheduled_tokens,

            "scheduled_request_count":
                len(scheduled_tokens_by_request),

            "scheduled_tokens_by_request":
                scheduled_tokens_by_request,

            "new_request_ids":
                new_request_ids,

            "cached_request_ids":
                cached_request_ids,

            "resumed_request_ids":
                resumed_request_ids,

            "finished_request_ids":
                finished_request_ids,

            "request_states":
                request_states,
        }

        print(
            "[BATCH_TRACE] "
            + json.dumps(
                trace_record,
                sort_keys=True,
            ),
            flush=True,
        )

        return scheduler_output


    def _build_request_states(
        self,
        scheduler_output,
    ) -> dict[str, dict]:
        """
        Describe what each scheduled request looks like at this
        scheduler iteration.

        New requests:
            SchedulerOutput already contains their initial block table.

        Cached/running requests:
            Ask KVCacheManager for their complete current block table.

        Full physical block IDs are printed only when
        TRACE_KV_BLOCK_IDS=1.
        """

        request_states: dict[str, dict] = {}

        #
        # Requests being scheduled for the first time.
        #
        for new_request in scheduler_output.scheduled_new_reqs:

            block_ids_by_group = [
                list(block_group)
                for block_group in new_request.block_ids
            ]

            request_state = {
                "request_kind": "new",

                "prompt_token_count": (
                    len(new_request.prompt_token_ids)
                    if new_request.prompt_token_ids is not None
                    else 0
                ),

                "num_computed_tokens_before_step":
                    new_request.num_computed_tokens,

                "num_output_tokens_before_step": 0,

                "num_tokens_scheduled_now":
                    scheduler_output.num_scheduled_tokens.get(
                        new_request.req_id,
                        0,
                    ),

                "physical_block_count":
                    sum(
                        len(block_group)
                        for block_group in block_ids_by_group
                    ),
            }

            if self._trace_kv_block_ids:
                request_state[
                    "physical_block_ids_by_group"
                ] = block_ids_by_group

            request_states[
                new_request.req_id
            ] = request_state

        #
        # Requests that have already been scheduled previously.
        #
        cached_requests = (
            scheduler_output.scheduled_cached_reqs
        )

        for (
            request_id,
            num_computed_tokens,
            num_output_tokens,
            new_block_ids,
        ) in zip(
            cached_requests.req_ids,
            cached_requests.num_computed_tokens,
            cached_requests.num_output_tokens,
            cached_requests.new_block_ids,
        ):

            all_block_ids = (
                self.kv_cache_manager.get_block_ids(
                    request_id
                )
            )

            all_block_ids_by_group = [
                list(block_group)
                for block_group in all_block_ids
            ]

            if new_block_ids is None:
                new_block_ids_by_group = []
            else:
                new_block_ids_by_group = [
                    list(block_group)
                    for block_group in new_block_ids
                ]

            request_state = {
                "request_kind": "cached",

                "is_resumed": (
                    request_id
                    in cached_requests.resumed_req_ids
                ),

                "num_computed_tokens_before_step":
                    num_computed_tokens,

                "num_output_tokens_before_step":
                    num_output_tokens,

                "num_tokens_scheduled_now":
                    scheduler_output.num_scheduled_tokens.get(
                        request_id,
                        0,
                    ),

                "physical_block_count":
                    sum(
                        len(block_group)
                        for block_group
                        in all_block_ids_by_group
                    ),

                "new_block_count_this_step":
                    sum(
                        len(block_group)
                        for block_group
                        in new_block_ids_by_group
                    ),
            }

            if self._trace_kv_block_ids:

                request_state[
                    "physical_block_ids_by_group"
                ] = all_block_ids_by_group

                request_state[
                    "new_block_ids_this_step"
                ] = new_block_ids_by_group

            request_states[
                request_id
            ] = request_state

        return request_states
