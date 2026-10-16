# EXP-02 — Chunked Prefill

## QUESTION
How does vLLM process a prompt that is larger than one scheduler token budget?

## MENTAL MODEL
A large prefill can be divided across multiple scheduler iterations.

## PREDICTION
With a small max_num_batched_tokens, one long prompt will appear as several scheduled chunks.

## BASELINE
Long prompt with normal scheduler token budget.

## BREAK
Reduce max_num_batched_tokens enough to force chunking.

## OBSERVE
Per iteration:
- prefill tokens scheduled
- decode tokens scheduled
- remaining prompt tokens

## SUCCESS EVIDENCE
One long prompt visibly split across multiple iterations.

## GPU REQUIREMENT
Two short scheduler traces.
