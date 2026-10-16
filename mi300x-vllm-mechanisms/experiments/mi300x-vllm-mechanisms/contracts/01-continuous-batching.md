# EXP-01 — Continuous Batching

## QUESTION
How does vLLM rebuild the active batch every scheduler iteration?

## MENTAL MODEL
Requests do not remain in one fixed batch.
Every iteration, vLLM decides which requests need work and how many tokens each gets.

## PREDICTION
Staggered requests will join and leave the active batch dynamically.

## BASELINE
Run R1, R2, R3, R4 with staggered arrivals and different output lengths.

## BREAK
Set max_num_seqs=1.

## OBSERVE
Per scheduler iteration:
- request IDs
- scheduled tokens/request
- new vs running requests
- total scheduled tokens

## SUCCESS EVIDENCE
A timeline showing the batch changing as requests arrive and finish.

## GPU REQUIREMENT
One normal Qwen3 server run + one constrained run.
