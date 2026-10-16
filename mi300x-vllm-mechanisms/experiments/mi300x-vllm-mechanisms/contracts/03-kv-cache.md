# EXP-03 — KV Cache

## QUESTION
How does sequence growth consume KV-cache capacity?

## MENTAL MODEL
Every generated token adds K/V state that must remain available for future attention.

## PREDICTION
Longer/more concurrent sequences consume more KV blocks until capacity becomes limiting.

## BASELINE
Run requests with increasing sequence lengths.

## BREAK
Artificially reduce available KV-cache memory.

## OBSERVE
- total KV capacity
- used/free blocks
- active sequences
- allocation failures/preemption if triggered

## SUCCESS EVIDENCE
A clear sequence-length/concurrency → KV-capacity relationship.

## GPU REQUIREMENT
One normal-capacity run + one constrained-capacity run.
