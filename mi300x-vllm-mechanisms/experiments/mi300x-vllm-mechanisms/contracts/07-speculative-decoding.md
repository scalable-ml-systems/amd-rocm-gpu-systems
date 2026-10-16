# EXP-07 — Speculative Decoding

## QUESTION
How can one target-model verification step produce multiple accepted output tokens?

## MENTAL MODEL
A cheap proposer predicts several tokens; the target model verifies them together.

## PREDICTION
Predictable text will have higher acceptance than unpredictable text.

## BASELINE
Enable n-gram speculative decoding on a repetitive/predictable workload.

## BREAK
Use a low-repetition/unpredictable prompt.

## OBSERVE
- proposed tokens
- accepted tokens
- rejected tokens
- acceptance rate
- target-model decode steps

## SUCCESS EVIDENCE
High-acceptance and low-acceptance cases showing when speculation helps or wastes work.

## GPU REQUIREMENT
Two short inference workloads.
