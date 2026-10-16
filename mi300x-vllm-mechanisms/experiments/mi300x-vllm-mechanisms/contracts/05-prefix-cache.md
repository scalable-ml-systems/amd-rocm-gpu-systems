# EXP-05 — Prefix Cache

## QUESTION
When can one request reuse KV work produced by an earlier request?

## MENTAL MODEL
Identical full prefix blocks can reuse previously computed KV state.

## PREDICTION
A second request with the same long prefix will perform substantially less prefill work.

## BASELINE
R1 = long shared prefix + question A.
R2 = same prefix + question B.

## BREAK
Change one token early in the shared prefix.

## OBSERVE
- cached tokens
- newly computed tokens
- cache hits/misses
- prefill work

## SUCCESS EVIDENCE
Same prefix → reuse.
Modified prefix → reuse drops at/after the changed block.

## GPU REQUIREMENT
Three small requests.
