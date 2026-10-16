# EXP-04 — PagedAttention

## QUESTION
How does a decode token find K/V data stored in non-contiguous physical blocks?

## MENTAL MODEL
Logical KV positions map through a block table to physical KV-cache blocks.

## PREDICTION
One sequence will span several physical blocks while appearing logically contiguous to attention.

## BASELINE
Run a sequence long enough to cross multiple KV blocks.

## BREAK
Change supported block-size configuration or inspect a block-boundary transition.

## OBSERVE
- logical token position
- logical block
- physical block
- block table
- paged-attention kernel dispatch

## SUCCESS EVIDENCE
Trace:
token → logical block → physical block → attention kernel.

## GPU REQUIREMENT
One targeted request + one lightweight kernel trace.
