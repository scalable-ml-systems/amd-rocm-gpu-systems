# EXP-06 — Prefill Attention

## QUESTION
How does the MI300X attention backend execute long-prompt prefill without materializing the full attention matrix in HBM?

## MENTAL MODEL
Tiled/Flash-style attention processes Q/K/V blocks on-chip instead of repeatedly storing an N×N matrix.

## PREDICTION
Longer prompts change attention work dramatically, but the optimized kernel avoids naive N² intermediate-memory traffic.

## BASELINE
Prompt lengths:
- 256
- 1024
- 4096

## BREAK
Compare the selected optimized attention path with another supported backend/configuration if available.

## OBSERVE
- actual selected ROCm attention backend
- kernel names
- execution time
- memory traffic where practical

## SUCCESS EVIDENCE
Connect:
prompt length → attention algorithm → actual MI300X kernel → memory behavior.

## GPU REQUIREMENT
Three prefills + one targeted profiler capture.
