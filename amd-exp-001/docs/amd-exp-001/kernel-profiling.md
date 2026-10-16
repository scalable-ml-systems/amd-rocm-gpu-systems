# MIX300 Route A — Kernel Parallelism Observation

## Question

What happens when we run the **same GPU kernel** on the MI300X but give it much more independent work?

The kernel was a simple BF16 matrix–vector multiplication:

\[
y = Wx
\]

The kernel code did not change between the two runs. Each GPU workgroup contained **256 threads**, which corresponds to **4 AMD wavefronts of 64 threads each**.

The only variable we changed was the number of matrix rows.

## Test 1 — Small workload

We used:

- 64 matrix rows
- 4,096 columns
- 64 workgroups
- 256 threads per workgroup
- 256 total wavefronts

`rocprofv3` reported:

```text
Workgroup_Size_X = 256
Grid_Size_X      = 16,384
Kernel duration  = 10.464 µs
```

Since:

\[
16,384 / 256 = 64
\]

the profiler confirmed that the GPU received **64 workgroups**, exactly as our program requested.

The MI300X exposes **304 Compute Units (CUs)**. With only 64 independent workgroups, this launch does not provide enough workgroups to spread one across every CU simultaneously.

## Test 2 — Large workload

We then changed only the number of rows:

```text
64 → 8,192
```

The same kernel now generated:

- 8,192 workgroups
- 256 threads per workgroup
- 32,768 total wavefronts

`rocprofv3` reported:

```text
Workgroup_Size_X = 256
Grid_Size_X      = 2,097,152
Kernel duration  = 36.804 µs
```

Again:

\[
2,097,152 / 256 = 8,192
\]

so the profiler confirmed our predicted launch geometry.

## Observation

The large workload contained:

\[
8192 / 64 = 128\times
\]

more matrix rows and therefore **128× more workgroups**.

But GPU kernel execution time increased only from:

```text
10.464 µs → 36.804 µs
```

which is approximately:

\[
3.5\times
\]

longer.

So:

> **128× more available work did not produce 128× more execution time.**

The reason is GPU parallelism.

The small workload exposes only 64 independent workgroups to a GPU containing 304 Compute Units. Much of the GPU's parallel execution capacity cannot be used by that dispatch.

The 8,192-row workload exposes thousands of independent workgroups. The GPU has a much larger pool of work to distribute across its Compute Units and can execute many workgroups concurrently.

## Mental model

A GPU kernel does not become faster merely because the GPU is powerful.

The application must expose enough independent work for the hardware to execute in parallel.

```text
Problem size
     ↓
Number of workgroups
     ↓
Available parallel work
     ↓
GPU scheduler
     ↓
Compute Units execute work concurrently
     ↓
Observed kernel time
```

This is why **kernel execution time cannot be understood from the amount of mathematical work alone**.

We also need to know how that work was divided and how much parallelism was exposed to the GPU.

## What this does NOT prove

This experiment does not yet prove that all 304 CUs were busy, that the kernel achieved high occupancy, or that it used HBM bandwidth efficiently.

Those require hardware performance counters.

What we have established is simpler and fundamental:

> **Launch geometry determines how much parallel work is available to the MI300X, and increasing available parallelism can allow the GPU to process dramatically more work without a proportional increase in execution time.**

