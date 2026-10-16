import torch

assert torch.cuda.is_available()

device = "cuda"

M = N = K = 4096

A = torch.randn((M, K), device=device, dtype=torch.bfloat16)
B = torch.randn((K, N), device=device, dtype=torch.bfloat16)

# Warm up library selection / compilation paths.
for _ in range(3):
    C = torch.matmul(A, B)

torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
C = torch.matmul(A, B)
end.record()

torch.cuda.synchronize()

ms = start.elapsed_time(end)
flops = 2 * M * N * K
tflops = flops / (ms / 1000) / 1e12

print(f"GEMM {M}x{K} @ {K}x{N}")
print(f"time_ms={ms:.4f}")
print(f"TFLOP/s={tflops:.2f}")
print(f"checksum={C.float().mean().item():.6f}")
