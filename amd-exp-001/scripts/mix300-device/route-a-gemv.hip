// MIX300 Route A / A2: one BF16 matrix-vector GPU dispatch.
// Learning target: grid -> workgroup -> wavefront -> thread -> memory -> result.
// Not a tuned or MFMA-accelerated kernel; designed to expose execution mechanics.
//
// Compile on the MI300X VM:
//   hipcc -O2 --offload-arch=gfx942 route-a-gemv.hip -o route-a-gemv
// Run the first workload:
//   ./route-a-gemv 64
// Later, for A3, change only the number of rows:
//   ./route-a-gemv 8192

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define HIP_CHECK(call)                                                         \
  do {                                                                          \
    hipError_t status = (call);                                                  \
    if (status != hipSuccess) {                                                  \
      std::cerr << "HIP error at " << __FILE__ << ':' << __LINE__ << ": "         \
                << hipGetErrorString(status) << std::endl;                      \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

constexpr int COLS = 4096;
constexpr int THREADS_PER_BLOCK = 256;

// DEVICE CODE: One workgroup (blockIdx.x) computes exactly one output row.
__global__ void bf16_matvec(const __hip_bfloat16* __restrict__ weights,
                            const __hip_bfloat16* __restrict__ vector,
                            float* __restrict__ output, int cols) {
  const int row = blockIdx.x;      // Which output row belongs to this workgroup?
  const int lane = threadIdx.x;    // Which thread am I inside the workgroup?

  // Every thread reads 16 weights when cols = 4096 and blockDim.x = 256.
  // Adjacent threads read adjacent columns on each loop iteration.
  float partial_sum = 0.0f;
  for (int col = lane; col < cols; col += blockDim.x) {
    float w = __bfloat162float(weights[row * cols + col]);
    float x = __bfloat162float(vector[col]);
    partial_sum = fmaf(w, x, partial_sum);  // FP32 accumulation.
  }

  // Within one workgroup, combine the 256 thread-local partial sums.
  // HIP __shared__ memory maps to the compute unit's local data share (LDS).
  __shared__ float sums[THREADS_PER_BLOCK];
  sums[lane] = partial_sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (lane < offset) sums[lane] += sums[lane + offset];
    __syncthreads();
  }
  if (lane == 0) output[row] = sums[0];
}

int main(int argc, char** argv) {
  int rows = 64;
  if (argc == 2) {
    try {
      rows = std::stoi(argv[1]);
    } catch (...) {
      std::cerr << "Usage: " << argv[0] << " [rows: 1..8192]\n";
      return EXIT_FAILURE;
    }
  }
  if (rows < 1 || rows > 8192) {
    std::cerr << "Choose 1..8192 output rows.\n";
    return EXIT_FAILURE;
  }

  HIP_CHECK(hipSetDevice(0));
  hipDeviceProp_t gpu{};
  HIP_CHECK(hipGetDeviceProperties(&gpu, 0));
  std::cout << "DEVICE " << gpu.name << "  ARCH " << gpu.gcnArchName
            << "  CUs " << gpu.multiProcessorCount
            << "  WAVEFRONT " << gpu.warpSize << '\n';
  if (std::string(gpu.gcnArchName).find("gfx942") == std::string::npos) {
    std::cerr << "Expected gfx942 (MI300X); check the current device.\n";
    return EXIT_FAILURE;
  }

  // HOST CODE: Known, exactly representable BF16 input patterns.
  // Convert to BF16 on the CPU, then use those rounded values in the reference.
  std::vector<__hip_bfloat16> weights(static_cast<size_t>(rows) * COLS);
  std::vector<__hip_bfloat16> vector(COLS);
  for (int col = 0; col < COLS; ++col)
    vector[col] = __float2bfloat16((col % 2 == 0) ? 0.5f : -0.5f);
  for (int row = 0; row < rows; ++row)
    for (int col = 0; col < COLS; ++col)
      weights[static_cast<size_t>(row) * COLS + col] =
          __float2bfloat16(((row * 17 + col * 13) % 9 - 4) * 0.25f);

  std::vector<float> reference(rows, 0.0f), result(rows, 0.0f);
  for (int row = 0; row < rows; ++row)
    for (int col = 0; col < COLS; ++col)
      reference[row] +=
          __bfloat162float(weights[static_cast<size_t>(row) * COLS + col]) *
          __bfloat162float(vector[col]);

  __hip_bfloat16 *device_weights = nullptr, *device_vector = nullptr;
  float* device_output = nullptr;
  const size_t weight_bytes = weights.size() * sizeof(__hip_bfloat16);
  HIP_CHECK(hipMalloc(&device_weights, weight_bytes));
  HIP_CHECK(hipMalloc(&device_vector, vector.size() * sizeof(__hip_bfloat16)));
  HIP_CHECK(hipMalloc(&device_output, result.size() * sizeof(float)));
  HIP_CHECK(hipMemcpy(device_weights, weights.data(), weight_bytes,
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(device_vector, vector.data(),
                      vector.size() * sizeof(__hip_bfloat16),
                      hipMemcpyHostToDevice));

  // A single kernel dispatch: grid = rows; block = 256 threads.
  // Stream 0 is the default HIP stream; zero dynamic shared memory.
  std::cout << "LAUNCH grid=" << rows << " blocks, block=" << THREADS_PER_BLOCK
            << " threads, wavefronts/block="
            << (THREADS_PER_BLOCK / gpu.warpSize) << '\n';
  hipLaunchKernelGGL(bf16_matvec, dim3(rows), dim3(THREADS_PER_BLOCK), 0, 0,
                     device_weights, device_vector, device_output, COLS);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(result.data(), device_output,
                      result.size() * sizeof(float), hipMemcpyDeviceToHost));

  // Independent CPU reference: compare *every* output row.
  float max_error = 0.0f;
  for (int row = 0; row < rows; ++row) {
    float error = std::fabs(result[row] - reference[row]);
    max_error = std::max(max_error, error);
    if (error > 1.0e-3f) {
      std::cerr << "FAIL row " << row << ": GPU=" << result[row]
                << " CPU=" << reference[row] << " error=" << error << '\n';
      return EXIT_FAILURE;
    }
  }
  std::cout << "PASS rows=" << rows << " cols=" << COLS
            << " max_abs_error=" << max_error << '\n';
  std::cout << "EXAMPLE row0: GPU=" << result[0]
            << " CPU=" << reference[0] << '\n';

  HIP_CHECK(hipFree(device_weights));
  HIP_CHECK(hipFree(device_vector));
  HIP_CHECK(hipFree(device_output));
  return EXIT_SUCCESS;
}
