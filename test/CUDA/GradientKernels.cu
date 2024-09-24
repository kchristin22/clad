// RUN: %cladclang_cuda -I%S/../../include -fsyntax-only \
// RUN:     --cuda-gpu-arch=%cudaarch --cuda-path=%cudapath  -Xclang -verify \
// RUN:     %s 2>&1 | %filecheck %s
//
// RUN: %cladclang_cuda -I%S/../../include --cuda-path=%cudapath \
// RUN:     --cuda-gpu-arch=%cudaarch %cudaldflags -oGradientKernels.out %s
//
// RUN: ./GradientKernels.out | %filecheck_exec %s
//
// REQUIRES: cuda-runtime
//
// expected-no-diagnostics

#include "clad/Differentiator/Differentiator.h"

__global__ void kernel(int *a) {
  *a *= *a;
}

// CHECK:    void kernel_grad(int *a, int *_d_a) {
//CHECK-NEXT:    int _t0 = *a;
//CHECK-NEXT:    *a *= *a;
//CHECK-NEXT:    {
//CHECK-NEXT:        *a = _t0;
//CHECK-NEXT:        int _r_d0 = *_d_a;
//CHECK-NEXT:        *_d_a = 0;
//CHECK-NEXT:        *_d_a += _r_d0 * *a;
//CHECK-NEXT:       *_d_a += *a * _r_d0;
//CHECK-NEXT:    }
//CHECK-NEXT: }

void fake_kernel(int *a) {
  *a *= *a;
}

__global__ void add_kernel(int *out, int *in) {
  int index = threadIdx.x;
  out[index] += in[index];
}

// CHECK:    void add_kernel_grad(int *out, int *in, int *_d_out, int *_d_in) {
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:     int index0 = threadIdx.x;
//CHECK-NEXT:     int _t0 = out[index0];
//CHECK-NEXT:     out[index0] += in[index0];
//CHECK-NEXT:     {
//CHECK-NEXT:         out[index0] = _t0;
//CHECK-NEXT:         int _r_d0 = _d_out[index0];
//CHECK-NEXT:         _d_in[index0] += _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

__global__ void add_kernel_2(int *out, int *in) {
  out[threadIdx.x] += in[threadIdx.x];
}

// CHECK:    void add_kernel_2_grad(int *out, int *in, int *_d_out, int *_d_in) {
//CHECK-NEXT:     int _t0 = out[threadIdx.x];
//CHECK-NEXT:     out[threadIdx.x] += in[threadIdx.x];
//CHECK-NEXT:     {
//CHECK-NEXT:         out[threadIdx.x] = _t0;
//CHECK-NEXT:         int _r_d0 = _d_out[threadIdx.x];
//CHECK-NEXT:         _d_in[threadIdx.x] += _r_d0;
//CHECK-NEXT:     }
//CHECK-NEXT: }

__global__ void add_kernel_3(int *out, int *in) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  out[index] += in[index];
}

// CHECK:    void add_kernel_3_grad(int *out, int *in, int *_d_out, int *_d_in) {
//CHECK-NEXT:    unsigned int _t1 = blockIdx.x;
//CHECK-NEXT:    unsigned int _t0 = blockDim.x;
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:    int index0 = threadIdx.x + _t1 * _t0;
//CHECK-NEXT:    int _t2 = out[index0];
//CHECK-NEXT:    out[index0] += in[index0];
//CHECK-NEXT:    {
//CHECK-NEXT:        out[index0] = _t2;
//CHECK-NEXT:        int _r_d0 = _d_out[index0];
//CHECK-NEXT:        _d_in[index0] += _r_d0;
//CHECK-NEXT:    }
//CHECK-NEXT:}

__global__ void add_kernel_4(int *out, int *in) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < 5) {
    int sum = 0;
    // Each thread sums elements in steps of warpSize
    for (int i = index; i < 5; i += warpSize) {
        sum += in[i];
    }
    out[index] = sum;
  }
}

// CHECK: void add_kernel_4_grad(int *out, int *in, int *_d_out, int *_d_in) {
//CHECK-NEXT:    bool _cond0;
//CHECK-NEXT:    int _d_sum = 0;
//CHECK-NEXT:    int sum = 0;
//CHECK-NEXT:    unsigned long _t2;
//CHECK-NEXT:    int _d_i = 0;
//CHECK-NEXT:    int i = 0;
//CHECK-NEXT:    clad::tape<int> _t3 = {};
//CHECK-NEXT:    clad::tape<int> _t4 = {};
//CHECK-NEXT:    int _t5;
//CHECK-NEXT:    unsigned int _t1 = blockIdx.x;
//CHECK-NEXT:    unsigned int _t0 = blockDim.x;
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:    int index0 = threadIdx.x + _t1 * _t0;
//CHECK-NEXT:    {
//CHECK-NEXT:        _cond0 = index0 < 5;
//CHECK-NEXT:        if (_cond0) {
//CHECK-NEXT:            sum = 0;
//CHECK-NEXT:            _t2 = 0UL;
//CHECK-NEXT:            for (i = index0; ; clad::push(_t3, i) , (i += warpSize)) {
//CHECK-NEXT:                {
//CHECK-NEXT:                    if (!(i < 5))
//CHECK-NEXT:                        break;
//CHECK-NEXT:                }
//CHECK-NEXT:                _t2++;
//CHECK-NEXT:                clad::push(_t4, sum);
//CHECK-NEXT:                sum += in[i];
//CHECK-NEXT:            }
//CHECK-NEXT:            _t5 = out[index0];
//CHECK-NEXT:            out[index0] = sum;
//CHECK-NEXT:        }
//CHECK-NEXT:    }
//CHECK-NEXT:    if (_cond0) {
//CHECK-NEXT:        {
//CHECK-NEXT:            out[index0] = _t5;
//CHECK-NEXT:            int _r_d2 = _d_out[index0];
//CHECK-NEXT:            _d_out[index0] = 0;
//CHECK-NEXT:            _d_sum += _r_d2;
//CHECK-NEXT:        }
//CHECK-NEXT:        {
//CHECK-NEXT:            for (;; _t2--) {
//CHECK-NEXT:                {
//CHECK-NEXT:                    if (!_t2)
//CHECK-NEXT:                        break;
//CHECK-NEXT:                }
//CHECK-NEXT:                {
//CHECK-NEXT:                    i = clad::pop(_t3);
//CHECK-NEXT:                    int _r_d0 = _d_i;
//CHECK-NEXT:                }
//CHECK-NEXT:                {
//CHECK-NEXT:                    sum = clad::pop(_t4);
//CHECK-NEXT:                    int _r_d1 = _d_sum;
//CHECK-NEXT:                    _d_in[i] += _r_d1;
//CHECK-NEXT:                }
//CHECK-NEXT:            }
//CHECK-NEXT:            _d_index += _d_i;
//CHECK-NEXT:        }
//CHECK-NEXT:    }
//CHECK-NEXT:}

__global__ void add_kernel_5(int *out, int *in) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < 5) {
        int sum = 0;
        // Calculate the total number of threads in the grid
        int totalThreads = blockDim.x * gridDim.x;
        // Each thread sums elements in steps of the total number of threads in the grid
        for (int i = index; i < 5; i += totalThreads) {
            sum += in[i];
        }
        out[index] = sum;
    }
}

// CHECK: void add_kernel_5_grad(int *out, int *in, int *_d_out, int *_d_in) {
//CHECK-NEXT:    bool _cond0;
//CHECK-NEXT:   int _d_sum = 0;
//CHECK-NEXT:    int sum = 0;
//CHECK-NEXT:    unsigned int _t2;
//CHECK-NEXT:    unsigned int _t3;
//CHECK-NEXT:    int _d_totalThreads = 0;
//CHECK-NEXT:    int totalThreads = 0;
//CHECK-NEXT:    unsigned long _t4;
//CHECK-NEXT:    int _d_i = 0;
//CHECK-NEXT:    int i = 0;
//CHECK-NEXT:    clad::tape<int> _t5 = {};
//CHECK-NEXT:    clad::tape<int> _t6 = {};
//CHECK-NEXT:    int _t7;
//CHECK-NEXT:    unsigned int _t1 = blockIdx.x;
//CHECK-NEXT:    unsigned int _t0 = blockDim.x;
//CHECK-NEXT:    int _d_index = 0;
//CHECK-NEXT:    int index0 = threadIdx.x + _t1 * _t0;
//CHECK-NEXT:    {
//CHECK-NEXT:        _cond0 = index0 < 5;
//CHECK-NEXT:        if (_cond0) {
//CHECK-NEXT:            sum = 0;
//CHECK-NEXT:            _t3 = blockDim.x;
//CHECK-NEXT:            _t2 = gridDim.x;
//CHECK-NEXT:            totalThreads = _t3 * _t2;
//CHECK-NEXT:            _t4 = 0UL;
//CHECK-NEXT:            for (i = index0; ; clad::push(_t5, i) , (i += totalThreads)) {
//CHECK-NEXT:                {
//CHECK-NEXT:                   if (!(i < 5))
//CHECK-NEXT:                       break;
//CHECK-NEXT:                }
//CHECK-NEXT:                _t4++;
//CHECK-NEXT:                clad::push(_t6, sum);
//CHECK-NEXT:                sum += in[i];
//CHECK-NEXT:            }
//CHECK-NEXT:            _t7 = out[index0];
//CHECK-NEXT:            out[index0] = sum;
//CHECK-NEXT:        }
//CHECK-NEXT:    }
//CHECK-NEXT:    if (_cond0) {
//CHECK-NEXT:        {
//CHECK-NEXT:            out[index0] = _t7;
//CHECK-NEXT:            int _r_d2 = _d_out[index0];
//CHECK-NEXT:            _d_out[index0] = 0;
//CHECK-NEXT:            _d_sum += _r_d2;
//CHECK-NEXT:        }
//CHECK-NEXT:        {
//CHECK-NEXT:            for (;; _t4--) {
//CHECK-NEXT:                {
//CHECK-NEXT:                    if (!_t4)
//CHECK-NEXT:                        break;
//CHECK-NEXT:                }
//CHECK-NEXT:                {
//CHECK-NEXT:                    i = clad::pop(_t5);
//CHECK-NEXT:                    int _r_d0 = _d_i;
//CHECK-NEXT:                    _d_totalThreads += _r_d0;
//CHECK-NEXT:                }
//CHECK-NEXT:                {
//CHECK-NEXT:                    sum = clad::pop(_t6);
//CHECK-NEXT:                    int _r_d1 = _d_sum;
//CHECK-NEXT:                    _d_in[i] += _r_d1;
//CHECK-NEXT:                }
//CHECK-NEXT:            }
//CHECK-NEXT:            _d_index += _d_i;
//CHECK-NEXT:        }
//CHECK-NEXT:    }
//CHECK-NEXT:}

#define TEST(F, grid, block, shared_mem, use_stream, x, dx, N)              \
  {                                                                         \
    int *fives = (int*)malloc(N * sizeof(int));                             \
    for(int i = 0; i < N; i++) {                                            \
      fives[i] = 5;                                                         \
    }                                                                       \
    int *ones = (int*)malloc(N * sizeof(int));                              \
    for(int i = 0; i < N; i++) {                                            \
      ones[i] = 1;                                                          \
    }                                                                       \
    cudaMemcpy(x, fives, N * sizeof(int), cudaMemcpyHostToDevice);          \
    cudaMemcpy(dx, ones, N * sizeof(int), cudaMemcpyHostToDevice);          \
    auto test = clad::gradient(F);                                          \
    if constexpr (use_stream) {                                             \
      cudaStream_t cudaStream;                                              \
      cudaStreamCreate(&cudaStream);                                        \
      test.execute_kernel(grid, block, shared_mem, cudaStream, x, dx);      \
    }                                                                       \
    else {                                                                  \
      test.execute_kernel(grid, block, x, dx);                              \
    }                                                                       \
    cudaDeviceSynchronize();                                                \
    int *res = (int*)malloc(N * sizeof(int));                               \
    cudaMemcpy(res, dx, N * sizeof(int), cudaMemcpyDeviceToHost);           \
    for (int i = 0; i < (N - 1); i++) {                                     \
      printf("%d, ", res[i]);                                               \
    }                                                                       \
    printf("%d\n", res[N-1]);                                               \
    free(fives);                                                            \
    free(ones);                                                             \
    free(res);                                                              \
  }


#define TEST_2(F, grid, block, shared_mem, use_stream, args, y, x, dy, dx, N) \
  {                                                                           \
    int *fives = (int*)malloc(N * sizeof(int));                               \
    for(int i = 0; i < N; i++) {                                              \
      fives[i] = 5;                                                           \
    }                                                                         \
    int *zeros = (int*)malloc(N * sizeof(int));                               \
    for(int i = 0; i < N; i++) {                                              \
      zeros[i] = 0;                                                           \
    }                                                                         \
    cudaMemcpy(x, fives, N * sizeof(int), cudaMemcpyHostToDevice);            \
    cudaMemcpy(y, zeros, N * sizeof(int), cudaMemcpyHostToDevice);            \
    cudaMemcpy(dy, fives, N * sizeof(int), cudaMemcpyHostToDevice);           \
    cudaMemcpy(dx, zeros, N * sizeof(int), cudaMemcpyHostToDevice);           \
    auto test = clad::gradient(F, args);                                      \
    if constexpr (use_stream) {                                               \
      cudaStream_t cudaStream;                                                \
      cudaStreamCreate(&cudaStream);                                          \
      test.execute_kernel(grid, block, shared_mem, cudaStream, y, x, dy, dx); \
    }                                                                         \
    else {                                                                    \
      test.execute_kernel(grid, block, y, x, dy, dx);                         \
    }                                                                         \
    cudaDeviceSynchronize();                                                  \
    int *res = (int*)malloc(N * sizeof(int));                                 \
    cudaMemcpy(res, dx, N * sizeof(int), cudaMemcpyDeviceToHost);             \
    for (int i = 0; i < (N - 1); i++) {                                       \
      printf("%d, ", res[i]);                                                 \
    }                                                                         \
    printf("%d\n", res[N-1]);                                                 \
    free(fives);                                                              \
    free(zeros);                                                              \
    free(res);                                                                \
  }


int main(void) {
  int *a, *d_a;
  cudaMalloc(&a, sizeof(int));
  cudaMalloc(&d_a, sizeof(int));

  TEST(kernel, dim3(1), dim3(1), 0, false, a, d_a, 1); // CHECK-EXEC: 10
  TEST(kernel, dim3(1), dim3(1), 0, true, a, d_a, 1); // CHECK-EXEC: 10

  auto error = clad::gradient(fake_kernel); 
  error.execute_kernel(dim3(1), dim3(1), a, d_a); // CHECK-EXEC: Use execute() for non-global CUDA kernels

  auto test = clad::gradient(kernel);
  test.execute(a, d_a); // CHECK-EXEC: Use execute_kernel() for global CUDA kernels

  cudaFree(a);
  cudaFree(d_a);


  int *dummy_in, *dummy_out, *d_out, *d_in;
  cudaMalloc(&dummy_in, 5 * sizeof(int));
  cudaMalloc(&dummy_out, 5 * sizeof(int));
  cudaMalloc(&d_out, 5 * sizeof(int));
  cudaMalloc(&d_in, 5 * sizeof(int));

  TEST_2(add_kernel, dim3(1), dim3(5, 1, 1), 0, false, "in, out", dummy_out, dummy_in, d_out, d_in, 5); // CHECK-EXEC: 5, 5, 5, 5, 5
  TEST_2(add_kernel_2, dim3(1), dim3(5, 1, 1), 0, true, "in, out", dummy_out, dummy_in, d_out, d_in, 5); // CHECK-EXEC: 5, 5, 5, 5, 5
  TEST_2(add_kernel_3, dim3(5, 1, 1), dim3(1), 0, false, "in, out", dummy_out, dummy_in, d_out, d_in, 5); // CHECK-EXEC: 5, 5, 5, 5, 5
  TEST_2(add_kernel_4, dim3(1), dim3(5, 1, 1), 0, false, "in, out", dummy_out, dummy_in, d_out, d_in, 5); // CHECK-EXEC: 5, 5, 5, 5, 5
  TEST_2(add_kernel_5, dim3(2, 1, 1), dim3(1), 0, false, "in, out", dummy_out, dummy_in, d_out, d_in, 5); // CHECK-EXEC: 5, 5, 5, 5, 5

  cudaFree(dummy_in);
  cudaFree(dummy_out);
  cudaFree(d_out);
  cudaFree(d_in);

  return 0;
}
