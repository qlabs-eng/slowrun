"""
Custom CUDA/Triton kernels for the tiny track.

Contains:
  - `transpose_copy`: a tiled Triton transpose-copy kernel.
  - `FusedSoftcappedCrossEntropy`: a fused fp8 softcapped cross-entropy autograd
    Function with multi-token prediction (MTP), backed by a hand-written CUDA kernel.

IMPORTANT: the fused CE CUDA kernel is compiled at *import time* via
`torch.cuda._compile_kernel`, which binds the resulting function to whatever CUDA
context is current. Callers must therefore bind the rank to its GPU
(`torch.cuda.set_device(LOCAL_RANK)`) *before* importing this module, otherwise every
rank compiles against cuda:0 and non-zero ranks later hit "CUDA error: invalid
resource handle" when launching the kernel on their own device.

CE_KERNEL_VOCAB_SIZE is fixed at 50304 = the model's padded vocab.
"""

import torch

import triton
import triton.language as tl


@triton.jit
def _transpose_copy_kernel(
    src_ptr, dst_ptr,
    M, N,
    src_stride_m, src_stride_n,
    dst_stride_0, dst_stride_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Coalesced read from src (M, N)
    tile = tl.load(
        src_ptr + offs_m[:, None] * src_stride_m + offs_n[None, :] * src_stride_n,
        mask=mask, other=0.0,
    )

    # Coalesced write to dst (N, M): dst[n, m] = src[m, n]
    mask_T = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(
        dst_ptr + offs_n[:, None] * dst_stride_0 + offs_m[None, :] * dst_stride_1,
        tl.trans(tile), mask=mask_T,
    )


def transpose_copy(src: torch.Tensor, dst: torch.Tensor):
    """Tiled transpose copy: dst = src.T where src is (M, N) and dst is (N, M).

    Uses a 64x128 tiled Triton kernel with coalesced reads AND writes,
    achieving near memory-bandwidth-limited performance.
    """
    assert src.ndim == 2 and dst.ndim == 2
    M, N = src.shape
    assert dst.shape == (N, M), f"Expected dst shape ({N}, {M}), got {dst.shape}"

    BLOCK_M, BLOCK_N = 64, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _transpose_copy_kernel[grid](
        src, dst,
        M, N,
        src.stride(0), src.stride(1),
        dst.stride(0), dst.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=8,
        num_stages=2,
    )


CE_KERNEL_BLOCK_SIZE = 256
CE_KERNEL_VOCAB_SIZE = 50304

CE_KERNEL_DECLS = f"""
constexpr int VOCAB_SIZE = {CE_KERNEL_VOCAB_SIZE};
constexpr int BLOCK_SIZE = {CE_KERNEL_BLOCK_SIZE};
"""

CE_KERNEL_SOURCE = """
#include <cuda_bf16.h>
#include <math_constants.h>

#define __nv_fp8_e5m2 char
#define uint16_t unsigned short
#define uint8_t unsigned char
#define int64_t long long

__device__ __forceinline__ __nv_fp8_e5m2 f32_to_fp8_e5m2(float x) {
    uint16_t packed;
    asm volatile(
        "cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;"
        : "=h"(packed)
        : "f"(x), "f"(0.0f)
    );
    __nv_fp8_e5m2 result;
    *reinterpret_cast<uint8_t*>(&result) = (packed & (0xFF << 8)) >> 8;
    return result;
}

struct __align__(16) __nv_bfloat168 {
    __nv_bfloat16 data[8];
    __device__ __nv_bfloat16& operator[](int i) { return data[i]; }
    __device__ const __nv_bfloat16& operator[](int i) const { return data[i]; }
};

struct __align__(8) __nv_fp8_e5m28 {
    __nv_fp8_e5m2 data[8];
    __device__ __nv_fp8_e5m2& operator[](int i) { return data[i]; }
    __device__ const __nv_fp8_e5m2& operator[](int i) const { return data[i]; }
};

template<typename T> __device__ constexpr T CEIL_DIV(T a, T b) { return (a + b - 1) / b; }


extern "C"
__launch_bounds__(BLOCK_SIZE, 2)
__global__ void ce_fwd_bwd_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const int64_t* __restrict__ targets,
    const float* __restrict__ mtp_weights,
    float* __restrict__ losses,
    __nv_fp8_e5m2* grad_input,
    int batch_size,
    int n_predict,
    double cap_param,
    double grad_s_param,
    double grad_scale_param)
{
  constexpr int VEC_WIDTH = 8;
  constexpr int NUM_FULL_LOADS = VOCAB_SIZE / (BLOCK_SIZE * VEC_WIDTH);
  constexpr int NUM_LOADS = CEIL_DIV(VOCAB_SIZE, BLOCK_SIZE * VEC_WIDTH);

  float cap = (float)cap_param;
  float grad_s = (float)grad_s_param;
  float grad_scale = (float)grad_scale_param;

  extern __shared__ __nv_bfloat16 smem[];

  static_assert(VEC_WIDTH == 8);

  const __nv_bfloat16 *block_logit_ptr = logits + VOCAB_SIZE * blockIdx.x;

  float thread_max = -CUDART_INF_F;

  #pragma unroll 25
  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      __nv_bfloat168 result = *(__nv_bfloat168*)(&block_logit_ptr[idx]);
      __nv_bfloat168 result_t;
      #pragma unroll
      for (int k = 0; k < VEC_WIDTH; k++) {
        float tmp = __bfloat162float(result[k]);
        tmp = tanhf(tmp / cap);
        result_t[k] = __float2bfloat16(tmp);
        tmp = cap * tmp;
        thread_max = max(tmp, thread_max);
      }
      *(__nv_bfloat168*)(&smem[idx]) = result_t;
    }
  }

  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  int warp_id = threadIdx.x / 32;
  __shared__ float block_maxs[NUM_WARPS];
  __shared__ float block_sums[NUM_WARPS];

  for (int offset = 16; offset > 0; offset >>= 1)
    thread_max = fmaxf(thread_max, __shfl_down_sync(0xFFFFFFFF, thread_max, offset));

  if (threadIdx.x % 32 == 0) {
    block_maxs[warp_id] = thread_max;
  }

  __syncthreads();

  float block_max = -CUDART_INF_F;
  for (int i = 0; i < NUM_WARPS; i++) {
    block_max = fmaxf(block_max, block_maxs[i]);
  }

  float thread_sum = 0.0f;
  #pragma unroll 2
  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    __nv_bfloat168 l;
    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      l = *(__nv_bfloat168*)(&smem[idx]);
    }
    #pragma unroll
    for (int k = 0; k < VEC_WIDTH; k++) {
      float tmp = cap * __bfloat162float(l[k]);
      tmp = __expf(tmp - block_max);
      if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
        thread_sum += tmp;
      }
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1)
    thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);

  if (threadIdx.x % 32 == 0) {
    block_sums[warp_id] = thread_sum;
  }

  __syncthreads();

  float block_sum = 0.0f;
  for (int i = 0; i < NUM_WARPS; i++) {
    block_sum += block_sums[i];
  }

  float lse = block_max + __logf(block_sum);

  if (threadIdx.x == 0) {
    float total_loss = 0.0f;
    for (int k = 0; k < n_predict; k++) {
      int64_t target_idx = blockIdx.x + k;
      if (target_idx < batch_size) {
        float weight = mtp_weights[k];
        int64_t target = targets[target_idx];
        if (target >= 0 && target < VOCAB_SIZE) {
          float z_target = cap * __bfloat162float(smem[target]);
          total_loss += weight * (lse - z_target);
        }
      }
    }
    losses[blockIdx.x] = total_loss;
  }

  float S_w = 0.0f;

  for (int i = 0; i < n_predict; i++) {
    S_w += mtp_weights[i];
  }

  #pragma unroll 4
  for (int i = 0; i < NUM_LOADS; i++) {
    int idx = i * BLOCK_SIZE * VEC_WIDTH + threadIdx.x * VEC_WIDTH;
    __nv_fp8_e5m28 result;

    if (i < NUM_FULL_LOADS || idx < VOCAB_SIZE) {
      __nv_bfloat168 ts = *(__nv_bfloat168*)(&smem[idx]);
      #pragma unroll
      for (int j = 0; j < VEC_WIDTH; j++) {
        float t = __bfloat162float(ts[j]);
        float z = cap * t;
        float p = __expf(z - lse);

        float term1 = S_w * p;
        float term2 = 0.0f;

        float grad_z = term1 - term2;
        float grad_x = grad_scale * (1.0f / grad_s) * grad_z * (1.0f - t * t);
        auto result_tmp = f32_to_fp8_e5m2(grad_x);
        result[j] = *reinterpret_cast<__nv_fp8_e5m2*>(&result_tmp);
      }
      *(__nv_fp8_e5m28*)(&grad_input[blockIdx.x * VOCAB_SIZE + idx]) = result;
    }
  }

  __syncthreads();

  if (threadIdx.x < n_predict && blockIdx.x + threadIdx.x < batch_size) {
    int i = threadIdx.x;
    int64_t target = targets[blockIdx.x + i];

    float t = __bfloat162float(smem[target]);
    float z = cap * t;
    float p = __expf(z - lse);

    float term1 = S_w * p;
    float term2 = 0.0f;

    #pragma unroll
    for (int k = 0; k < 3; k++) {
      int64_t target_idx = blockIdx.x + k;
      if (target_idx < batch_size && k < n_predict) {
        if (targets[target_idx] == target) {
          term2 += mtp_weights[k];
        }
      }
    }

    float grad_z = term1 - term2;
    float grad_x = grad_scale * (1.0f / grad_s) * grad_z * (1.0f - t * t);
    auto result_tmp = f32_to_fp8_e5m2(grad_x);
    auto result = *reinterpret_cast<__nv_fp8_e5m2*>(&result_tmp);
    grad_input[blockIdx.x * VOCAB_SIZE + target] = result;
  }
}
"""

ce_fwd_bwd_kernel = torch.cuda._compile_kernel(
    CE_KERNEL_DECLS + CE_KERNEL_SOURCE,
    "ce_fwd_bwd_kernel",
    compute_capability="90",
    cuda_include_dirs=["/usr/local/cuda/include/"],
    nvcc_options=["-lineinfo", "--use_fast_math"],
)
ce_fwd_bwd_kernel.set_shared_memory_config(CE_KERNEL_VOCAB_SIZE * 2)

@torch.library.custom_op("nanogpt::ce_fwd_bwd", mutates_args={"losses", "grad_input"})
def ce_fwd_bwd(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mtp_weights: torch.Tensor,
    losses: torch.Tensor,
    grad_input: torch.Tensor,
    n_rows: int,
    n_predict: int,
    cap: float,
    grad_s: float,
    grad_scale: float,
) -> None:
    grid = (n_rows, 1, 1)
    ce_fwd_bwd_kernel(
        grid,
        (CE_KERNEL_BLOCK_SIZE, 1, 1),
        (logits, targets, mtp_weights, losses, grad_input,
         n_rows, n_predict, cap, grad_s, grad_scale),
        shared_mem=CE_KERNEL_VOCAB_SIZE * 2,
    )

class FusedSoftcappedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, targets, mtp_weights, lm_head_weight, x_s, w_s, grad_s, grad_scale, cap=15.0):

        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = lm_head_weight.div(w_s).to(torch.float8_e4m3fn)

        w_f8_col_major = w_f8.T.contiguous().T

        logits = torch._scaled_mm(
            x_f8,
            w_f8_col_major,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )

        n_rows, n_cols = logits.shape
        if mtp_weights is None:
             mtp_weights = torch.tensor([1.0], device=logits.device, dtype=torch.float32)
        n_predict = mtp_weights.shape[0]

        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        lse = torch.empty(n_rows, dtype=torch.float32, device=logits.device)

        logits = logits.contiguous()
        targets = targets.contiguous()
        mtp_weights = mtp_weights.contiguous()

        grad_input = torch.empty((n_rows, n_cols), dtype=torch.float8_e5m2, device=logits.device)

        ce_fwd_bwd(logits, targets, mtp_weights, losses, grad_input,
             n_rows, n_predict, cap, grad_s, grad_scale)

        ctx.save_for_backward(logits, targets, mtp_weights, lse, x, lm_head_weight, x_f8, w_f8, grad_input)
        ctx.params = (cap, x_s, w_s, grad_s)
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, mtp_weights, lse, x, lm_head_weight, x_f8, w_f8, grad_input = ctx.saved_tensors
        _, x_s, w_s, grad_s = ctx.params
        n_rows, n_cols = logits.shape
        n_predict = mtp_weights.shape[0]

        grad_output = grad_output.contiguous()

        x_scale = grad_input.new_tensor(x_s, dtype=torch.float32)
        w_scale = grad_input.new_tensor(w_s, dtype=torch.float32)
        grad_scale = grad_input.new_tensor(grad_s, dtype=torch.float32)

        grad_x = torch._scaled_mm(
            grad_input,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=grad_scale,
            scale_b=w_scale,
            use_fast_accum=False,
        )
        grad_x = grad_x * grad_output.unsqueeze(-1)

        x_f8_T = torch.empty((x_f8.shape[1], x_f8.shape[0]), dtype=x_f8.dtype, device=x_f8.device)
        transpose_copy(x_f8, x_f8_T)  # (H, n_rows) row-major

        grad_input_T = torch.empty((n_cols, n_rows), dtype=grad_input.dtype, device=grad_input.device)
        transpose_copy(grad_input, grad_input_T)  # (V, n_rows) row-major

        grad_w = torch._scaled_mm(
            x_f8_T,            # (H, n_rows) row-major
            grad_input_T.T,    # (n_rows, V) column-major view
            out_dtype=torch.float32,
            scale_a=x_scale,
            scale_b=grad_scale,
            use_fast_accum=False,
        )
        grad_w = grad_w * grad_output.mean()

        return grad_x, None, None, grad_w, None, None, None, None, None
