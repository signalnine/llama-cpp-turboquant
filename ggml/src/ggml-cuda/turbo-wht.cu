#include "turbo-quant.cuh"
#include "turbo-wht.cuh"

// ─── CUDA kernel ──────────────────────────────────────────────────────────────
//
// Templated on direction and group_size (128 or 64).
// One block per group, group_size threads per block.
// direction: 0 = forward (signs1 → WHT → signs2), 1 = inverse (signs2 → WHT → signs1)
//
// When head_dim is not a multiple of group_size, only the full groups
// within each head are processed.  Tail elements are left unchanged (identity).
//
// Algorithm mirrors the CPU implementation in ggml-cpu/ops.cpp:
//   1. Apply s_first elementwise
//   2. Radix-2 Hadamard butterfly (log2(group_size) stages, in-place)
//   3. Normalize by 1/sqrt(group_size) and apply s_second elementwise
//
// InnerQ scale_inv: when non-null, applies per-channel inverse scaling for
// Q/V equalization. For forward (Q rotation): multiply BEFORE signs+WHT.
// For inverse (V un-rotation): multiply AFTER WHT+signs.

template <int direction, int group_size>
static __global__ void k_turbo_wht_f32(const float * __restrict__ src,
                                        float * __restrict__ dst,
                                        const float * __restrict__ scale_inv,
                                        int64_t n_groups,
                                        int64_t head_dim,
                                        int64_t groups_per_head) {
    static_assert(group_size == 128 || group_size == 64, "group_size must be 128 or 64");

    const int64_t g = blockIdx.x;
    if (g >= n_groups) return;

    const int t = threadIdx.x;  // 0 .. group_size-1

    // Map group index to position in the tensor:
    // each head has groups_per_head full groups, then a gap of tail elements.
    const int64_t head_idx     = g / groups_per_head;
    const int64_t grp_in_head  = g % groups_per_head;
    const int64_t base         = head_idx * head_dim + grp_in_head * group_size;

    __shared__ float x[group_size];

    // Load from global memory
    x[t] = src[base + t];
    __syncthreads();

    // InnerQ forward: apply scale_inv BEFORE signs+WHT (for Q pre-rotation)
    if (direction == 0 && scale_inv != nullptr) {
        x[t] *= scale_inv[t % group_size];
        __syncthreads();
    }

    // Apply first sign array
    if (group_size == 128) {
        x[t] *= (direction == 0) ? TURBO_WHT_SIGNS1[t] : TURBO_WHT_SIGNS2[t];
    } else {
        x[t] *= (direction == 0) ? TURBO_WHT_SIGNS1_64[t] : TURBO_WHT_SIGNS2_64[t];
    }
    __syncthreads();

    // WHT butterfly — log2(group_size) stages.
    // In stage h, threads where (t % (2h)) < h read x[t] and x[t+h],
    // then write x[t] = a+b and x[t+h] = a-b.  Each active thread
    // owns a disjoint pair, so no intra-stage conflicts exist.
#define WHT_STAGE(h) \
    if (t % (2*(h)) < (h)) { float a = x[t], b = x[t+(h)]; x[t] = a+b; x[t+(h)] = a-b; } \
    __syncthreads();

    WHT_STAGE(1)
    WHT_STAGE(2)
    WHT_STAGE(4)
    WHT_STAGE(8)
    WHT_STAGE(16)
    WHT_STAGE(32)
    if (group_size == 128) { WHT_STAGE(64) }
#undef WHT_STAGE

    // Normalize and apply second sign array, write to output
    constexpr float inv_sqrt = (group_size == 128) ? 0.08838834764831845f : 0.125f;
    float result;
    if (group_size == 128) {
        result = x[t] * inv_sqrt *
            ((direction == 0) ? TURBO_WHT_SIGNS2[t] : TURBO_WHT_SIGNS1[t]);
    } else {
        result = x[t] * inv_sqrt *
            ((direction == 0) ? TURBO_WHT_SIGNS2_64[t] : TURBO_WHT_SIGNS1_64[t]);
    }

    // InnerQ inverse: apply scale_inv AFTER WHT+signs (for V un-rotation)
    if (direction == 1 && scale_inv != nullptr) {
        result *= scale_inv[t % group_size];
    }

    dst[base + t] = result;
}

// ─── Vilenkin-Hartley kernel (non-power-of-2 group sizes) ────────────────────
//
// Mixed-radix butterfly in shared memory. One block per group, group_size threads.
// Binary stages (p=2) use the same add/subtract butterfly as WHT.
// p-ary stages (p=3,5,7,...) use the Discrete Hartley kernel: cas(2πkm/p).
//
// Runtime group_size (not templated) since non-pow2 dims are diverse.
// No sign arrays for non-pow2 — the VHT itself provides decorrelation.

static __global__ void k_vilenkin_f32(const float * __restrict__ src,
                                       float * __restrict__ dst,
                                       const float * __restrict__ scale_inv,
                                       const int direction,
                                       const int group_size,
                                       const int n_factors,
                                       const int * __restrict__ factors,
                                       const int64_t n_groups,
                                       const int64_t head_dim,
                                       const int64_t groups_per_head) {
    const int64_t g = blockIdx.x;
    if (g >= n_groups) return;

    const int t = threadIdx.x;
    if (t >= group_size) return;

    const int64_t head_idx    = g / groups_per_head;
    const int64_t grp_in_head = g % groups_per_head;
    const int64_t base        = head_idx * head_dim + grp_in_head * group_size;

    extern __shared__ float s[];
    // s[0..group_size-1] = working buffer
    // s[group_size..2*group_size-1] = temp for p-ary stages

    // Load
    s[t] = src[base + t];
    __syncthreads();

    // InnerQ forward
    if (direction == 0 && scale_inv != nullptr) {
        s[t] *= scale_inv[t % group_size];
        __syncthreads();
    }

    // Mixed-radix butterfly stages
    int stride = 1;
    for (int f = 0; f < n_factors; f++) {
        const int p = factors[f];

        if (p == 2) {
            // Binary butterfly (same as WHT)
            const int block_sz = stride * 2;
            const int block_idx = t / block_sz;
            const int pos = t % block_sz;
            if (pos < stride) {
                const int i = block_idx * block_sz + pos;
                float a = s[i], b = s[i + stride];
                s[i]          = a + b;
                s[i + stride] = a - b;
            }
            __syncthreads();
        } else {
            // p-ary Hartley butterfly
            float * tmp = s + group_size; // temp buffer in shmem
            const int block_sz = stride * p;
            const int block_idx = t / block_sz;
            const int pos = t % block_sz;

            if (pos < stride) {
                const int j = pos;  // position within stride
                const int base_idx = block_idx * block_sz + j;

                // Gather p elements at stride-separated positions
                // Compute p-point DHT: result[k] = sum_m val[m] * cas(2π*k*m/p)
                for (int k = 0; k < p; k++) {
                    float sum = 0.0f;
                    for (int m = 0; m < p; m++) {
                        float theta = 2.0f * 3.14159265358979323846f * (float)(k * m) / (float)p;
                        sum += s[base_idx + m * stride] * (__cosf(theta) + __sinf(theta));
                    }
                    tmp[base_idx + k * stride] = sum;
                }
            }
            __syncthreads();

            // Copy back from temp
            if (t < group_size) {
                s[t] = tmp[t];
            }
            __syncthreads();
        }

        stride *= p;
    }

    // Normalize
    const float inv_sqrt = rsqrtf((float)group_size);
    float result = s[t] * inv_sqrt;

    // InnerQ inverse
    if (direction == 1 && scale_inv != nullptr) {
        result *= scale_inv[t % group_size];
    }

    dst[base + t] = result;
}

// ─── Simple copy kernel for tail elements (identity pass-through) ────────────

static __global__ void k_turbo_wht_copy_tail(const float * __restrict__ src,
                                              float * __restrict__ dst,
                                              int64_t n_heads,
                                              int64_t head_dim,
                                              int64_t tail_offset,
                                              int tail_size) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_heads * tail_size) return;

    const int64_t head_idx  = i / tail_size;
    const int64_t tail_elem = i % tail_size;
    const int64_t offset    = head_idx * head_dim + tail_offset + tail_elem;
    dst[offset] = src[offset];
}

// ─── Dispatch ─────────────────────────────────────────────────────────────────

void ggml_cuda_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    const ggml_tensor * scale_tensor = dst->src[1];  // InnerQ scale_inv (may be NULL)

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int direction;
    int group_size;
    memcpy(&direction, dst->op_params + 0, sizeof(int));
    memcpy(&group_size, dst->op_params + sizeof(int), sizeof(int));

    const int64_t head_dim        = src->ne[0];
    const int64_t n_heads         = ggml_nelements(src) / head_dim;

    GGML_ASSERT(group_size > 0);
    const int64_t groups_per_head = head_dim / group_size;
    const int     tail_size       = (int)(head_dim % group_size);
    const int64_t n_groups        = groups_per_head * n_heads;

    const float * src_ptr = (const float *) src->data;
    float       * dst_ptr = (float       *) dst->data;
    const float * scale_inv_ptr = scale_tensor ? (const float *) scale_tensor->data : nullptr;

    cudaStream_t stream = ctx.stream();

    const bool is_pow2 = (group_size & (group_size - 1)) == 0;

    // Process full groups
    if (n_groups > 0) {
        dim3 blocks(n_groups);

        if (is_pow2 && (group_size == 128 || group_size == 64)) {
            // Power-of-2: use templated WHT kernel (existing fast path)
            if (group_size == 128) {
                dim3 threads(128);
                if (direction == 0) {
                    k_turbo_wht_f32<0, 128><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
                } else {
                    k_turbo_wht_f32<1, 128><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
                }
            } else {
                dim3 threads(64);
                if (direction == 0) {
                    k_turbo_wht_f32<0, 64><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
                } else {
                    k_turbo_wht_f32<1, 64><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, scale_inv_ptr, n_groups, head_dim, groups_per_head);
                }
            }
        } else {
            // Non-power-of-2 (or non-standard power-of-2): Vilenkin-Hartley kernel
            // Compute prime factorization on host, upload to device
            int h_factors[20];
            int n_factors = 0;
            {
                int n = group_size;
                int d = 2;
                while (d * d <= n && n_factors < 20) {
                    while (n % d == 0) { h_factors[n_factors++] = d; n /= d; }
                    d++;
                }
                if (n > 1 && n_factors < 20) h_factors[n_factors++] = n;
            }

            // Upload factors to device (small, ~80 bytes max)
            int * d_factors = nullptr;
            CUDA_CHECK(cudaMalloc(&d_factors, n_factors * sizeof(int)));
            CUDA_CHECK(cudaMemcpyAsync(d_factors, h_factors, n_factors * sizeof(int),
                                        cudaMemcpyHostToDevice, stream));

            // Shared memory: 2 × group_size floats (working buffer + temp for p-ary stages)
            const size_t shmem = 2 * group_size * sizeof(float);

            // Round thread count up to next multiple of 32 (warp alignment)
            const int n_threads = ((group_size + 31) / 32) * 32;

            k_vilenkin_f32<<<blocks, n_threads, shmem, stream>>>(
                src_ptr, dst_ptr, scale_inv_ptr,
                direction, group_size, n_factors, d_factors,
                n_groups, head_dim, groups_per_head);

            CUDA_CHECK(cudaFree(d_factors));
        }
    }

    // Pass through tail elements unchanged (no rotation)
    // Not needed for 64-aligned dims but kept for completeness
    if (tail_size > 0) {
        const int64_t total_tail = n_heads * tail_size;
        const int block_sz = 256;
        const int n_blocks = (int)((total_tail + block_sz - 1) / block_sz);
        k_turbo_wht_copy_tail<<<n_blocks, block_sz, 0, stream>>>(
            src_ptr, dst_ptr, n_heads, head_dim, groups_per_head * group_size, tail_size);
    }
}
