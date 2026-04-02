/*
 * Fused mul_mat_vec for TQ4_1S / TQ3_1S / TQ4_0 weight types.
 *
 * TQ4_1S/TQ3_1S: Pre-rotate activation + centroid lookup (scalar FMA path).
 * TQ4_0: Pre-rotate activation to q8_1 + dp4a (same as q4_0 but with WHT pre-rotation).
 */

#include "mmvq-tq.cuh"
#include "turbo-quant.cuh"

#define MMVQ_TQ_NWARPS 8

// ============================================================================
// TQ4_0: Pre-rotate activation to q8_1 + dp4a
// ============================================================================

// Pre-rotate activation and quantize to standard block_q8_1 layout.
// Each warp (32 threads) processes one 32-element block.
static __global__ void tq_prerotate_q8_1(
        const float * __restrict__ src,
        block_q8_1  * __restrict__ dst,
        const int n_elements) {

    const int block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    const int offset = block_idx * 32 + lane;
    if (offset >= n_elements) return;

    // Forward WHT via warp shuffle
    float val = src[offset];
    val *= TQ_WEIGHT_SIGNS[lane];

    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h);
        val = (lane & h) ? (o - val) : (val + o);
    }
    val *= 0.17677669529663688f;

    // Warp reduction: max abs for scale
    float amax = fabsf(val);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));

    // Warp reduction: sum for q4_0 offset correction
    float sum = val;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);

    const float d = amax / 127.0f;
    const float id = (d > 0.0f) ? 127.0f / amax : 0.0f;

    dst[block_idx].qs[lane] = (int8_t)roundf(val * id);
    if (lane == 0) {
        dst[block_idx].ds = make_half2(__float2half(d), __float2half(sum));
    }
}

// TQ4_0 dp4a kernel: q4_0-style nibble unpack + dp4a with q8_1 activation.
// Each lane processes entire blocks (strided), doing 8 dp4a calls per block.
static __global__ void mul_mat_vec_tq4_0_dp4a(
        const void      * __restrict__ vx,
        const block_q8_1 * __restrict__ vy_q8,
        float            * __restrict__ dst,
        const int ncols_x,
        const int nrows_x) {

    const int row  = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ4_0;
    const block_tq4_0 * x_row = ((const block_tq4_0 *) vx) + (int64_t)row * blocks_per_row;

    float sumf = 0.0f;

    // Each lane processes blocks strided by 32 (full warp)
    for (int ib = lane; ib < blocks_per_row; ib += WARP_SIZE) {
        const block_tq4_0 * blk = &x_row[ib];
        const float d_w = __half2float(blk->d);
        const block_q8_1 * a_blk = &vy_q8[ib];
        const float2 ds8 = __half22float2(a_blk->ds);

        int sumi = 0;

        // 4 int32 loads from q4 qs (16 bytes = 32 nibbles)
        // Each int32: low nibbles = elems j..j+3, high nibbles = elems j+16..j+19
        // Note: qs is at 2-byte alignment (after fp16 d), so use uint16 loads
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // Load 4 bytes via 2x uint16 (handles 2-byte alignment)
            const uint16_t * qs16 = (const uint16_t *)(blk->qs);
            const int v = (int)qs16[2*j] | ((int)qs16[2*j + 1] << 16);

            // Unpack: low nibbles = elements [j*4 .. j*4+3]
            //         high nibbles = elements [j*4+16 .. j*4+19]
            const int vi0 = (v >> 0) & 0x0F0F0F0F;
            const int vi1 = (v >> 4) & 0x0F0F0F0F;

            // Load matching q8_1 values
            const int u0 = ((const int *)(a_blk->qs))[j];     // elements [j*4 .. j*4+3]
            const int u1 = ((const int *)(a_blk->qs))[j + 4]; // elements [j*4+16 .. j*4+19]

            sumi = __dp4a(vi0, u0, sumi);
            sumi = __dp4a(vi1, u1, sumi);
        }

        // q4_0 offset correction: value = (nibble - 8) * d_w
        // dp4a computed: sum(nibble * q8), need to subtract 8 * sum(q8) * d_act
        // ds8.y = sum of original float values ≈ sum(q8) * d_act
        sumf += d_w * ((float)sumi * ds8.x - 8.0f * ds8.y);
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset);

    if (lane == 0) dst[row] = sumf;
}

// ============================================================================
// Fallback: V8 scalar kernel (pre-rotated float activation)
// ============================================================================

static __global__ void tq_prerotate_activation(
        const float * __restrict__ src,
        float       * __restrict__ dst,
        const int n_elements) {

    const int block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    const int offset = block_idx * 32 + lane;
    if (offset >= n_elements) return;

    float val = src[offset];
    val *= TQ_WEIGHT_SIGNS[lane];

    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h);
        val = (lane & h) ? (o - val) : (val + o);
    }
    val *= 0.17677669529663688f;
    dst[offset] = val;
}

static __global__ void mul_mat_vec_tq4_1s_fused(
        const void  * __restrict__ vx,
        const float * __restrict__ vy_rot,
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x) {

    const int row  = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ4_1S;
    const block_tq4_1s * x_row = ((const block_tq4_1s *) vx) + (int64_t)row * blocks_per_row;

    float sum = 0.0f;

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float act = vy_rot[ib * QK_TQ4_1S + lane];
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint8_t idx = (x_row[ib].qs[lane / 2] >> ((lane & 1) * 4)) & 0xF;
        sum += act * TQ4_CENTROIDS_WEIGHT[idx] * d;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);

    if (lane == 0) dst[row] = sum;
}

static __device__ __forceinline__ uint8_t tq3_extract_index(const uint8_t * __restrict__ qs, int lane) {
    const int group = lane / 8;
    const int lane_in_group = lane % 8;
    const uint8_t * qp = qs + group * 3;
    const uint32_t packed = (uint32_t)qp[0] | ((uint32_t)qp[1] << 8) | ((uint32_t)qp[2] << 16);
    return (packed >> (lane_in_group * 3)) & 7;
}

static __global__ void mul_mat_vec_tq3_1s_fused(
        const void  * __restrict__ vx,
        const float * __restrict__ vy_rot,
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x) {

    const int row  = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ3_0;
    const block_tq3_1s * x_row = ((const block_tq3_1s *) vx) + (int64_t)row * blocks_per_row;

    float sum = 0.0f;

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float act = vy_rot[ib * QK_TQ3_0 + lane];
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint8_t idx = tq3_extract_index(x_row[ib].qs, lane);
        sum += act * TQ3_CENTROIDS_WEIGHT[idx] * d;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);

    if (lane == 0) dst[row] = sum;
}

// ============================================================================
// Dispatch: try dp4a, fall back to V8 scalar
// ============================================================================

static float * d_act_buf = nullptr;
static size_t  d_act_buf_size = 0;
static block_q8_1 * d_q8_1_buf = nullptr;
static size_t  d_q8_1_buf_size = 0;

void ggml_cuda_mul_mat_vec_tq(ggml_backend_cuda_context & ctx,
                               const ggml_tensor * src0,
                               const ggml_tensor * src1,
                               ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_TQ4_1S || src0->type == GGML_TYPE_TQ3_1S || src0->type == GGML_TYPE_TQ4_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);

    const int ncols_x = src0->ne[0];
    const int nrows_x = src0->ne[1];
    GGML_ASSERT(ncols_x % 32 == 0);

    const void  * src0_d = src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    // Graph-safe scratch allocation
    cudaStreamCaptureStatus capture_status;
    cudaStreamIsCapturing(stream, &capture_status);

    if (capture_status == cudaStreamCaptureStatusNone) {
        const size_t act_needed = ncols_x * sizeof(float);
        if (act_needed > d_act_buf_size) {
            if (d_act_buf) cudaFree(d_act_buf);
            cudaMalloc(&d_act_buf, act_needed);
            d_act_buf_size = act_needed;
        }

        const int n_blocks = ncols_x / 32;
        const size_t q8_1_needed = n_blocks * sizeof(block_q8_1);
        if (q8_1_needed > d_q8_1_buf_size) {
            if (d_q8_1_buf) cudaFree(d_q8_1_buf);
            cudaMalloc(&d_q8_1_buf, q8_1_needed);
            d_q8_1_buf_size = q8_1_needed;
        }
    }

    if (src0->type == GGML_TYPE_TQ4_0) {
        // TQ4_0: dp4a path with q8_1 pre-rotated activation
        GGML_ASSERT(d_q8_1_buf != nullptr);
        const int n_blocks = ncols_x / 32;

        // Phase 1: Pre-rotate + q8_1 quantize
        {
            const int wpb = 4;
            const dim3 block(32, wpb);
            const dim3 grid((n_blocks + wpb - 1) / wpb);
            tq_prerotate_q8_1<<<grid, block, 0, stream>>>(src1_d, d_q8_1_buf, ncols_x);
        }

        // Phase 2: dp4a mmvq
        {
            const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
            const dim3 grid((nrows_x + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS);
            mul_mat_vec_tq4_0_dp4a<<<grid, block, 0, stream>>>(src0_d, d_q8_1_buf, dst_d, ncols_x, nrows_x);
        }
    } else {
        // TQ4_1S / TQ3_1S: V8 scalar path with float pre-rotated activation
        GGML_ASSERT(d_act_buf != nullptr);
        {
            const int n_blocks = ncols_x / 32;
            const int wpb = 4;
            const dim3 block(32, wpb);
            const dim3 grid((n_blocks + wpb - 1) / wpb);
            tq_prerotate_activation<<<grid, block, 0, stream>>>(src1_d, d_act_buf, ncols_x);
        }
        {
            const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
            const dim3 grid((nrows_x + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS);
            if (src0->type == GGML_TYPE_TQ4_1S) {
                mul_mat_vec_tq4_1s_fused<<<grid, block, 0, stream>>>(src0_d, d_act_buf, dst_d, ncols_x, nrows_x);
            } else {
                mul_mat_vec_tq3_1s_fused<<<grid, block, 0, stream>>>(src0_d, d_act_buf, dst_d, ncols_x, nrows_x);
            }
        }
    }
}

// ============================================================================
// Load-time conversion: TQ4_1S → q8_0
//
// Fused kernel: dequant TQ4_1S (centroid lookup + inverse WHT) → quantize q8_0.
// One warp (32 threads) per block of 32 elements.
// Used at model load to convert TQ4_1S weights to q8_0 in VRAM for dp4a decode.
// ============================================================================

static __global__ void k_convert_tq4_1s_to_q8_0(
        const block_tq4_1s * __restrict__ src,
        block_q8_0         * __restrict__ dst,
        const int n_blocks) {

    const int block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (block_idx >= n_blocks) return;

    const int lane = threadIdx.x;
    const block_tq4_1s * blk = &src[block_idx];

    // Step 1: Dequant — centroid lookup × half-block scale
    const float d_scale = (lane < 16) ? __half2float(blk->d0) : __half2float(blk->d1);
    const uint8_t idx = (blk->qs[lane / 2] >> ((lane & 1) * 4)) & 0xF;
    float val = TQ4_CENTROIDS_WEIGHT[idx] * d_scale;

    // Step 2: Inverse WHT via warp shuffle (same as dequant path)
    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h);
        val = (lane & h) ? (o - val) : (val + o);
    }
    val *= 0.17677669529663688f;  // 1/sqrt(32)
    val *= TQ_WEIGHT_SIGNS[lane];

    // Step 3: Quantize to q8_0 — find block amax, compute scale, round
    float amax = fabsf(val);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));

    const float d = amax / 127.0f;
    const float id = (d > 0.0f) ? 127.0f / amax : 0.0f;

    // Step 4: Write q8_0 block
    dst[block_idx].qs[lane] = (int8_t)roundf(val * id);
    if (lane == 0) {
        dst[block_idx].d = __float2half(d);
    }
}

void ggml_cuda_convert_tq4_1s_to_q8_0(const void * src_tq4, void * dst_q8, int64_t n_elements, cudaStream_t stream) {
    GGML_ASSERT(n_elements % QK_TQ4_1S == 0);
    const int n_blocks = n_elements / QK_TQ4_1S;

    const int wpb = 4;  // warps per CUDA block
    const dim3 block(32, wpb);
    const dim3 grid((n_blocks + wpb - 1) / wpb);

    k_convert_tq4_1s_to_q8_0<<<grid, block, 0, stream>>>(
        (const block_tq4_1s *)src_tq4,
        (block_q8_0 *)dst_q8,
        n_blocks);
}
