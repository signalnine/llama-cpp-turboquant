/*
 * TurboQuant: KV cache compression via PolarQuant + QJL
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * Implements GGML_TYPE_TURBO2_0 (2-bit), GGML_TYPE_TURBO3_0 (3-bit) and
 * GGML_TYPE_TURBO4_0 (4-bit) for use as --cache-type-k turboN in llama-server.
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

/* Global: WHT group size for CPU quantize path (set by CPU SET_ROWS handler) */
int turbo3_cpu_wht_group_size = 0;

/* ---------- constants ---------- */

#define TURBO_SEED_ROTATION 42
#define TURBO_SEED_QJL      1042
#define TURBO_D             128  /* rotation group size = head_dim (independent of block size) */
#define TURBO_QJL_CONST     1.2533141373155003f  /* sqrt(pi/2) */

/* TURBO_D must match QK_TURBO3_GROUP from ggml-common.h — they represent
 * the same rotation group size but are defined separately. Guard against
 * silent divergence so GPU kernels and CPU reference stay in sync. */
static_assert(TURBO_D == QK_TURBO3_GROUP,
    "TURBO_D must equal QK_TURBO3_GROUP (rotation group size)");

/* Optimal centroids from paper (scaled by 1/sqrt(d)) */
/* 2-bit: {±0.453, ±1.51} / sqrt(d) */
static const float CENTROIDS_2BIT[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };

/* 3-bit: Lloyd-Max for N(0, 1/128), pre-computed */
static const float CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

/* ---------- rotation matrix (lazy init) ---------- */

static float turbo_rotation[TURBO_D * TURBO_D];
static float turbo_rotation_t[TURBO_D * TURBO_D]; /* transpose */
static int   turbo_rotation_initialized = 0;

/* Simple LCG PRNG for deterministic rotation generation */
static uint64_t turbo_prng_state;

static void turbo_prng_seed(uint64_t seed) {
    turbo_prng_state = seed;
}

static double turbo_prng_normal(void) {
    /* Box-Muller transform from uniform LCG */
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void turbo_init_rotation(void) {
    if (turbo_rotation_initialized) return;

    const int d = TURBO_D;

    /* Generate random Gaussian matrix directly into turbo_rotation.
     * Previous code used a 64KB stack-local G[128*128] then memcpy'd —
     * this segfaults on llama.cpp worker threads with reduced stack
     * sizes (512KB macOS, 64KB some Linux configs). Writing directly
     * into the static array avoids the stack allocation entirely. */
    turbo_prng_seed(TURBO_SEED_ROTATION);
    for (int i = 0; i < d * d; i++) {
        turbo_rotation[i] = (float)turbo_prng_normal();
    }

    /* QR decomposition via modified Gram-Schmidt */
    /* Q stored column-major in turbo_rotation */

    for (int j = 0; j < d; j++) {
        /* Normalize column j */
        float norm = 0.0f;
        for (int i = 0; i < d; i++) {
            norm += turbo_rotation[i * d + j] * turbo_rotation[i * d + j];
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + j] /= norm;
            }
        }

        /* Orthogonalize remaining columns against j */
        for (int k = j + 1; k < d; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) {
                dot += turbo_rotation[i * d + j] * turbo_rotation[i * d + k];
            }
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + k] -= dot * turbo_rotation[i * d + j];
            }
        }
    }

    /* Compute transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_rotation_t[i * d + j] = turbo_rotation[j * d + i];
        }
    }

    turbo_rotation_initialized = 1;
}

/* ---------- QJL projection matrix (lazy init, seed-based) ---------- */

static float turbo_qjl_matrix[TURBO_D * TURBO_D];
static float turbo_qjl_matrix_t[TURBO_D * TURBO_D];
static int   turbo_qjl_initialized = 0;

static void turbo_init_qjl(void) {
    if (turbo_qjl_initialized) return;

    const int d = TURBO_D;
    turbo_prng_seed(TURBO_SEED_QJL);

    for (int i = 0; i < d * d; i++) {
        turbo_qjl_matrix[i] = (float)turbo_prng_normal();
    }

    /* Transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_qjl_matrix_t[i * d + j] = turbo_qjl_matrix[j * d + i];
        }
    }

    turbo_qjl_initialized = 1;
}

/* ---------- helper: matrix-vector multiply ---------- */

static void matvec(const float * M, const float * x, float * y, int d) {
    /* y = M @ x, M is row-major d×d */
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += M[i * d + j] * x[j];
        }
        y[i] = sum;
    }
}

/* ---------- nearest centroid ---------- */

static int nearest_centroid_2bit(float val) {
    /* Binary search on midpoints: {-0.133, -0.040, 0.040, 0.133} */
    if (val < -0.086728f) return 0;       /* midpoint(-0.133, -0.040) */
    if (val <  0.000000f) return 1;       /* midpoint(-0.040, 0.040) */
    if (val <  0.086728f) return 2;       /* midpoint(0.040, 0.133) */
    return 3;
}

static int nearest_centroid_3bit(float val) {
    /* 8 centroids, find nearest via midpoints */
    if (val < -0.154259f) return 0;
    if (val < -0.091775f) return 1;
    if (val < -0.043589f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.043589f) return 4;
    if (val <  0.091775f) return 5;
    if (val <  0.154259f) return 6;
    return 7;
}

static int nearest_centroid_4bit(float val) {
    /* 16 centroids, optimal for N(0, 1/sqrt(128)), find nearest via midpoints */
    if (val < -0.145560f) return 0;
    if (val < -0.103361f) return 1;
    if (val < -0.079142f) return 2;
    if (val < -0.060009f) return 3;
    if (val < -0.043430f) return 4;
    if (val < -0.028293f) return 5;
    if (val < -0.013963f) return 6;
    if (val <  0.000000f) return 7;
    if (val <  0.013963f) return 8;
    if (val <  0.028293f) return 9;
    if (val <  0.043430f) return 10;
    if (val <  0.060009f) return 11;
    if (val <  0.079142f) return 12;
    if (val <  0.103361f) return 13;
    if (val <  0.145560f) return 14;
    return 15;
}

/* ---------- WHT sign arrays (must match CUDA/Metal, seed=42) ---------- */

static const float turbo_cpu_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static const float turbo_cpu_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

/* ---------- CPU forward WHT (in-place, group_size elements) ---------- */

static void turbo_cpu_fwht(float * x, int group_size) {
    const float * s1 = turbo_cpu_s1;
    const float * s2 = turbo_cpu_s2;
    const float inv_sqrt = (group_size == 128) ? 0.08838834764831845f : 0.125f;

    // signs1
    for (int i = 0; i < group_size; i++) x[i] *= s1[i];

    // butterfly stages
    for (int h = 1; h < group_size; h *= 2) {
        for (int i = 0; i < group_size; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // normalize + signs2
    for (int i = 0; i < group_size; i++) x[i] *= inv_sqrt * s2[i];
}

/* ---------- Vilenkin-Hartley Transform (mixed-radix, any dimension) ---------- */
/*
 * Generalizes WHT to non-power-of-2 dimensions via mixed-radix butterfly.
 * For power-of-2 dimensions, reduces exactly to WHT.
 *
 * Factorizes n into primes, applies one butterfly stage per factor:
 *   p=2: standard WHT butterfly (add/subtract)
 *   p>2: p-point Discrete Hartley Transform (cas kernel: cos+sin)
 *
 * The DHT is real-valued and self-inverse: VHT(VHT(x)) = x (after normalization).
 */

/* Prime factorization: fills factors[], returns count. Max 20 factors for any practical dim. */
static int vilenkin_prime_factors(int n, int factors[20]) {
    int count = 0;
    int d = 2;
    while (d * d <= n && count < 20) {
        while (n % d == 0) {
            factors[count++] = d;
            n /= d;
        }
        d++;
    }
    if (n > 1 && count < 20) factors[count++] = n;
    return count;
}

/* In-place Vilenkin-Hartley Transform. Self-inverse (with 1/sqrt(n) normalization). */
static void vilenkin_hartley_transform(float * x, int n) {
    if (n <= 1) return;

    int factors[20];
    int n_factors = vilenkin_prime_factors(n, factors);

    int stride = 1;
    for (int f = 0; f < n_factors; f++) {
        int p = factors[f];

        if (p == 2) {
            /* Binary butterfly — same as WHT stage */
            for (int i = 0; i < n; i += stride * 2) {
                for (int j = 0; j < stride; j++) {
                    float a = x[i + j], b = x[i + j + stride];
                    x[i + j]          = a + b;
                    x[i + j + stride] = a - b;
                }
            }
        } else {
            /* p-ary Hartley butterfly: cas(2*pi*k*m/p) = cos + sin */
            float tmp[16]; /* p <= 13 for any practical head_dim */
            assert(p <= 16);
            for (int i = 0; i < n; i += stride * p) {
                for (int j = 0; j < stride; j++) {
                    /* Gather p elements */
                    for (int k = 0; k < p; k++)
                        tmp[k] = x[i + j + k * stride];

                    /* Apply p-point DHT: result[k] = sum_m tmp[m] * cas(2*pi*k*m/p) */
                    for (int k = 0; k < p; k++) {
                        float sum = 0.0f;
                        for (int m = 0; m < p; m++) {
                            float theta = 2.0f * 3.14159265358979323846f * (float)(k * m) / (float)p;
                            sum += tmp[m] * (cosf(theta) + sinf(theta));
                        }
                        x[i + j + k * stride] = sum;
                    }
                }
            }
        }
        stride *= p;
    }

    /* Normalize */
    float inv_sqrt = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) x[i] *= inv_sqrt;
}

/* Forward Vilenkin rotation: signs1 → VHT → signs2 (or signs1 → WHT → signs2 for pow2) */
static void turbo_cpu_vilenkin_forward(float * x, int group_size) {
    const float * s1 = turbo_cpu_s1;
    const float * s2 = turbo_cpu_s2;

    /* Check if power-of-2 — use fast WHT path */
    if ((group_size & (group_size - 1)) == 0 && group_size <= 128) {
        turbo_cpu_fwht(x, group_size);
        return;
    }

    /* Non-power-of-2: generate signs from PRNG (same seed as CUDA/Metal would use) */
    /* For now, use identity signs for non-pow2 (no precomputed sign arrays for these dims) */
    /* TODO: generate signs from seed=42 for arbitrary group_size */

    vilenkin_hartley_transform(x, group_size);
}

/* Inverse Vilenkin rotation: signs2 → VHT → signs1 (self-inverse property) */
static void turbo_cpu_vilenkin_inverse(float * x, int group_size) {
    /* VHT is self-inverse, so inverse = forward with swapped sign arrays */
    if ((group_size & (group_size - 1)) == 0 && group_size <= 128) {
        /* Power-of-2: use existing inverse WHT path */
        const float * s1 = turbo_cpu_s1;
        const float * s2 = turbo_cpu_s2;
        const float inv_sqrt = 1.0f / sqrtf((float)group_size);

        /* signs2 first (inverse order) */
        for (int i = 0; i < group_size; i++) x[i] *= s2[i];

        /* WHT butterfly */
        for (int h = 1; h < group_size; h *= 2) {
            for (int i = 0; i < group_size; i += h * 2) {
                for (int j = i; j < i + h; j++) {
                    float a = x[j], b = x[j + h];
                    x[j]     = a + b;
                    x[j + h] = a - b;
                }
            }
        }

        /* normalize + signs1 */
        for (int i = 0; i < group_size; i++) x[i] *= inv_sqrt * s1[i];
        return;
    }

    /* Non-power-of-2: VHT is self-inverse */
    vilenkin_hartley_transform(x, group_size);
}

/* Public API for Vilenkin-Hartley Transform (called from ops.cpp) */
void ggml_vilenkin_hartley_transform(float * x, int n) {
    vilenkin_hartley_transform(x, n);
}

/* ---------- TURBO3_0: 3-bit PolarQuant with WHT rotation ---------- */

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);

    // Read WHT group size from global (set by CPU SET_ROWS handler before each call).
    // Fallback: 128 if row is 128-aligned, else 64.
    extern int turbo3_cpu_wht_group_size;
    int group_size = turbo3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) group_size = (group_size == 128) ? 64 : 128;
    assert(k % group_size == 0);

    const int n_groups = k / group_size;
    const int blocks_per_group = group_size / QK_TURBO3;

    for (int g = 0; g < n_groups; g++) {
        const float * grp_src = x + g * group_size;
        block_turbo3_0 * grp_dst = y + g * blocks_per_group;

        // 1. L2 norm over the group
        float norm_sq = 0.0f;
        float buf[128];  // max group_size
        for (int j = 0; j < group_size; j++) {
            buf[j] = grp_src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        // 2. Normalize
        for (int j = 0; j < group_size; j++) buf[j] *= inv_norm;

        // 3. Forward WHT rotation
        turbo_cpu_fwht(buf, group_size);

        // 4. Quantize + pack into sub-blocks
        float recon_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turbo3_0 * blk = &grp_dst[b];
            const int off = b * QK_TURBO3;

            memset(blk->qs, 0, QK_TURBO3 / 4);
            memset(blk->signs, 0, QK_TURBO3 / 8);

            for (int j = 0; j < QK_TURBO3; j++) {
                int idx = nearest_centroid_3bit(buf[off + j]);
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                if (idx & 0x4) {
                    blk->signs[j / 8] |= (1 << (j % 8));
                }
                recon_sq += CENTROIDS_3BIT[idx] * CENTROIDS_3BIT[idx];
            }
        }

        // 5. Corrected norm: grp_norm / recon_norm (matching CUDA kernel)
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            grp_dst[b].norm = GGML_FP32_TO_FP16(corrected);
        }
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    // Stub — Metal shader handles dequant on GPU.
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO3; j++) {
            uint8_t low2 = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            uint8_t hi1 = (x[block].signs[j/8] >> (j%8)) & 0x1;
            uint8_t idx = low2 | (hi1 << 2);
            y[block * QK_TURBO3 + j] = CENTROIDS_3BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);

    size_t row_size = (n_per_row / QK_TURBO3) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_0_ref(
            src + row * n_per_row,
            (block_turbo3_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO2_0: 2-bit PolarQuant (no QJL) ---------- */

void quantize_row_turbo2_0_ref(const float * GGML_RESTRICT x, block_turbo2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);

    extern int turbo3_cpu_wht_group_size;
    int group_size = turbo3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) group_size = (group_size == 128) ? 64 : 128;
    assert(k % group_size == 0);

    const int n_groups = k / group_size;
    const int blocks_per_group = group_size / QK_TURBO2;

    for (int g = 0; g < n_groups; g++) {
        const float * grp_src = x + g * group_size;
        block_turbo2_0 * grp_dst = y + g * blocks_per_group;

        /* 1. L2 norm over the group */
        float norm_sq = 0.0f;
        float buf[128];
        for (int j = 0; j < group_size; j++) {
            buf[j] = grp_src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        /* 2. Normalize */
        for (int j = 0; j < group_size; j++) buf[j] *= inv_norm;

        /* 3. Forward WHT rotation */
        turbo_cpu_fwht(buf, group_size);

        /* 4. Quantize + pack into sub-blocks */
        float recon_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turbo2_0 * blk = &grp_dst[b];
            const int off = b * QK_TURBO2;

            memset(blk->qs, 0, QK_TURBO2 / 4);

            for (int j = 0; j < QK_TURBO2; j++) {
                int idx = nearest_centroid_2bit(buf[off + j]);
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                recon_sq += CENTROIDS_2BIT[idx] * CENTROIDS_2BIT[idx];
            }
        }

        /* 5. Corrected norm */
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            grp_dst[b].norm = GGML_FP32_TO_FP16(corrected);
        }
    }
}

void dequantize_row_turbo2_0(const block_turbo2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);
    const int nb = k / QK_TURBO2;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO2; j++) {
            uint8_t idx = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            y[block * QK_TURBO2 + j] = CENTROIDS_2BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO2 == 0);

    size_t row_size = (n_per_row / QK_TURBO2) * sizeof(block_turbo2_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo2_0_ref(
            src + row * n_per_row,
            (block_turbo2_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO4_0: 3-bit PolarQuant + 1-bit QJL ---------- */

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();
    turbo_init_qjl();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        /* Step 1: Extract norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[TURBO_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, d * sizeof(float));
        }

        /* Step 2: Forward WHT rotation (matches CUDA set_rows) */
        float rotated[TURBO_D];
        memcpy(rotated, normalized, d * sizeof(float));
        turbo_cpu_fwht(rotated, d);

#if TURBO4_USE_4BIT
        /* Step 3: 4-bit quantization (16 centroids) */
        static const float CENTROIDS_4BIT[16] = {
            -0.173926f, -0.117195f, -0.089527f, -0.068756f,
            -0.051262f, -0.035597f, -0.020989f, -0.006938f,
             0.006938f,  0.020989f,  0.035597f,  0.051262f,
             0.068756f,  0.089527f,  0.117195f,  0.173926f
        };
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_4bit(rotated[i]);
        }

        /* Norm correction */
        float recon_norm_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            recon_norm_sq += CENTROIDS_4BIT[indices[i]] * CENTROIDS_4BIT[indices[i]];
        }
        float recon_norm = sqrtf(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? norm / recon_norm : norm;
        y[block].norm = GGML_FP32_TO_FP16(corrected_norm);
#else
        /* Step 3: 3-bit quantization (8 centroids) */
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_3bit(rotated[i]);
        }

        /* Step 4: Residual */
        float reconstructed[TURBO_D];
        for (int i = 0; i < d; i++) {
            reconstructed[i] = CENTROIDS_3BIT[indices[i]];
        }
        float mse_recon[TURBO_D];
        matvec(turbo_rotation_t, reconstructed, mse_recon, d);

        float residual[TURBO_D];
        for (int i = 0; i < d; i++) {
            residual[i] = normalized[i] - mse_recon[i];
        }

        /* Step 5: QJL */
        float projected[TURBO_D];
        matvec(turbo_qjl_matrix, residual, projected, d);
#endif

        /* Pack */
#if !TURBO4_USE_4BIT
        y[block].norm  = GGML_FP32_TO_FP16(norm);
#endif

#if TURBO4_USE_4BIT
        /* 4-bit PolarQuant: nibble pack into qs[64] */
        memset(y[block].qs, 0, d / 2);
        for (int i = 0; i < d; i++) {
            y[block].qs[i / 2] |= (uint8_t)((indices[i] & 0xF) << ((i % 2) * 4));
        }
        y[block].rnorm = GGML_FP32_TO_FP16(0.0f);
#else
        /* Legacy 3-bit + QJL: pack 3-bit indices + QJL signs */
        memset(y[block].qs, 0, d * 3 / 8);
        for (int i = 0; i < d; i++) {
            int bit_offset = i * 3;
            int byte_idx   = bit_offset / 8;
            int bit_pos    = bit_offset % 8;
            uint16_t val   = (uint16_t)(indices[i] & 0x7);
            y[block].qs[byte_idx] |= (uint8_t)(val << bit_pos);
            if (bit_pos > 5 && byte_idx + 1 < d * 3 / 8) {
                y[block].qs[byte_idx + 1] |= (uint8_t)(val >> (8 - bit_pos));
            }
        }
        memset(y[block].signs, 0, d / 8);
        for (int i = 0; i < d; i++) {
            if (projected[i] >= 0.0f) {
                y[block].signs[i / 8] |= (1 << (i % 8));
            }
        }
#endif
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

#if TURBO4_USE_4BIT
    /* 4-bit PolarQuant: nibble unpack → centroid → inverse rotate → scale */
    /* TODO: add proper 4-bit centroid table to C code (currently only in Metal) */
    static const float CENTROIDS_4BIT[16] = {
        -0.173926f, -0.117195f, -0.089527f, -0.068756f,
        -0.051262f, -0.035597f, -0.020989f, -0.006938f,
         0.006938f,  0.020989f,  0.035597f,  0.051262f,
         0.068756f,  0.089527f,  0.117195f,  0.173926f
    };
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        float * dst = y + block * d;
        for (int i = 0; i < d; i++) {
            uint8_t idx = (x[block].qs[i / 2] >> ((i % 2) * 4)) & 0xF;
            dst[i] = CENTROIDS_4BIT[idx] * norm;
        }
        /* No inverse WHT, dequant stays in the rotated domain.
        * Q is WHT-rotated by the graph, so <Q_rot, K_rot> gives correct attention scores.
        * The inverse WHT is applied to the attention output via GGML_OP_TURBO_WHT (direction=1) in the graph. 
        */
    }
#else
    /* Legacy 3-bit + QJL dequant */
    turbo_init_qjl();
    for (int block = 0; block < nb; block++) {
        float norm  = GGML_FP16_TO_FP32(x[block].norm);

        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            int bit_offset = i * 3;
            int byte_idx   = bit_offset / 8;
            int bit_pos    = bit_offset % 8;
            uint16_t raw   = (uint16_t)x[block].qs[byte_idx];
            if (byte_idx + 1 < d * 3 / 8) {
                raw |= (uint16_t)x[block].qs[byte_idx + 1] << 8;
            }
            indices[i] = (uint8_t)((raw >> bit_pos) & 0x7);
        }

        float signs[TURBO_D];
        for (int i = 0; i < d; i++) {
            signs[i] = (x[block].signs[i / 8] & (1 << (i % 8))) ? 1.0f : -1.0f;
        }

        float rnorm = GGML_FP16_TO_FP32(x[block].rnorm);
        const float qjl_scale = TURBO_QJL_CONST / (float)d * rnorm;

        float rotated_recon[TURBO_D];
        for (int i = 0; i < d; i++) {
            rotated_recon[i] = CENTROIDS_3BIT[indices[i]];
        }
        float mse_recon[TURBO_D];
        matvec(turbo_rotation_t, rotated_recon, mse_recon, d);

        float qjl_recon[TURBO_D];
        matvec(turbo_qjl_matrix_t, signs, qjl_recon, d);
        for (int i = 0; i < d; i++) {
            qjl_recon[i] *= qjl_scale;
        }

        float * dst = y + block * d;
        for (int i = 0; i < d; i++) {
            dst[i] = (mse_recon[i] + qjl_recon[i]) * norm;
        }
    }
#endif
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4 == 0);

    size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_ref(
            src + row * n_per_row,
            (block_turbo4_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ===================================================================
 * Vilenkin Coefficient Cache (GGML_TYPE_VILENKIN_3)
 *
 * Stores sparse VHT coefficients instead of quantized rotated values.
 * 48 coefficients × int4 + norm + scale = 28 bytes per 128 elements.
 * 1.75 bpv = 9.1× compression vs fp16.
 *
 * The shared basis mask (which 48 of 128 VHT indices to store) is
 * per-(layer,head) metadata set via ggml_vilenkin_set_basis_mask().
 * =================================================================== */

static uint16_t vk_basis_mask[VK_N_COEFFS] = {0};
static int      vk_mask_ready = 0;

static void vk_init_default_mask(int d) {
    for (int i = 0; i < VK_N_COEFFS && i < d; i++)
        vk_basis_mask[i] = (uint16_t)i;
    vk_mask_ready = 1;
}

void ggml_vilenkin_set_basis_mask(const uint16_t * mask, int n_coeffs) {
    int n = (n_coeffs < VK_N_COEFFS) ? n_coeffs : VK_N_COEFFS;
    for (int i = 0; i < n; i++) vk_basis_mask[i] = mask[i];
    for (int i = n; i < VK_N_COEFFS; i++) vk_basis_mask[i] = 0;
    vk_mask_ready = 1;
}

const uint16_t * ggml_vilenkin_get_basis_mask(void) {
    return vk_basis_mask;
}

/* Load mask from binary file: [uint16 head_dim] [uint16 n_coeffs] [uint16 × n_coeffs] */
int ggml_vilenkin_load_mask(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) return -1;

    uint16_t head_dim, n_coeffs;
    if (fread(&head_dim, sizeof(uint16_t), 1, f) != 1 ||
        fread(&n_coeffs, sizeof(uint16_t), 1, f) != 1) {
        fclose(f);
        return -2;
    }

    uint16_t buf[256];
    int n = (n_coeffs < VK_N_COEFFS) ? n_coeffs : VK_N_COEFFS;
    if ((int)fread(buf, sizeof(uint16_t), n, f) != n) {
        fclose(f);
        return -3;
    }
    fclose(f);

    ggml_vilenkin_set_basis_mask(buf, n);
    return n;
}

/* Encode: float → block_vilenkin_3 */
void quantize_row_vilenkin_3_ref(const float * GGML_RESTRICT x,
                                  block_vilenkin_3 * GGML_RESTRICT y,
                                  int64_t k) {
    assert(k % QK_VILENKIN_3 == 0);
    if (!vk_mask_ready) vk_init_default_mask(QK_VILENKIN_3);
    const int nb = k / QK_VILENKIN_3;

    for (int bi = 0; bi < nb; bi++) {
        const float * src = x + bi * QK_VILENKIN_3;

        float norm_sq = 0.0f;
        for (int i = 0; i < QK_VILENKIN_3; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);
        y[bi].norm = GGML_FP32_TO_FP16(norm);

        if (norm < 1e-12f) {
            y[bi].scale = GGML_FP32_TO_FP16(0.0f);
            memset(y[bi].coeffs, 0, VK_N_COEFFS / 2);
            continue;
        }

        /* Forward VHT on normalized vector */
        float buf[QK_VILENKIN_3];
        float inv_norm = 1.0f / norm;
        for (int i = 0; i < QK_VILENKIN_3; i++) buf[i] = src[i] * inv_norm;
        vilenkin_hartley_transform(buf, QK_VILENKIN_3);

        /* Extract coefficients at mask positions, find max */
        float raw[VK_N_COEFFS];
        float amax = 0.0f;
        for (int i = 0; i < VK_N_COEFFS; i++) {
            int idx = vk_basis_mask[i];
            raw[i] = (idx < QK_VILENKIN_3) ? buf[idx] : 0.0f;
            float a = fabsf(raw[i]);
            if (a > amax) amax = a;
        }

        /* Quantize to int4 [-8, 7] */
        float scale = amax / 7.0f;
        y[bi].scale = GGML_FP32_TO_FP16(scale);
        float inv_scale = (scale > 1e-12f) ? 7.0f / amax : 0.0f;

        memset(y[bi].coeffs, 0, VK_N_COEFFS / 2);
        for (int i = 0; i < VK_N_COEFFS; i++) {
            int q = (int)roundf(raw[i] * inv_scale);
            if (q < -8) q = -8;
            if (q >  7) q =  7;
            uint8_t uq = (uint8_t)(q + 8);
            if (i & 1)
                y[bi].coeffs[i / 2] |= (uq << 4);
            else
                y[bi].coeffs[i / 2] = uq;
        }
    }
}

/* Decode: block_vilenkin_3 → float */
void dequantize_row_vilenkin_3(const block_vilenkin_3 * GGML_RESTRICT x,
                                float * GGML_RESTRICT y,
                                int64_t k) {
    assert(k % QK_VILENKIN_3 == 0);
    if (!vk_mask_ready) vk_init_default_mask(QK_VILENKIN_3);
    const int nb = k / QK_VILENKIN_3;

    for (int bi = 0; bi < nb; bi++) {
        float norm  = GGML_FP16_TO_FP32(x[bi].norm);
        float scale = GGML_FP16_TO_FP32(x[bi].scale);

        /* Reconstruct sparse coefficient vector */
        float buf[QK_VILENKIN_3];
        memset(buf, 0, QK_VILENKIN_3 * sizeof(float));
        for (int i = 0; i < VK_N_COEFFS; i++) {
            uint8_t packed = x[bi].coeffs[i / 2];
            uint8_t uq = (i & 1) ? (packed >> 4) : (packed & 0xF);
            float val = ((int)uq - 8) * scale;
            int idx = vk_basis_mask[i];
            if (idx < QK_VILENKIN_3) buf[idx] = val;
        }

        /* Inverse VHT (self-inverse) */
        vilenkin_hartley_transform(buf, QK_VILENKIN_3);

        /* Rescale */
        float * dst = y + bi * QK_VILENKIN_3;
        for (int i = 0; i < QK_VILENKIN_3; i++) dst[i] = buf[i] * norm;
    }
}
