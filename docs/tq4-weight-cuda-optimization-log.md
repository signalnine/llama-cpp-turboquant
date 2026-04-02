# TQ4 Weight Compression — CUDA Kernel Optimization Log

Hardware: NVIDIA GeForce RTX 5090 (SM 12.0, Blackwell), 32 GB VRAM
Model: Qwen2.5-7B-Instruct (7.62B params, head_dim=128)
Benchmark: `llama-bench -ngl 99 -p 0 -n 128` (decode-only, single-token generation)

## Context

TQ4_1S is a WHT-rotated 4-bit weight quantization format from the TurboQuant paper.
It achieves excellent quality by combining Walsh-Hadamard decorrelation with Lloyd-Max
optimal centroids for N(0,1) distributions. On Apple Silicon (Metal), the non-linear
centroid lookup has no speed penalty due to SIMD architecture. On CUDA, however, the
centroid lookup prevents use of dp4a integer dot product intrinsics, resulting in
significantly lower decode throughput than standard q4_0.

This log documents the systematic optimization effort to close the speed gap.

## Baseline

| Format | bpv | Block layout | PPL (wiki2) | Decode t/s |
|--------|-----|-------------|-------------|-----------|
| f16 | 16.0 | — | 7.301 | — |
| TQ4_1S | 5.0 | d0(fp16) + d1(fp16) + qs[16] (4-bit nibbles) = 20B/32elem | 7.599 | 20 (cuBLAS) |
| q4_0 | 4.5 | d(fp16) + qs[16] (4-bit interleaved) = 18B/32elem | 7.843 | 267 |

TQ4_1S cuBLAS path: dequant to fp16 → cuBLAS SGEMM. 20 t/s for decode (ne[1]=1).
q4_0 uses the mmvq template with dp4a: 267 t/s.

## Kernel Versions

### V1–V7: Early fused kernels (not detailed here)
Progression from naive CPU-like dequant to warp-level optimizations.

### V8: Scalar fused kernel with pre-rotated float activation — 69 t/s

The first major win. Instead of inverse-WHT on every weight block, pre-rotate the
activation vector once (WHT forward pass via warp shuffle), then the inner loop is
just `centroid[idx] * d * activation[lane]`.

```
Phase 1: tq_prerotate_activation — warp shuffle WHT on activation (once)
Phase 2: mul_mat_vec_tq4_1s_fused — 1 element/lane/block, float FMA
```

**Architecture:**
- 8 warps per block (MMVQ_TQ_NWARPS = 8), each warp handles one row
- 32 lanes per warp, each lane handles element `lane` of every block
- Pre-rotated activation stored as float in scratch buffer

**Result:** 69 t/s — **3.5x over cuBLAS**, but only 26% of q4_0 (267 t/s).

### V9–V16: Multi-row, tiled, WMMA attempts (all failed)

| Version | Approach | Result | Why it failed |
|---------|----------|--------|--------------|
| V9 | NR0=2 (2 rows per warp, shmem) | Regressed | `__syncthreads` overhead > row parallelism |
| V10 | NR0=2 (per-warp registers) | Regressed | Register spill from 2x accumulator state |
| V11 | NR0=4 (4 rows per block) | Regressed | Same issues at larger scale |
| V12 | Shared-memory centroid LUT | ~69 t/s | LUT build cost offsets shmem latency gain |
| V13 | Loop unroll ×4 | ~69 t/s | Compiler already unrolling optimally |
| V14 | WMMA (tensor cores) | 6 t/s | Setup overhead (shmem + sync) >> TC throughput for matvec |
| V15 | L2 prefetch hints | ~69 t/s | No measurable effect on RTX 5090 |
| V16 | __launch_bounds__ tuning | ~69 t/s | Occupancy 3 vs 1 no effect (memory bound) |

**Key insight:** CUDA multi-row approaches all fail because CUDA lacks Metal's implicit
SIMD-group synchronization. Every inter-lane communication requires either `__syncthreads`
(block-level) or explicit `__shfl_sync` (warp-level). The per-element centroid lookup
creates a serial dependency that can't be parallelized across rows efficiently.

### V17: dp4a attempt with pre-rotated q8 activation — 69 t/s

Hypothesis: quantize the pre-rotated activation to int8 (q8_1) and use dp4a.
Problem: the centroid lookup still produces floats, so the actual dot product is
`float(centroid[idx] * d) * float(q8_value) * float(d_activation)` — not dp4a.

```
Phase 1: tq_prerotate_q8 — WHT + quantize to custom q8 block {float d; int8_t qs[32]}
Phase 2: float centroid lookup × int8 activation (NOT actual dp4a)
```

**Result:** 69 t/s — identical to V8. The q8 activation saves nothing because the
centroid lookup is still float-domain.

### V18: True dp4a with per-block int8 centroid LUT — 46 t/s

Attempted real dp4a by building a 16-entry int8 LUT per half-block at runtime:
`lut[c] = round(centroid[c] * d_half * 127 / max_abs)`.

```
Per block:
  1. Half-block max reduction (5x __shfl_xor_sync)
  2. Build int8 LUT from centroids × scale
  3. Pack 4 int8 weight values via __shfl_sync gather
  4. __dp4a(w_packed, a_packed, sum)
```

**Result:** 46 t/s — **34% regression**. The per-block overhead of:
- Half-block max warp reduction (5 shuffles)
- Runtime int8 quantization (roundf per element)
- Warp-cooperative packing (8 shuffles per lane per block)
- 75% lane idle time (only lane 0 of each group-of-4 does dp4a)

...far outweighs the dp4a throughput benefit.

### V19: ILP 4× unroll with q8 activation — 70 t/s

Unroll 4 blocks per iteration to maximize instruction-level parallelism.
All 4 blocks load independently before accumulating.

**Result:** 70.4 t/s — negligible improvement (+0.9%). The compiler was already
optimizing the instruction schedule adequately.

### Centroid lookup ablation — 69 t/s

Replaced `TQ4_CENTROIDS_WEIGHT[idx] * d` with `(idx - 8) * d` (uniform dequant,
no table lookup) in the V8 scalar kernel.

**Result:** 69.3 t/s — **identical** to V8 with centroid lookup.

**This disproved our hypothesis.** The centroid lookup (constant memory indirect access)
is NOT the bottleneck. The actual bottleneck is:

1. **Activation bandwidth:** V8 reads activation as float32 (4 bytes/element).
   q4_0's kernel reads q8_1 activation (1 byte/element + scale). 4× difference.
2. **Arithmetic density:** V8 does 1 float FMA per element per thread.
   q4_0 packs 4 MACs into one dp4a instruction. 4× difference.

Combined, these explain the ~4× gap between V8 (69 t/s) and q4_0 (267 t/s).

### TQ4_0: WHT-rotated uniform q4_0 (new format) — 237 t/s

Created a new quantization type (GGML_TYPE_TQ4_0) that combines WHT rotation with
standard q4_0-style uniform quantization and interleaved nibble packing.

**Format:** Identical to q4_0 — `d(fp16) + qs[16]` = 18 bytes / 32 elements = 4.5 bpv.
Nibble packing: `qs[j] = elem_j | (elem_{j+16} << 4)` (same interleaving as q4_0).

**Quantization:** WHT forward rotation → find amax → uniform quantize to [0,15] → pack.
**Dequantization:** Unpack → `(nibble - 8) * d` → WHT inverse rotation.

**CUDA kernel:** Standard dp4a approach:
```
Phase 1: tq_prerotate_q8_1 — warp shuffle WHT + quantize to block_q8_1 (with sum for offset correction)
Phase 2: mul_mat_vec_tq4_0_dp4a — strided block processing, 8 dp4a calls per block
         dot = d_w * (dp4a_sum * d_act - 8 * sum_act)
```

Key implementation detail: q4_0's `qs` field is at byte offset 2 (after fp16 `d`),
so only 2-byte aligned. Must use `uint16_t*` loads (like `get_int_b2`), not `int*`
casts, to avoid misaligned address CUDA errors.

**Result:** 237 t/s — **3.4× faster than TQ4_1S**.

**Quality:** PPL 7.883 — 0.04 worse than plain q4_0 (7.843). The WHT rotation adds
no quality benefit with uniform quantization; it slightly hurts due to boundary
rounding effects. The decorrelation is only useful when paired with non-linear centroids.

## Final Comparison

All results on Qwen2.5-7B-Instruct, f16 source, wikitext-2-raw-v1 (20 chunks).

| Format | bpv | PPL | Δ PPL | Decode t/s | Notes |
|--------|-----|-----|-------|-----------|-------|
| f16 | 16.0 | 7.301 | — | — | baseline |
| **TQ4_1S** | **5.0** | **7.599** | **+0.298** | **69** | WHT + Lloyd-Max centroids |
| Q4_K_M | 4.8 | 7.648 | +0.347 | 247 | K-quant super-block + importance weights |
| Q4_K_S | 4.6 | 7.776 | +0.475 | 258 | K-quant super-block |
| q4_0 | 4.5 | 7.843 | +0.542 | 267 | uniform, dp4a |
| TQ4_0 | 4.5 | 7.883 | +0.582 | 237 | WHT + uniform (prototype) |

## Conclusions

1. **TQ4_1S achieves the best quality at 4-5 bpv** (7.599 PPL, +0.298 over f16) but
   the non-linear centroid lookup limits CUDA decode to 69 t/s regardless of kernel
   optimization. This is a fundamental architectural mismatch, not an optimization gap.

2. **Q4_K_M is the practical CUDA answer.** Only 0.05 PPL worse than TQ4_1S (7.648 vs
   7.599) but 3.6× faster (247 vs 69 t/s). The importance-weighted super-block approach
   achieves nearly equivalent quality with full dp4a compatibility.

3. **WHT rotation without non-linear centroids provides no quality benefit.** TQ4_0
   (WHT + uniform) scored 0.04 PPL worse than plain q4_0, confirming that the WHT
   decorrelation is only valuable when combined with distribution-matched centroids.

4. **The centroid lookup is not the bottleneck.** Replacing it with arithmetic (uniform
   dequant) produced identical speed. The real bottleneck is float32 activation bandwidth
   (4× vs q8_1) and float FMA arithmetic density (1× vs dp4a's 4×).

5. **TQ4_1S remains optimal for Metal/Apple Silicon** where the centroid lookup has no
   penalty due to SIMD architecture. The format should be recommended for Apple Silicon
   deployments and for users who prioritize quality over CUDA decode speed.

## Potential Future Directions

- **Affine codebook:** Constrain centroids to `a·idx + b` per half-block, enabling
  dp4a with a correction term. Quality between uniform and Lloyd-Max.
- **Per-super-block int8 codebook:** Amortize int8 LUT construction over 256 elements
  (IQ4_NL-style) instead of 32 to reduce overhead of V18 approach.
- **Runtime dequant at load:** Store TQ4_1S on disk (5.0 bpv), decompress to q8_0 in
  VRAM at model load. Exact TQ4_1S quality at q8_0 speed (~180 t/s), costs 8.5 bpv VRAM.
