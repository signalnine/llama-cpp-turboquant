# Vilenkin-Hartley Transform for KV Cache Compression

Research proposal based on observations from ggml-org/llama.cpp#20969.

## Motivation

Every per-layer workaround in TurboQuant implementations points to the same gap:
the Walsh-Hadamard Transform (WHT) is a fixed base-2 decorrelation that doesn't
adapt to the actual structure of transformer hidden states. The empirical patches
we and others have built — InnerQ per-channel equalization, auto-asymmetric K/V,
first/last layer protection, outlier channel splitting — are all compensating for
structural information that WHT discards.

### Evidence from implementations

| Finding | What it implies |
|---------|----------------|
| K compresses 2-3x better than V | K carries structured (positional) information; V carries diffuse (content) information. A structure-aware transform should handle both. |
| First/last layers need protection | These are boundary layers of the recurrence. WHT treats all layers identically. |
| InnerQ calibration helps some models, hurts others | Per-channel scaling is a blunt proxy for structural adaptation. |
| Lloyd-Max centroids >> uniform | Post-WHT distribution is Gaussian, confirming WHT decorrelates but doesn't normalize. A better-matched transform might produce a distribution closer to uniform. |
| Larger WHT groups (128 vs 32) improve quality | More structure captured per group, but still base-2 only. |
| Per-layer sensitivity varies 10x across models | The mismatch between WHT's fixed structure and the model's actual structure varies per architecture. |

## The Proposal: Vilenkin System Transforms

### Background

The Walsh-Hadamard transform is the base-2 special case of the **Vilenkin system**
(also called Walsh-Vilenkin or generalized Walsh functions). Where WHT uses the
group Z_2^n (binary butterflies with ±1), the Vilenkin system uses Z_{p1} × Z_{p2} × ... × Z_{pk}
where p_i are primes.

- **WHT**: d = 2^n, butterfly stages with {-1, +1}, O(d log d)
- **Vilenkin**: d = p1 × p2 × ... × pk, butterfly stages with p-th roots of unity, O(d log d)
- **Hartley**: real-valued variant (cos + sin instead of complex exponentials)

The **Vilenkin-Hartley transform** combines both: a real-valued transform matched to
the multiplicative structure of the dimension, using the actual prime factorization
of the head dimension rather than forcing everything into base-2.

### Why this might work

1. **Dimension-matched factorization**: head_dim=128 = 2^7 (pure binary, WHT is optimal).
   But head_dim=80 = 2^4 × 5 (WHT can't handle the factor of 5 — current implementations
   pad to 128 or fall back to dense rotation). Vilenkin handles 80 natively as
   Z_2^4 × Z_5 with a 4-stage binary butterfly + 1-stage 5-ary butterfly.

2. **Adaptive decorrelation**: If transformer layers organize information along
   different prime-factor subspaces (as the discussion comment argues), a Vilenkin
   transform that matches the data's multiplicative structure would decorrelate
   more efficiently than fixed binary WHT.

3. **Eliminates per-layer hacks**: If the transform adapts to structure rather than
   treating everything as base-2, InnerQ calibration and auto-asymmetric might become
   unnecessary. This is testable.

### Specific predictions (testable)

1. **head_dim=80 models (Qwen3-4B)**: Vilenkin(2^4 × 5) should significantly
   outperform zero-padded WHT(128). Currently these models can't use TurboQuant
   at all in llama.cpp.

2. **Per-layer PPL variance**: If Vilenkin captures structure better, the PPL
   variance across layers should decrease (less need for per-layer adaptation).

3. **K vs V asymmetry**: If K carries positional (periodic/structured) information
   and Vilenkin captures periodic structure better, the K compression quality
   should improve most.

4. **Lloyd-Max vs uniform gap**: If Vilenkin produces a more uniform post-transform
   distribution, the gap between Lloyd-Max and uniform centroids should shrink —
   meaning dp4a-friendly uniform quantization would become more viable.

## Implementation Plan

### Phase 1: Python prototype (turboquant_plus)

Implement Vilenkin-Hartley transform in NumPy and compare against WHT on the
existing test suite.

```python
def vilenkin_transform(x, dim):
    """Vilenkin-Hartley transform for arbitrary dimensions.

    Factorizes dim into primes, applies mixed-radix butterfly.
    For dim=2^n, this reduces to standard WHT.
    For dim=80=2^4*5, this is a 4-stage binary + 1-stage 5-ary butterfly.
    """
    factors = prime_factorization(dim)  # e.g., [2,2,2,2,5] for dim=80

    for p in factors:
        # p-ary butterfly stage
        # For p=2: standard WHT butterfly (add/subtract)
        # For p=5: 5-point DFT butterfly (Rader's algorithm or direct)
        ...

    return x * (1.0 / sqrt(dim))
```

**Metrics:**
- MSE vs WHT at 2/3/4 bit across dimensions 64, 80, 128, 256
- Post-transform distribution shape (Gaussian? uniform? how close?)
- Per-layer PPL variance with and without InnerQ

### Phase 2: CPU implementation (ggml-turbo-quant.c)

Add `vilenkin_forward()` and `vilenkin_inverse()` alongside existing `tq3_0_rht_forward/inverse`.
Mixed-radix butterfly with compile-time factor tables for common head dims.

### Phase 3: CUDA kernel (if Phase 1 shows promise)

The binary butterfly stages use `__shfl_xor_sync` (same as current WHT).
The p-ary stages (p=3,5,7) use `__shfl_sync` with index arithmetic.
For head_dim=128 (pure binary), the kernel is identical to current WHT — no regression.

### Phase 4: Validation

- Full PPL sweep across model families (Llama, Qwen, DeepSeek, GLM)
- Test with and without InnerQ — if Vilenkin eliminates the need, remove it
- Compare head_dim=80 quality vs zero-padding approach
- Benchmark kernel speed (should be comparable to WHT for power-of-2 dims)

## Open Questions

1. **Is the prime factorization hypothesis actually correct?** The discussion
   comment asserts that transformer layers correspond to dimensional projections
   related to primes. This is a strong claim. Phase 1 will test it empirically.

2. **Sign patterns**: WHT uses random sign patterns (D1, D2 diagonal matrices)
   for the rotation. What's the Vilenkin analog? Random phases on Z_p?

3. **Self-inversive property**: WHT is self-inverse (H^T = H, up to scaling).
   Vilenkin-Hartley preserves this. Standard Vilenkin (complex) does not. We need
   the Hartley variant specifically.

4. **Codebook interaction**: If the post-transform distribution changes shape,
   the Lloyd-Max centroids need recomputation. The codebook.py infrastructure
   supports this (Lloyd's algorithm on empirical distribution).

## References

- Vilenkin, N. Ya. "On a class of complete orthonormal systems." Izv. Akad. Nauk SSSR, 1947.
- Fine, N. J. "On the Walsh functions." Trans. Amer. Math. Soc., 1949.
- Hartley, R. V. L. "A more symmetrical Fourier analysis applied to transmission problems." Proc. IRE, 1942.
- Zandieh et al. "TurboQuant: Online KV Cache Quantization via Rotation and Scalar Quantization." ICLR 2026.
- ggml-org/llama.cpp discussion #20969, comment by @[redacted] on structural interpretation.
