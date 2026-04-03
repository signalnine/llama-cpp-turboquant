#!/usr/bin/env python3
"""Calibrate Vilenkin basis mask for a model's KV cache.

Runs a forward pass, captures K/V cache tensors, transforms via VHT,
and discovers the top-48 basis indices by energy across all positions.

Outputs a binary mask file that can be loaded at inference time via
ggml_vilenkin_set_basis_mask().

Usage:
    python3 scripts/vilenkin-calibrate.py <model_name_or_path> [--output mask.bin] [--n-coeffs 48]
"""

import argparse
import struct
import sys
import numpy as np


def vilenkin_hartley_transform(x):
    """VHT — matching the C implementation exactly."""
    n = len(x)
    if n <= 1:
        return x.copy()
    x = x.copy().astype(np.float64)

    def prime_factors(n):
        factors = []
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors

    factors = prime_factors(n)
    stride = 1
    for p in factors:
        if p == 2:
            for i in range(0, n, stride * 2):
                for j in range(stride):
                    a, b = x[i+j], x[i+j+stride]
                    x[i+j] = a + b
                    x[i+j+stride] = a - b
        else:
            for i in range(0, n, stride * p):
                for j in range(stride):
                    tmp = [x[i + j + k * stride] for k in range(p)]
                    for k in range(p):
                        s = 0.0
                        for m in range(p):
                            theta = 2.0 * np.pi * k * m / p
                            s += tmp[m] * (np.cos(theta) + np.sin(theta))
                        x[i + j + k * stride] = s
        stride *= p
    return x / np.sqrt(n)


def capture_kv_cache(model_name, prompt, device="cuda"):
    """Capture KV cache from a forward pass."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device, trust_remote_code=True
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print(f"  Prompt tokens: {inputs['input_ids'].shape[1]}", flush=True)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    past_kv = outputs.past_key_values
    K_tensors = []
    V_tensors = []

    if hasattr(past_kv, 'layers'):
        for layer in past_kv.layers:
            K_tensors.append(layer.keys[0].cpu().float().numpy())
            V_tensors.append(layer.values[0].cpu().float().numpy())
    elif hasattr(past_kv, 'key_cache'):
        for i in range(len(past_kv.key_cache)):
            K_tensors.append(past_kv.key_cache[i].squeeze(0).cpu().float().numpy())
            V_tensors.append(past_kv.value_cache[i].squeeze(0).cpu().float().numpy())
    else:
        for i in range(len(past_kv)):
            K_tensors.append(past_kv[i][0].squeeze(0).cpu().float().numpy())
            V_tensors.append(past_kv[i][1].squeeze(0).cpu().float().numpy())

    head_dim = K_tensors[0].shape[-1]
    n_layers = len(K_tensors)
    n_kv_heads = K_tensors[0].shape[0]
    seq_len = K_tensors[0].shape[1]

    print(f"  Captured: {n_layers} layers, {n_kv_heads} KV heads, seq_len={seq_len}, head_dim={head_dim}")

    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()

    return K_tensors, V_tensors, head_dim, n_layers, n_kv_heads, seq_len


def find_optimal_mask(K_tensors, V_tensors, head_dim, n_coeffs=48):
    """Find the top-N VHT basis indices by energy across all KV vectors."""
    print(f"\nCalibrating mask (n_coeffs={n_coeffs}, head_dim={head_dim})...", flush=True)

    # Accumulate energy per basis index across all vectors
    energy_accum = np.zeros(head_dim, dtype=np.float64)
    n_vectors = 0

    for layer_idx, (K, V) in enumerate(zip(K_tensors, V_tensors)):
        n_heads, seq_len, d = K.shape
        for h in range(n_heads):
            for s in range(seq_len):
                for vec in [K[h, s], V[h, s]]:
                    norm = np.linalg.norm(vec)
                    if norm < 1e-12:
                        continue
                    coeffs = vilenkin_hartley_transform(vec / norm)
                    energy_accum += coeffs ** 2
                    n_vectors += 1

        if (layer_idx + 1) % 4 == 0:
            print(f"  Processed {layer_idx + 1}/{len(K_tensors)} layers ({n_vectors} vectors)...", flush=True)

    # Top-N indices by accumulated energy
    top_indices = np.argsort(-energy_accum)[:n_coeffs]
    top_indices = np.sort(top_indices)  # sort for cache-friendly access

    # Report
    total_energy = np.sum(energy_accum)
    captured_energy = np.sum(energy_accum[top_indices])
    pct = captured_energy / total_energy * 100

    print(f"\n  Total vectors analyzed: {n_vectors}")
    print(f"  Top-{n_coeffs} indices capture {pct:.1f}% of total energy")
    print(f"  Indices: {top_indices.tolist()}")

    # Show per-index energy contribution
    top_energies = energy_accum[top_indices] / total_energy * 100
    print(f"  Top-5 indices by energy: ", end="")
    for i in range(min(5, len(top_indices))):
        print(f"[{top_indices[i]}]={top_energies[i]:.2f}%", end=" ")
    print()

    return top_indices.astype(np.uint16)


def save_mask(mask, output_path, head_dim):
    """Save mask as binary file: [uint16 head_dim] [uint16 n_coeffs] [uint16 × n_coeffs indices]"""
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<HH', head_dim, len(mask)))
        for idx in mask:
            f.write(struct.pack('<H', int(idx)))
    print(f"\nMask saved to {output_path} ({4 + len(mask) * 2} bytes)")


def verify_mask(K_tensors, V_tensors, mask, head_dim):
    """Verify reconstruction quality with the discovered mask."""
    print(f"\nVerifying reconstruction quality...", flush=True)

    n_coeffs = len(mask)
    cos_total = 0.0
    mse_total = 0.0
    n = 0

    for K, V in zip(K_tensors, V_tensors):
        n_heads, seq_len, d = K.shape
        for h in range(n_heads):
            for s in range(min(seq_len, 50)):  # sample first 50 positions
                for vec in [K[h, s], V[h, s]]:
                    norm = np.linalg.norm(vec)
                    if norm < 1e-12:
                        continue

                    x_unit = vec / norm
                    coeffs = vilenkin_hartley_transform(x_unit)

                    # Simulate encode: keep only mask positions
                    sparse = np.zeros(head_dim)
                    sparse[mask] = coeffs[mask]

                    # Simulate decode: inverse VHT
                    x_hat = vilenkin_hartley_transform(sparse) * norm

                    cos = np.dot(vec, x_hat) / (np.linalg.norm(vec) * np.linalg.norm(x_hat) + 1e-10)
                    mse = np.mean((vec - x_hat) ** 2)
                    cos_total += cos
                    mse_total += mse
                    n += 1

    print(f"  Vectors tested: {n}")
    print(f"  Mean cosine similarity: {cos_total / n:.5f}")
    print(f"  Mean MSE: {mse_total / n:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate Vilenkin basis mask")
    parser.add_argument("model", help="HuggingFace model name or path")
    parser.add_argument("--output", "-o", default="vilenkin_mask.bin", help="Output mask file")
    parser.add_argument("--n-coeffs", type=int, default=48, help="Number of coefficients to store")
    parser.add_argument("--prompt", default=None, help="Calibration prompt (longer = better)")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    if args.prompt is None:
        args.prompt = (
            "The history of artificial intelligence began in antiquity, with myths and stories "
            "of artificial beings endowed with intelligence. The seeds of modern AI were planted "
            "by philosophers who attempted to describe the process of human thinking as the "
            "mechanical manipulation of symbols. This work culminated in the invention of the "
            "programmable digital computer in the 1940s. " * 5
        )

    K, V, hd, nl, nkv, sl = capture_kv_cache(args.model, args.prompt, args.device)
    mask = find_optimal_mask(K, V, hd, args.n_coeffs)
    save_mask(mask, args.output, hd)
    verify_mask(K, V, mask, hd)


if __name__ == "__main__":
    main()
