# Worked Problem: Inference Latency Optimization

## Problem Statement

You are optimizing the inference latency of a 7B parameter language model for interactive chat (target: under 50ms per token). The model runs on a single H100 GPU.

Model specs: 32 layers, D=4096, D_ff=11008, H=32, H_kv=32 (no GQA), D_h=128.

Current measurement: 28ms per token at batch size 1 with FP16 weights, sequence length 2048.

Evaluate the latency impact of:
1. INT8 weight quantization
2. INT4 weight quantization (group size 128)
3. Grouped-Query Attention (H_kv=8 instead of 32)
4. Combining INT4 + GQA

---

## Worked Solution

### Step 1: Analyze the baseline bottleneck

Weight size (FP16): 7B * 2 = 14 GB
KV cache at S=2048 (FP16): 2 * 32 * 32 * 128 * 2048 * 2 = 1.07 GB

Total memory read per token: 14 + 1.07 = 15.07 GB

H100 HBM3 bandwidth: 3.35 TB/s
Theoretical minimum time: 15.07 GB / 3350 GB/s = 4.5 ms

Measured: 28 ms. The gap (28 vs 4.5 ms) is due to:
- Kernel launch overhead (~0.5 ms per layer * 32 layers = 16 ms from small kernels)
- Memory access inefficiency (not achieving full HBM bandwidth)
- Softmax, normalization, and other non-GEMM operations
- Framework overhead

Let us model the per-token latency as:
```
Latency = weight_load_time + kv_cache_load_time + overhead
        = weights / bandwidth_eff + kv_cache / bandwidth_eff + overhead
```

With measured 28 ms and estimated overhead of ~10 ms:
```
18 ms = 15.07 GB / bandwidth_eff
bandwidth_eff = 15.07 / 0.018 = 837 GB/s (25% of peak -- realistic for small kernels)
```

### Step 2: INT8 weight quantization

Weight size: 7B * 1 = 7 GB (+ negligible scale storage)
KV cache: 1.07 GB (unchanged, still FP16)
Total: 8.07 GB

```
Memory load time = 8.07 / 837 = 9.64 ms
Total latency = 9.64 + 10 (overhead) = 19.6 ms
Speedup: 28 / 19.6 = 1.43x
```

### Step 3: INT4 weight quantization (group 128)

Weight size: 7B * 0.5 + 7B/128 * 2 (scales) = 3.5 + 0.109 = 3.61 GB
KV cache: 1.07 GB
Total: 4.68 GB

```
Memory load time = 4.68 / 837 = 5.59 ms
Dequantization overhead: ~1 ms (INT4 -> FP16 conversion)
Total latency = 5.59 + 1 + 10 = 16.6 ms
Speedup: 28 / 16.6 = 1.69x
```

### Step 4: Grouped-Query Attention (H_kv=8)

Weights: unchanged at 14 GB (for FP16 baseline). Actually, GQA reduces KV projection weights:
- Original K, V weights: 2 * D * D = 2 * 4096 * 4096 = 32M params per layer
- GQA K, V weights: 2 * D * (H_kv * D_h) = 2 * 4096 * 1024 = 8M params per layer
- Savings per layer: 24M * 2 bytes = 48 MB per layer, 32 layers = 1.5 GB weight savings
- Total weights: 14 - 1.5 = 12.5 GB

KV cache with GQA: 2 * 32 * 8 * 128 * 2048 * 2 = 0.268 GB (4x smaller)

Total: 12.5 + 0.268 = 12.77 GB

```
Memory load time = 12.77 / 837 = 15.26 ms
Total latency = 15.26 + 10 = 25.3 ms
Speedup: 28 / 25.3 = 1.11x
```

GQA alone provides modest improvement because the KV cache is only 7% of total memory load at S=2048. At S=32768, KV cache would be 17 GB (without GQA) or 4.3 GB (with GQA), making GQA much more impactful.

### Step 5: Combining INT4 + GQA

Weights (INT4): 3.61 GB (with GQA weight reduction: ~3.2 GB)
KV cache (GQA, FP16): 0.268 GB
Total: 3.47 GB

```
Memory load time = 3.47 / 837 = 4.14 ms
Dequantization: ~1 ms
Total latency = 4.14 + 1 + 10 = 15.1 ms
Speedup: 28 / 15.1 = 1.85x
```

### Step 6: Summary

| Configuration | Memory load | Total latency | Speedup |
|---|---|---|---|
| FP16 baseline | 15.07 GB | 28.0 ms | 1.0x |
| INT8 weights | 8.07 GB | 19.6 ms | 1.43x |
| INT4 (g128) | 4.68 GB | 16.6 ms | 1.69x |
| FP16 + GQA | 12.77 GB | 25.3 ms | 1.11x |
| INT4 + GQA | 3.47 GB | 15.1 ms | 1.85x |

All configurations meet the 50ms target. The combination of INT4 quantization and GQA provides the best latency.

### Key Insight

The fixed overhead (kernel launches, framework overhead) becomes an increasingly large fraction of total latency as memory load time decreases with quantization. Further latency reduction requires kernel fusion (reducing launch overhead), CUDA graph optimization (reducing framework overhead), and custom kernels (improving bandwidth utilization). This is why inference optimization for latency-critical applications requires both algorithmic (quantization, GQA) and systems (kernel optimization, batching) improvements.
