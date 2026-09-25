# Worked Problem: Quantization Impact

## Problem Statement

You have a transformer model layer with weight matrix W of shape (4096, 4096) in FP16. The layer processes an input of shape (1, 4096) -- batch size 1 inference.

Compare the following quantization strategies:
1. FP16 (baseline)
2. INT8 per-tensor symmetric quantization
3. INT4 with group size 128
4. INT4 with group size 128 + 2:4 structured sparsity

For each, calculate: weight storage size, memory bandwidth per inference, effective throughput on a chip with 500 TFLOPS FP16 / 1000 TOPS INT8 / 2 TB/s HBM bandwidth.

---

## Worked Solution

### Step 1: Weight storage size

**FP16 baseline:**
```
Size = 4096 * 4096 * 2 bytes = 32 MiB (33.6 MB)
```

**INT8 per-tensor:**
```
Weights = 4096 * 4096 * 1 byte = 16 MiB
Scale = 1 * 4 bytes (FP32 scale) = 4 bytes (negligible)
Total = 16 MiB (16.8 MB)
```

**INT4 group-128:**
```
Weights = 4096 * 4096 * 0.5 bytes = 8 MiB
Scales = (4096 * 4096 / 128) * 2 bytes (FP16 scale per group) = 131,072 * 2 = 256 KiB
Total = 8.25 MiB (8.65 MB)
```

**INT4 group-128 + 2:4 sparsity:**
```
Nonzero weights = 4096 * 4096 * 0.5 (50% nonzero) * 0.5 bytes = 4 MiB
Sparsity metadata = 4096 * 4096 / 4 * 0.5 bytes = 2 MiB (4 bits per group of 4)
Scales = 256 KiB (same as above)
Total ≈ 6.25 MiB (6.55 MB)
```

### Step 2: Memory bandwidth per inference

For batch-1 matrix-vector multiply, the dominant cost is loading the weight matrix. The input vector (4096 * 2 bytes = 8 KB) and output vector are negligible.

| Strategy | Weight load | Bandwidth at 100 tok/s |
|---|---|---|
| FP16 | 33.6 MB | 3.36 GB/s |
| INT8 | 16.8 MB | 1.68 GB/s |
| INT4 g128 | 8.65 MB | 0.865 GB/s |
| INT4 g128 + 2:4 sparse | 6.55 MB | 0.655 GB/s |

### Step 3: Compute requirements

The layer computes output = input * W^T, which is a (1, 4096) * (4096, 4096) -> (1, 4096) operation.

**FLOPS (dense):**
```
FLOPS = 2 * 1 * 4096 * 4096 = 33,554,432 = 33.6 MFLOPS
```

**With 2:4 sparsity (50% of multiplies are skipped):**
```
Effective FLOPS = 33.6 / 2 = 16.8 MFLOPS
```

### Step 4: Determine bottleneck

For each strategy, compute the time for both memory transfer and compute:

**FP16:**
```
Memory time = 33.6 MB / 2 TB/s = 16.8 us
Compute time = 33.6M FLOPS / 500 TFLOPS = 0.067 us
Bottleneck: Memory (250x slower than compute)
Effective throughput: limited by memory -> 33.6M FLOPS / 16.8 us = 2.0 TFLOPS
```

**INT8:**
```
Memory time = 16.8 MB / 2 TB/s = 8.4 us
Compute time = 33.6M ops / 1000 TOPS = 0.034 us (INT8 compute is 2x faster)
Bottleneck: Memory (250x slower)
Effective throughput: 33.6M / 8.4 us = 4.0 TFLOPS equivalent
```

**INT4 group-128:**
```
Memory time = 8.65 MB / 2 TB/s = 4.33 us
Compute time ≈ 0.034 us (dequantize to FP16/INT8 then compute)
Bottleneck: Memory
Effective throughput: 33.6M / 4.33 us = 7.8 TFLOPS equivalent
```

**INT4 g128 + 2:4 sparse:**
```
Memory time = 6.55 MB / 2 TB/s = 3.28 us
Compute time ≈ 0.017 us (half the multiplies)
Bottleneck: Memory
Effective throughput: 33.6M / 3.28 us = 10.2 TFLOPS equivalent
```

### Step 5: Summary

| Strategy | Storage | Bandwidth | Per-layer time | Speedup vs FP16 |
|---|---|---|---|---|
| FP16 | 33.6 MB | 33.6 MB/token | 16.8 us | 1.0x |
| INT8 | 16.8 MB | 16.8 MB/token | 8.4 us | 2.0x |
| INT4 g128 | 8.65 MB | 8.65 MB/token | 4.33 us | 3.88x |
| INT4 g128 + 2:4 | 6.55 MB | 6.55 MB/token | 3.28 us | 5.12x |

### Key Insight

At batch size 1, all strategies are massively memory-bound. The speedup from quantization is almost exactly proportional to the compression ratio, because reducing weight size directly reduces the dominant cost (loading weights from HBM). The compute savings from sparsity and lower precision are irrelevant because compute is never the bottleneck.

This analysis explains why weight-only quantization is so effective for LLM inference: it directly addresses the memory bandwidth bottleneck.
