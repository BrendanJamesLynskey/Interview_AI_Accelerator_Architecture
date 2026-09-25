# Worked Problem: Workload Profiling

## Problem Statement

You are profiling the forward pass of a single transformer layer during inference with the following parameters:

- Model dimension D = 4096
- FFN hidden dimension D_ff = 16384 (4x model dimension)
- Number of attention heads H = 32
- Head dimension D_h = D / H = 128
- Sequence length S = 2048
- Batch size B = 1
- Data type: FP16 (2 bytes per element)

The hardware platform has:
- Peak FP16 compute: 500 TFLOPS
- HBM bandwidth: 2.0 TB/s
- On-chip SRAM: 40 MB

For each operation in the forward pass, calculate:
1. The number of FLOPS
2. The number of bytes moved (assuming no on-chip reuse, worst case)
3. The arithmetic intensity
4. Whether the operation is compute-bound or memory-bound
5. The estimated latency

---

## Worked Solution

### Step 1: Identify the operations and calculate ridge point

**Ridge point** = 500 TFLOPS / 2.0 TB/s = 250 FLOPS/byte

The forward pass of one transformer layer consists of:
1. QKV projection (three GEMMs combined)
2. Attention score computation (Q * K^T)
3. Softmax
4. Attention value aggregation (scores * V)
5. Output projection
6. FFN first linear layer
7. Activation function (GELU)
8. FFN second linear layer
9. Two layer norm operations and two residual additions

### Step 2: Analyze each operation

#### Operation 1: QKV Projection
Three weight matrices, each of shape (D, D) applied to input (B*S, D).
Often fused into a single (D, 3D) weight matrix.

```
FLOPS = 2 * B*S * D * 3*D = 2 * 2048 * 4096 * 12288 = 2.06 * 10^11
Bytes = (B*S*D + D*3*D + B*S*3*D) * 2 = (2048*4096 + 4096*12288 + 2048*12288) * 2
      = (8,388,608 + 50,331,648 + 25,165,824) * 2 = 167,772,160 bytes = 167.8 MB
AI = 2.06 * 10^11 / 1.68 * 10^8 = 1229 FLOPS/byte
```

**Compute-bound** (1229 >> 250). Note that $B = 1$ does not make this a matrix-vector multiply: with $S = 2048$ tokens in prefill, the $(2048, 4096)$ input is multiplied by the $(4096, 12288)$ weights, so the weight bytes are amortised over 2048 rows and are already counted above:
```
Input bytes: 2048 * 4096 * 2 = 16.8 MB
Weight bytes: 4096 * 12288 * 2 = 100.7 MB  
Output bytes: 2048 * 12288 * 2 = 50.3 MB
Total: 167.8 MB
FLOPS: 2.06 * 10^11
AI: 2.06 * 10^11 / 1.678 * 10^8 = 1229 FLOPS/byte
```

**Status: Compute-bound.** Latency = 2.06 * 10^11 / (500 * 10^12) = 0.412 ms

#### Operation 2: Attention Score Computation (Q * K^T)
Per head: Q_h is (S, D_h) = (2048, 128), K_h is (S, D_h) = (2048, 128).
Score = Q_h * K_h^T produces (S, S) = (2048, 2048). Done for all H=32 heads.

```
FLOPS = 2 * H * S * S * D_h = 2 * 32 * 2048 * 2048 * 128 = 3.44 * 10^10
Bytes = (H*S*D_h + H*S*D_h + H*S*S) * 2 = (32*2048*128 + 32*2048*128 + 32*2048*2048) * 2
      = (8,388,608 + 8,388,608 + 134,217,728) * 2 = 301,989,888 bytes = 302.0 MB
AI = 3.44 * 10^10 / 3.02 * 10^8 = 114 FLOPS/byte
```

**Memory-bound** (114 < 250). Latency = 3.02 * 10^8 / (2.0 * 10^12) = 0.151 ms (bandwidth-limited)

#### Operation 3: Softmax
Applied row-wise to H attention score matrices, each (S, S).

```
FLOPS = H * S * S * 5 = 32 * 2048 * 2048 * 5 = 6.71 * 10^8 (approx 5 ops per element: subtract max, exp, sum, divide, plus the max reduction)
Bytes = 2 * H * S * S * 2 = 2 * 32 * 2048 * 2048 * 2 = 536,870,912 bytes = 536.9 MB (read + write)
AI = 6.71 * 10^8 / 5.37 * 10^8 = 1.25 FLOPS/byte
```

**Memory-bound** (1.25 << 250). Latency = 5.37 * 10^8 / (2.0 * 10^12) = 0.269 ms

#### Operation 4: Attention Value Aggregation (scores * V)
Per head: scores_h is (S, S) = (2048, 2048), V_h is (S, D_h) = (2048, 128).
Output is (S, D_h) per head.

```
FLOPS = 2 * H * S * S * D_h = 3.44 * 10^10 (same as Q*K^T)
Bytes = (H*S*S + H*S*D_h + H*S*D_h) * 2 = (134,217,728 + 8,388,608 + 8,388,608) * 2 = 301,989,888 bytes
AI = 114 FLOPS/byte
```

**Memory-bound** (114 < 250). Latency = 0.151 ms

#### Operation 5: Output Projection
Weight matrix (D, D) applied to attention output (B*S, D).

```
FLOPS = 2 * B*S * D * D = 2 * 2048 * 4096 * 4096 = 6.87 * 10^10
Bytes = (2048*4096 + 4096*4096 + 2048*4096) * 2 = (8.4M + 16.8M + 8.4M) * 2 = 67.1 MB
AI = 6.87 * 10^10 / 6.71 * 10^7 = 1024 FLOPS/byte
```

**Compute-bound**. Latency = 6.87 * 10^10 / (500 * 10^12) = 0.137 ms

#### Operation 6: FFN First Linear Layer
Weight (D, D_ff) = (4096, 16384), input (B*S, D).

```
FLOPS = 2 * 2048 * 4096 * 16384 = 2.75 * 10^11
Bytes = (2048*4096 + 4096*16384 + 2048*16384) * 2 = (8.4M + 67.1M + 33.6M) * 2 = 218.1 MB
AI = 2.75 * 10^11 / 2.18 * 10^8 = 1261 FLOPS/byte
```

**Compute-bound**. Latency = 2.75 * 10^11 / (500 * 10^12) = 0.550 ms

#### Operation 7: GELU Activation
Element-wise on (B*S, D_ff) = (2048, 16384) tensor.

```
FLOPS = 2048 * 16384 * 8 = 2.68 * 10^8 (GELU requires ~8 ops: polynomial approximation)
Bytes = 2 * 2048 * 16384 * 2 = 134,217,728 bytes = 134.2 MB (read + write)
AI = 2.68 * 10^8 / 1.34 * 10^8 = 2.0 FLOPS/byte
```

**Memory-bound** (2.0 << 250). Latency = 1.34 * 10^8 / (2.0 * 10^12) = 0.067 ms

#### Operation 8: FFN Second Linear Layer
Weight (D_ff, D) = (16384, 4096), input (B*S, D_ff).

```
FLOPS = 2 * 2048 * 16384 * 4096 = 2.75 * 10^11
Bytes = (2048*16384 + 16384*4096 + 2048*4096) * 2 = (33.6M + 67.1M + 8.4M) * 2 = 218.1 MB
AI = 1261 FLOPS/byte
```

**Compute-bound**. Latency = 0.550 ms

#### Operation 9: Layer Norms (2x) and Residual Adds (2x)
Each operates on (B*S, D) = (2048, 4096) tensors.

```
LayerNorm FLOPS = 2048 * 4096 * 10 = 8.39 * 10^7 per instance (mean, variance, normalize, scale, shift)
Residual FLOPS = 2048 * 4096 = 8.39 * 10^6 per instance
Total FLOPS = 2 * 8.39 * 10^7 + 2 * 8.39 * 10^6 = 1.85 * 10^8
Total Bytes = 4 * 2 * 2048 * 4096 * 2 = 134,217,728 bytes (4 operations, read+write)
AI = 1.85 * 10^8 / 1.34 * 10^8 = 1.38 FLOPS/byte
```

**Memory-bound**. Latency = 1.34 * 10^8 / (2.0 * 10^12) = 0.067 ms

### Step 3: Summary table

| Operation | FLOPS | Bytes | AI (F/B) | Bound | Latency (ms) |
|---|---|---|---|---|---|
| QKV Projection | 2.06 * 10^11 | 167.8 MB | 1229 | Compute | 0.412 |
| Q * K^T | 3.44 * 10^10 | 302.0 MB | 114 | Memory | 0.151 |
| Softmax | 6.71 * 10^8 | 536.9 MB | 1.25 | Memory | 0.269 |
| Scores * V | 3.44 * 10^10 | 302.0 MB | 114 | Memory | 0.151 |
| Output Proj | 6.87 * 10^10 | 67.1 MB | 1024 | Compute | 0.137 |
| FFN Layer 1 | 2.75 * 10^11 | 218.1 MB | 1261 | Compute | 0.550 |
| GELU | 2.68 * 10^8 | 134.2 MB | 2.0 | Memory | 0.067 |
| FFN Layer 2 | 2.75 * 10^11 | 218.1 MB | 1261 | Compute | 0.550 |
| LN + Residual | 1.85 * 10^8 | 134.2 MB | 1.38 | Memory | 0.067 |
| **Total** | **8.94 * 10^11** | **2080 MB** | -- | -- | **2.354 ms** |

### Step 4: Key observations

1. **Compute-bound kernels** (QKV, Output Proj, FFN layers) account for 92% of the FLOPS but only 70% of the latency (1.649 ms).

2. **Memory-bound kernels** (attention, softmax, activation, normalization) account for 30% of the latency (0.705 ms) despite contributing only about 8% of total FLOPS.

3. **Operator fusion** could dramatically reduce the memory-bound overhead. FlashAttention-style fusion eliminates the separate softmax pass and the intermediate score materialization, potentially saving 0.269 ms (the softmax latency) and reducing the bytes for attention operations.

4. The **total compute utilization** across the full layer is: 8.94 * 10^11 FLOPS / (2.354 * 10^-3 s * 500 * 10^12 FLOPS/s) = 76%. The 24% lost is due to memory-bound operations where compute units sit idle.

5. Increasing batch size would **not** improve the arithmetic intensity of the attention operations (Q*K^T and scores*V): each sequence has its own Q, K, V and score matrices, so nothing is shared across the batch. Batching raises intensity only for the weight-sharing GEMMs, which are already compute-bound here. Fusion (FlashAttention) is the remedy for attention.
