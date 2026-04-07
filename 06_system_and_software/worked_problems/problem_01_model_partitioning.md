# Worked Problem: Model Partitioning

## Problem Statement

You need to deploy a 405B parameter language model (Llama 3.1 405B) for inference with the following constraints:
- Model: 126 layers, D=16384, D_ff=53248, H=128 heads, H_kv=8 (GQA), D_h=128
- Target: serve at batch size 64, sequence length 4096, using BF16 weights
- Hardware: NVIDIA H100 GPUs with 80 GB HBM each, NVLink 900 GB/s intra-node (8 GPUs), InfiniBand 400 Gb/s inter-node

Determine: (1) minimum number of GPUs, (2) optimal parallelism strategy, (3) estimated per-token latency.

---

## Worked Solution

### Step 1: Memory requirements

**Weights (BF16)**: 405B * 2 bytes = 810 GB

**KV cache per request at S=4096**:
```
KV cache = 2 * 126 * 8 * 128 * 4096 * 2 = 2 * 126 * 8 * 128 * 4096 * 2 = 33.6 GB
```

For 64 concurrent requests: 64 * 33.6 = 2,150 GB = 2.1 TB

**Total memory**: 810 + 2150 = 2960 GB

### Step 2: Minimum GPU count

For memory: 2960 GB / 80 GB = 37 GPUs minimum. Round up to 40 (5 nodes of 8) or 48 (6 nodes of 8).

Actually, let us recalculate the KV cache more carefully:
```
Per layer: 2 * H_kv * D_h * S * 2 = 2 * 8 * 128 * 4096 * 2 = 16.78 MB per layer per request
126 layers: 126 * 16.78 = 2.11 GB per request
64 requests: 64 * 2.11 = 135 GB
```

Revised total: 810 + 135 = 945 GB. Need 945/80 = 12 GPUs minimum. Use 16 GPUs (2 nodes).

### Step 3: Parallelism strategy

With 16 GPUs across 2 nodes (8 per node with NVLink):

**Option A: TP=8, PP=2** (8-way tensor parallelism within each node, 2-way pipeline parallelism across nodes)
- Per-GPU weights: 810 / 16 = 50.6 GB
- Per-GPU KV cache: 135 / 8 = 16.9 GB (KV cache is replicated across PP stages but sharded across TP)
- Actually, with PP=2, each node handles 63 layers. KV cache per node: 64 * 63 * 16.78 MB = 67.5 GB. Per GPU: 67.5 / 8 = 8.44 GB.
- Per-GPU total: 50.6 + 8.44 = 59 GB. Fits in 80 GB.

**Option B: TP=16** (16-way tensor parallelism)
- Per-GPU weights: 810 / 16 = 50.6 GB
- Per-GPU KV cache: 135 / 16 = 8.44 GB
- Per-GPU total: 59 GB. Fits.
- Problem: TP=16 spans 2 nodes, requiring inter-node all-reduce every layer. With InfiniBand (50 GB/s), this is too slow.

Choose **Option A: TP=8, PP=2**.

### Step 4: Per-token latency estimate

**Compute per token (decode phase)**: Each layer involves GEMMs totaling approximately 2 * model_params_per_layer * 2 FLOPS per token.
```
Params per layer ≈ 405B / 126 = 3.21B
FLOPS per layer ≈ 2 * 3.21B * 2 = 12.9 TFLOPS (rough)
```

Wait, let us be precise. Per layer:
- QKV: 2 * (D*D + D*D_kv + D*D_kv) = 2 * (16384^2 + 16384*1024 + 16384*1024) = 2 * (268M + 16.8M + 16.8M) = 604M FLOPS
  - Actually with B=64: 2 * 64 * (16384*16384 + 16384*1024 + 16384*1024) = 2*64*302M = 38.7G FLOPS

For batch 64 decode, this is a (64, 16384) * (16384, 16384) GEMM -- not a matrix-vector anymore. This has reasonable arithmetic intensity.

Let us estimate total FLOPS per step (all 126 layers, batch 64):
```
Per layer FLOPS ≈ 2 * 64 * (3 * 16384 * D_kv_total + 16384^2 + 2 * 16384 * 53248)
= 2 * 64 * (3 * 16384 * 1024 + 268M + 2 * 16384 * 53248)
= 2 * 64 * (50.3M + 268M + 1744M)
= 2 * 64 * 2062M = 264G FLOPS per layer
Total: 126 * 264G = 33.3 TFLOPS
```

With 16 H100s at 50% utilization: 16 * 990 * 0.5 = 7920 TFLOPS sustained.
Compute time: 33.3T / 7920T = 4.2 ms

**Communication time (TP all-reduce per layer, within node)**:
Activation size per layer: 64 * 16384 * 2 = 2 MB
Ring all-reduce within 8 GPUs: 2 * (7/8) * 2 MB / 900 GB/s = 0.0039 ms per layer
Total TP communication: 126 * 0.0039 = 0.49 ms (negligible)

**PP communication (between nodes)**:
Activation transfer per micro-batch: 64 * 16384 * 2 = 2 MB
Over InfiniBand (50 GB/s): 2 MB / 50 GB/s = 0.04 ms
With PP bubble overhead (approximately 1/PP_stages = 50% for PP=2): adds ~4.2 ms of bubble.

### Step 5: Total estimated latency per token

```
Compute: ~4.2 ms
TP communication: ~0.5 ms
PP overhead: ~4.2 ms (pipeline bubble)
KV cache loading: ~0.5 ms (135 GB / 16 GPUs / decode_portion)
Total: ~9.4 ms per token ≈ ~106 tokens/second for 64 concurrent requests
Per-request throughput: 106/64 ≈ 1.7 tokens/second per request
```

This is on the low end for interactive serving. Optimizations: INT8 quantization (halves weights, doubles effective bandwidth), speculative decoding, or larger batch size to improve compute utilization.
